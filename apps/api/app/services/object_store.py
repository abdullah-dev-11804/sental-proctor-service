from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

from botocore.client import BaseClient
from botocore.exceptions import ClientError

from app.core.config import Settings, get_settings


@dataclass(frozen=True)
class StoredObject:
    bucket: str
    key: str
    size: int
    content_type: str
    checksum: str | None = None


class ObjectStore:
    """Private object storage with an explicit local development fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.backend = self.settings.storage_backend.strip().lower()
        self.local_root = self.settings.local_storage_root / "objects"
        self.local_root.mkdir(parents=True, exist_ok=True)
        self._client: BaseClient | None = None

    @property
    def uses_s3(self) -> bool:
        return self.backend in {"s3", "minio"}

    @property
    def client(self) -> BaseClient:
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=self.settings.s3_endpoint,
                aws_access_key_id=self.settings.s3_access_key,
                aws_secret_access_key=self.settings.s3_secret_key,
                region_name=self.settings.s3_region,
                use_ssl=bool(self.settings.s3_secure),
            )
        return self._client

    def ensure_ready(self) -> None:
        if not self.uses_s3:
            return
        existing = {item["Name"] for item in self.client.list_buckets().get("Buckets", [])}
        for bucket in self.buckets:
            if bucket not in existing:
                self.client.create_bucket(Bucket=bucket)
            self.client.put_public_access_block(
                Bucket=bucket,
                PublicAccessBlockConfiguration={
                    "BlockPublicAcls": True,
                    "IgnorePublicAcls": True,
                    "BlockPublicPolicy": True,
                    "RestrictPublicBuckets": True,
                },
            )

    @property
    def buckets(self) -> tuple[str, str, str]:
        return (
            self.settings.s3_bucket_temp,
            self.settings.s3_bucket_evidence,
            self.settings.s3_bucket_reports,
        )

    def put_bytes(
        self,
        bucket: str,
        key: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> StoredObject:
        key = self._safe_key(key)
        if self.uses_s3:
            self.client.put_object(
                Bucket=bucket,
                Key=key,
                Body=io.BytesIO(content),
                ContentLength=len(content),
                ContentType=content_type,
                Metadata=metadata or {},
            )
        else:
            path = self.local_root / bucket / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return StoredObject(bucket=bucket, key=key, size=len(content), content_type=content_type)

    def put_file(
        self,
        bucket: str,
        key: str,
        source: str | Path,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> StoredObject:
        path = Path(source)
        if self.uses_s3:
            with path.open("rb") as handle:
                self.client.upload_fileobj(
                    handle,
                    bucket,
                    self._safe_key(key),
                    ExtraArgs={"ContentType": content_type, "Metadata": metadata or {}},
                )
            return StoredObject(bucket, self._safe_key(key), path.stat().st_size, content_type)
        return self.put_bytes(bucket, key, path.read_bytes(), content_type, metadata)

    def get_bytes(self, bucket: str, key: str) -> bytes:
        key = self._safe_key(key)
        if self.uses_s3:
            return self.client.get_object(Bucket=bucket, Key=key)["Body"].read()
        return (self.local_root / bucket / key).read_bytes()

    def download_file(self, bucket: str, key: str, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.uses_s3:
            self.client.download_file(bucket, self._safe_key(key), str(destination))
        else:
            destination.write_bytes(self.get_bytes(bucket, key))
        return destination

    def delete(self, bucket: str, key: str) -> bool:
        key = self._safe_key(key)
        if self.uses_s3:
            self.client.delete_object(Bucket=bucket, Key=key)
            return True
        path = self.local_root / bucket / key
        if not path.exists():
            return False
        path.unlink()
        return True

    def delete_prefix(self, bucket: str, prefix: str) -> int:
        keys = list(self.list_keys(bucket, prefix))
        if not keys:
            return 0
        if self.uses_s3:
            for offset in range(0, len(keys), 1000):
                self.client.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": [{"Key": key} for key in keys[offset:offset + 1000]], "Quiet": True},
                )
            return len(keys)
        deleted = 0
        for key in keys:
            deleted += 1 if self.delete(bucket, key) else 0
        return deleted

    def exists(self, bucket: str, key: str) -> bool:
        key = self._safe_key(key)
        if self.uses_s3:
            try:
                self.client.head_object(Bucket=bucket, Key=key)
                return True
            except ClientError as exc:
                if int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)) == 404:
                    return False
                raise
        return (self.local_root / bucket / key).is_file()

    def list_keys(self, bucket: str, prefix: str) -> Iterable[str]:
        prefix = self._safe_key(prefix).rstrip("/") + "/"
        if self.uses_s3:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    yield str(item["Key"])
            return
        root = self.local_root / bucket
        target = root / prefix
        if target.exists():
            for path in target.rglob("*"):
                if path.is_file():
                    yield path.relative_to(root).as_posix()

    def status(self) -> dict:
        try:
            self.ensure_ready()
            return {"backend": self.backend, "ready": True, "buckets": list(self.buckets), "private": True}
        except Exception as exc:
            return {"backend": self.backend, "ready": False, "error": str(exc)[:300]}

    @staticmethod
    def _safe_key(value: str) -> str:
        parts = []
        for part in str(value).replace("\\", "/").split("/"):
            if not part or part in {".", ".."}:
                continue
            safe = "".join(ch for ch in part if ch.isalnum() or ch in "-_.")
            if safe:
                parts.append(safe[:255])
        if not parts:
            raise ValueError("invalid_object_key")
        return "/".join(parts)
