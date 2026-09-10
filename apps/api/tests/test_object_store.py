from pathlib import Path
from types import SimpleNamespace

from botocore.exceptions import ClientError

from app.services.object_store import ObjectStore


class _FakeClient:
    def __init__(self, buckets: list[str] | None = None) -> None:
        self.buckets = list(buckets or [])
        self.created: list[str] = []
        self.deleted_policies: list[str] = []
        self.access_blocks: list[str] = []

    def list_buckets(self) -> dict:
        return {"Buckets": [{"Name": bucket} for bucket in self.buckets]}

    def create_bucket(self, *, Bucket: str) -> None:
        self.buckets.append(Bucket)
        self.created.append(Bucket)

    def delete_bucket_policy(self, *, Bucket: str) -> None:
        self.deleted_policies.append(Bucket)
        raise ClientError(
            {
                "Error": {"Code": "NoSuchBucketPolicy", "Message": "No policy"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "DeleteBucketPolicy",
        )

    def put_public_access_block(self, *, Bucket: str, PublicAccessBlockConfiguration: dict) -> None:
        self.access_blocks.append(Bucket)


def _settings(tmp_path: Path, backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        storage_backend=backend,
        local_storage_root=tmp_path,
        s3_endpoint="http://minio:9000",
        s3_access_key="test",
        s3_secret_key="test-secret",
        s3_region="us-east-1",
        s3_secure=False,
        s3_bucket_temp="proctoring-temp",
        s3_bucket_evidence="proctoring-evidence",
        s3_bucket_reports="proctoring-reports",
    )


def test_minio_creates_all_buckets_without_public_access_block(tmp_path: Path) -> None:
    client = _FakeClient()
    store = ObjectStore(_settings(tmp_path, "minio"))
    store._client = client

    store.ensure_ready()

    assert client.created == list(store.buckets)
    assert client.deleted_policies == list(store.buckets)
    assert client.access_blocks == []


def test_s3_uses_public_access_block(tmp_path: Path) -> None:
    client = _FakeClient(["proctoring-temp", "proctoring-evidence", "proctoring-reports"])
    store = ObjectStore(_settings(tmp_path, "s3"))
    store._client = client

    store.ensure_ready()

    assert client.deleted_policies == []
    assert client.access_blocks == list(store.buckets)
