from pathlib import Path
import base64
import hashlib
import json
import os
import shutil
import time
from uuid import uuid4

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.config import get_settings
from app.services.object_store import ObjectStore


class LocalStorage:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.objects = ObjectStore(self.settings)

    def save_identity_image(
        self,
        company_id: int,
        session_id: str,
        role: str,
        content: bytes,
        suffix: str = ".jpg",
    ) -> str:
        safe_session = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_"))[:128]
        safe_role = "live" if role == "live" else "reference"
        relative = Path("evidence") / str(company_id) / safe_session / "identity" / f"{safe_role}-{uuid4().hex}{suffix}"
        path = self.settings.local_storage_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return relative.as_posix()

    def save_face_reference(
        self,
        company_id: int,
        user_id: int,
        content: bytes,
        suffix: str = ".jpg",
    ) -> tuple[str, str]:
        reference_id = uuid4().hex
        key = f"references/{int(company_id)}/user_{int(user_id)}/reference-{reference_id}{suffix}.enc"
        encrypted = self._encrypt(content, f"reference:{company_id}:{user_id}".encode("utf-8"))
        self.objects.put_bytes(self.settings.s3_bucket_evidence, key, encrypted, "application/octet-stream")
        return reference_id, key

    def save_face_template(
        self,
        company_id: int,
        user_id: int,
        template: dict,
        best_reference_content: bytes | None = None,
        suffix: str = ".jpg",
    ) -> tuple[str, str, str | None]:
        template_id = uuid4().hex
        template_payload = dict(template)
        template_payload["templateId"] = template_id
        template_payload["companyId"] = int(company_id)
        template_payload["userId"] = int(user_id)
        prefix = f"references/{int(company_id)}/user_{int(user_id)}"
        template_key = f"{prefix}/template.json.enc"
        aad = f"template:{company_id}:{user_id}".encode("utf-8")
        encrypted_template = self._encrypt(
            json.dumps(template_payload, separators=(",", ":"), sort_keys=True).encode("utf-8"),
            aad,
        )
        self.objects.put_bytes(
            self.settings.s3_bucket_evidence,
            template_key,
            encrypted_template,
            "application/octet-stream",
            {"encryption": "aes-256-gcm", "key-version": "1"},
        )

        best_reference_key = None
        if best_reference_content is not None:
            best_reference_key = f"{prefix}/reference-best-{template_id}{suffix}.enc"
            encrypted_image = self._encrypt(
                best_reference_content,
                f"reference-image:{company_id}:{user_id}:{template_id}".encode("utf-8"),
            )
            self.objects.put_bytes(
                self.settings.s3_bucket_evidence,
                best_reference_key,
                encrypted_image,
                "application/octet-stream",
                {"encryption": "aes-256-gcm", "key-version": "1"},
            )

        return template_id, template_key, best_reference_key

    def latest_face_template(self, company_id: int, user_id: int) -> tuple[str, dict] | None:
        object_key = f"references/{int(company_id)}/user_{int(user_id)}/template.json.enc"
        if self.objects.exists(self.settings.s3_bucket_evidence, object_key):
            encrypted = self.objects.get_bytes(self.settings.s3_bucket_evidence, object_key)
            plaintext = self._decrypt(encrypted, f"template:{company_id}:{user_id}".encode("utf-8"))
            return object_key, json.loads(plaintext.decode("utf-8"))

        # Read-only migration support for references created before encryption.
        directory = self.settings.local_storage_root / "references" / str(company_id) / f"user_{int(user_id)}"
        template_path = directory / "template.json"
        if not template_path.exists():
            return None
        try:
            payload = json.loads(template_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        relative = template_path.relative_to(self.settings.local_storage_root).as_posix()
        return relative, payload

    def latest_face_reference(self, company_id: int, user_id: int) -> tuple[str, bytes] | None:
        prefix = f"references/{int(company_id)}/user_{int(user_id)}"
        encrypted = sorted(
            key for key in self.objects.list_keys(self.settings.s3_bucket_evidence, prefix)
            if "/reference-" in key and key.endswith(".enc")
        )
        if encrypted:
            key = encrypted[-1]
            template = self.latest_face_template(company_id, user_id)
            template_id = str((template or ("", {}))[1].get("templateId") or "")
            aad = (
                f"reference-image:{company_id}:{user_id}:{template_id}" if "reference-best-" in key
                else f"reference:{company_id}:{user_id}"
            ).encode("utf-8")
            return key, self._decrypt(self.objects.get_bytes(self.settings.s3_bucket_evidence, key), aad)

        directory = self.settings.local_storage_root / "references" / str(company_id) / f"user_{int(user_id)}"
        if not directory.exists():
            return None
        candidates = sorted(
            [path for path in directory.iterdir() if path.is_file() and path.name.startswith("reference-")],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        path = candidates[0]
        relative = path.relative_to(self.settings.local_storage_root).as_posix()
        return relative, path.read_bytes()

    def delete_face_reference(self, company_id: int, user_id: int) -> bool:
        prefix = f"references/{int(company_id)}/user_{int(user_id)}"
        deleted = False
        for key in list(self.objects.list_keys(self.settings.s3_bucket_evidence, prefix)):
            deleted = self.objects.delete(self.settings.s3_bucket_evidence, key) or deleted
        directory = self.settings.local_storage_root / "references" / str(company_id) / f"user_{int(user_id)}"
        if directory.exists():
            shutil.rmtree(directory)
            deleted = True
        return deleted

    def save_debug_artifact(self, relative_path: str, content: bytes) -> str:
        relative = Path(relative_path)
        path = self.settings.local_storage_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return relative.as_posix()

    def save_deletion_receipt(self, company_id: int, user_id: int, reason: str, deleted: bool) -> dict:
        receipt_id = uuid4().hex
        payload = {
            "receiptId": receipt_id,
            "companyId": int(company_id),
            "userId": int(user_id),
            "reason": str(reason)[:1000],
            "deleted": bool(deleted),
            "deletedAt": int(time.time()),
        }
        key = f"deletion-receipts/{int(company_id)}/user_{int(user_id)}/{receipt_id}.json"
        self.objects.put_bytes(
            self.settings.s3_bucket_reports,
            key,
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"),
            "application/json",
        )
        payload["receiptKey"] = key
        return payload

    def _encryption_key(self) -> bytes | None:
        key_file = str(self.settings.reference_encryption_key_file or "").strip()
        if not key_file:
            if self.settings.storage_require_ready:
                raise RuntimeError("REFERENCE_ENCRYPTION_KEY_FILE is required.")
            return None
        raw = Path(key_file).read_bytes().strip()
        candidates = [raw]
        try:
            candidates.append(bytes.fromhex(raw.decode("ascii")))
        except (ValueError, UnicodeDecodeError):
            pass
        try:
            candidates.append(base64.b64decode(raw, validate=True))
        except Exception:
            pass
        for candidate in candidates:
            if len(candidate) == 32:
                return candidate
        raise RuntimeError("The reference encryption key must contain exactly 32 bytes, hex, or base64.")

    def _encrypt(self, content: bytes, aad: bytes) -> bytes:
        key = self._encryption_key()
        if key is None:
            return content
        nonce = os.urandom(12)
        ciphertext = AESGCM(key).encrypt(nonce, content, aad)
        return b"PCG1" + nonce + ciphertext

    def _decrypt(self, content: bytes, aad: bytes) -> bytes:
        if not content.startswith(b"PCG1"):
            return content
        key = self._encryption_key()
        if key is None:
            raise RuntimeError("Encrypted reference cannot be read without its key.")
        return AESGCM(key).decrypt(content[4:16], content[16:], aad)
