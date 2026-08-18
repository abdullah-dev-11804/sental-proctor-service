from pathlib import Path
import json
import shutil
from uuid import uuid4

from app.core.config import get_settings


class LocalStorage:
    def __init__(self) -> None:
        self.settings = get_settings()

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
        relative = (
            Path("references")
            / str(company_id)
            / f"user_{int(user_id)}"
            / f"reference-{reference_id}{suffix}"
        )
        path = self.settings.local_storage_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return reference_id, relative.as_posix()

    def save_face_template(
        self,
        company_id: int,
        user_id: int,
        template: dict,
        best_reference_content: bytes | None = None,
        suffix: str = ".jpg",
    ) -> tuple[str, str, str | None]:
        template_id = uuid4().hex
        directory = self.settings.local_storage_root / "references" / str(company_id) / f"user_{int(user_id)}"
        directory.mkdir(parents=True, exist_ok=True)

        template_path = directory / "template.json"
        template_payload = dict(template)
        template_payload["templateId"] = template_id
        template_payload["companyId"] = int(company_id)
        template_payload["userId"] = int(user_id)
        template_path.write_text(
            json.dumps(template_payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

        best_reference_key = None
        if best_reference_content is not None:
            best_reference_path = directory / f"reference-best-{template_id}{suffix}"
            best_reference_path.write_bytes(best_reference_content)
            best_reference_key = best_reference_path.relative_to(self.settings.local_storage_root).as_posix()

        return template_id, template_path.relative_to(self.settings.local_storage_root).as_posix(), best_reference_key

    def latest_face_template(self, company_id: int, user_id: int) -> tuple[str, dict] | None:
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
        directory = self.settings.local_storage_root / "references" / str(company_id) / f"user_{int(user_id)}"
        if not directory.exists():
            return False
        shutil.rmtree(directory)
        return True

    def save_debug_artifact(self, relative_path: str, content: bytes) -> str:
        relative = Path(relative_path)
        path = self.settings.local_storage_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return relative.as_posix()
