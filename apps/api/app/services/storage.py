from pathlib import Path
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
