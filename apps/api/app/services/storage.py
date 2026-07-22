from pathlib import Path
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
