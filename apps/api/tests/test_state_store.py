from pathlib import Path

import pytest

from app.core.config import Settings
from app.services.state_store import StateStore


class _UnavailableLock:
    def acquire(self, blocking=True):
        return False


class _Redis:
    def lock(self, *args, **kwargs):
        return _UnavailableLock()


def test_production_session_lock_fails_closed(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        local_storage_root=tmp_path,
        storage_require_ready=True,
    )
    store = StateStore(settings)
    store._redis = _Redis()

    with pytest.raises(TimeoutError, match="session_lock_timeout"):
        with store.session_lock("session-1"):
            pass
