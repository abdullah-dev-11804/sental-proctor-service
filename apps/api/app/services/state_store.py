from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.core.config import Settings, get_settings


class StateStore:
    """Redis active-state registry with durable local manifest compatibility."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.sessions_root = self.settings.local_storage_root / "media" / "sessions"
        self.asset_index_root = self.settings.local_storage_root / "media" / "asset-index"
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.asset_index_root.mkdir(parents=True, exist_ok=True)
        self._redis = None

    @property
    def redis(self):
        if self._redis is None:
            from redis import Redis

            self._redis = Redis.from_url(self.settings.redis_url, decode_responses=True, socket_timeout=2)
        return self._redis

    def redis_ready(self) -> bool:
        try:
            return bool(self.redis.ping())
        except Exception:
            return False

    def get_session(self, session_id: str) -> dict[str, Any]:
        safe_id = self._safe(session_id)
        try:
            raw = self.redis.get(f"proctorcore:session:{safe_id}")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        path = self.sessions_root / f"{safe_id}.json"
        if not path.exists():
            raise KeyError("session_not_found")
        return json.loads(path.read_text(encoding="utf-8"))

    def save_session(self, session: dict[str, Any]) -> None:
        session_id = self._safe(str(session["id"]))
        encoded = json.dumps(session, indent=2, sort_keys=True, ensure_ascii=False)
        path = self.sessions_root / f"{session_id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
        try:
            self.redis.set(f"proctorcore:session:{session_id}", encoded)
        except Exception:
            pass

    def index_asset(self, asset: dict[str, Any]) -> None:
        asset_id = self._safe(str(asset["assetId"]))
        encoded = json.dumps(asset, sort_keys=True, ensure_ascii=False)
        (self.asset_index_root / f"{asset_id}.json").write_text(encoded, encoding="utf-8")
        try:
            self.redis.set(f"proctorcore:asset:{asset_id}", encoded)
        except Exception:
            pass

    def get_asset(self, asset_id: str) -> dict[str, Any]:
        safe_id = self._safe(asset_id)
        try:
            raw = self.redis.get(f"proctorcore:asset:{safe_id}")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        path = self.asset_index_root / f"{safe_id}.json"
        if not path.exists():
            raise KeyError("asset_not_found")
        return json.loads(path.read_text(encoding="utf-8"))

    def delete_asset_index(self, asset_id: str) -> None:
        safe_id = self._safe(asset_id)
        (self.asset_index_root / f"{safe_id}.json").unlink(missing_ok=True)
        try:
            self.redis.delete(f"proctorcore:asset:{safe_id}")
        except Exception:
            pass

    @contextmanager
    def session_lock(self, session_id: str, timeout: int = 15) -> Iterator[None]:
        lock = None
        try:
            lock = self.redis.lock(f"proctorcore:lock:session:{self._safe(session_id)}", timeout=timeout, blocking_timeout=5)
            if lock.acquire(blocking=True):
                yield
                return
        except Exception:
            pass
        finally:
            try:
                if lock is not None and lock.owned():
                    lock.release()
            except Exception:
                pass
        yield

    @staticmethod
    def _safe(value: str) -> str:
        cleaned = "".join(ch for ch in str(value) if ch.isalnum() or ch in "-_.")
        if not cleaned:
            raise ValueError("invalid_identifier")
        return cleaned[:160]
