from __future__ import annotations

import base64
import hashlib
import hmac
import json
import mimetypes
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from app.core.config import get_settings


def _now() -> int:
    return int(time.time())


def _safe(value: str, limit: int = 128) -> str:
    cleaned = "".join(ch for ch in str(value) if ch.isalnum() or ch in ("-", "_", "."))
    return (cleaned or uuid4().hex)[:limit]


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


class MediaStore:
    """File-backed media/session coordinator.

    This is the first production-shaped media backend. It keeps temporary
    browser chunks while the session is active, materializes key-moment clips,
    and notifies Moodle through the signed webhook contract.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.root = self.settings.local_storage_root / "media"
        self.sessions_root = self.root / "sessions"
        self.assets_root = self.root / "assets"
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.assets_root.mkdir(parents=True, exist_ok=True)

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(uuid4())
        now = _now()
        session = {
            "id": session_id,
            "sessionId": session_id,
            "roomId": f"room-{session_id}",
            "status": "created",
            "companyId": int(payload.get("companyId") or payload.get("company_id") or 0),
            "moodleSessionId": payload.get("moodleSessionId") or payload.get("moodle_session_id"),
            "attemptId": payload.get("attemptId") or payload.get("quiz_attempt_id"),
            "quizId": payload.get("quizId") or payload.get("quiz_id"),
            "courseId": payload.get("courseId") or payload.get("course_id"),
            "userId": self._payload_user_id(payload),
            "callbackUrl": payload.get("callbackUrl") or self.settings.moodle_webhook_url,
            "returnUrl": payload.get("returnUrl"),
            "createdAt": now,
            "startedAt": None,
            "completedAt": None,
            "lastHeartbeatAt": None,
            "uploadTokenHash": None,
            "recording": {
                "state": "not_started",
                "currentSegment": 0,
                "segments": {},
            },
            "chunks": [],
            "pendingClips": [],
            "assets": [],
            "violations": [],
            "source": payload,
        }
        self._save_session(session)
        return session

    def get_session(self, session_id: str) -> dict[str, Any]:
        path = self._session_path(session_id)
        if not path.exists():
            raise KeyError("session_not_found")
        return json.loads(path.read_text(encoding="utf-8"))

    def start_session(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        now = _now()
        session["status"] = "active"
        session["startedAt"] = session.get("startedAt") or now
        session["lastHeartbeatAt"] = now
        session["lastStartPayload"] = payload
        self._save_session(session)
        return session

    def heartbeat(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        session["lastHeartbeatAt"] = _now()
        session["lastHeartbeatPayload"] = payload
        self._save_session(session)
        return session

    def create_media_token(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        token = self._livekit_token(
            room=str(session.get("roomId") or f"room-{session_id}"),
            identity=str(payload.get("participantIdentity") or f"user-{session.get('userId') or 'unknown'}"),
            name=str(payload.get("participantName") or ""),
        )
        upload_token = uuid4().hex + uuid4().hex
        session["uploadTokenHash"] = hashlib.sha256(upload_token.encode("utf-8")).hexdigest()
        session["uploadTokenExpiresAt"] = _now() + int(self.settings.media_upload_token_ttl_seconds)
        self._save_session(session)
        return {
            "ok": True,
            "url": self.settings.livekit_url,
            "token": token,
            "roomId": session.get("roomId"),
            "clientScriptUrl": self.settings.livekit_client_script_url,
            "tokenExpiresAt": _now() + 3600,
            "uploadUrl": f"/api/v1/sessions/{session_id}/media/chunks",
            "uploadToken": upload_token,
            "chunkMilliseconds": 5000,
        }

    def start_recording(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        recording = self._recording(session)
        if recording.get("state") == "active":
            return {
                "ok": True,
                "status": "active",
                "recordingId": recording.get("recordingId"),
                "segment": recording.get("currentSegment") or 1,
                "duplicate": True,
            }
        segment = int(payload.get("segment") or recording.get("currentSegment") or 0) or 1
        recording_id = f"rec-{session_id}-{segment}"
        now = _now()
        recording["state"] = "active"
        recording["currentSegment"] = segment
        recording["recordingId"] = recording_id
        recording.setdefault("segments", {})[str(segment)] = {
            "segment": segment,
            "recordingId": recording_id,
            "startedAt": now,
            "stoppedAt": None,
            "reason": payload.get("reason") or "attempt_page_connected",
        }
        session["recording"] = recording
        session["status"] = "active"
        session["startedAt"] = session.get("startedAt") or now
        self._save_session(session)
        return {"ok": True, "status": "active", "recordingId": recording_id, "segment": segment}

    def stop_recording(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        recording = self._recording(session)
        segment = int(payload.get("segment") or recording.get("currentSegment") or 1)
        now = _now()
        recording["state"] = "stopped"
        segment_entry = recording.setdefault("segments", {}).setdefault(str(segment), {"segment": segment})
        segment_entry["stoppedAt"] = now
        segment_entry["stopReason"] = payload.get("reason") or "submitted"
        session["recording"] = recording

        for clip in list(session.get("pendingClips") or []):
            if not clip.get("assetId"):
                asset = self._materialize_clip(
                    session,
                    reason=str(clip.get("reason") or "violation"),
                    occurred_at=int(clip.get("occurredAt") or now),
                    violation_id=clip.get("violationId"),
                    segment=int(clip.get("segment") or segment),
                    force=True,
                )
                if asset:
                    clip["assetId"] = asset["assetId"]

        submission_asset = self._materialize_clip(
            session,
            reason=str(payload.get("reason") or "submitted"),
            occurred_at=now,
            violation_id=None,
            segment=segment,
            force=True,
        )
        self._delete_temporary_chunks(session)
        self._finalize_session(session, result="passed", reason=str(payload.get("reason") or "submitted"))
        response = {"ok": True, "status": "stopped", "segment": segment}
        if submission_asset:
            response["assetId"] = submission_asset["assetId"]
        return response

    def interrupt_session(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        session["status"] = "interrupted"
        session["interruptedAt"] = _now()
        session["interruptPayload"] = payload
        self._save_session(session)
        return {"ok": True, "session": session}

    def resume_session(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        session["status"] = "active"
        session["resumedAt"] = _now()
        session["resumePayload"] = payload
        self._save_session(session)
        return {"ok": True, "session": session}

    def fail_session(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        self._finalize_session(session, result="failed", reason=str(payload.get("reason") or "server_failed"))
        return {"ok": True, "session": self.get_session(session_id)}

    def finish_session(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        result = "failed" if str(payload.get("result") or payload.get("reason") or "").lower() == "failed" else "passed"
        self._finalize_session(session, result=result, reason=str(payload.get("reason") or "completed"))
        return {"ok": True, "session": self.get_session(session_id)}

    def save_media_chunk(
        self,
        session_id: str,
        token: str,
        content: bytes,
        *,
        segment: int,
        sequence: int,
        mime_type: str,
        duration_ms: int | None = None,
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        self._require_upload_token(session, token)
        if len(content) <= 0 or len(content) > int(self.settings.media_chunk_max_bytes):
            raise ValueError("invalid_chunk_size")

        now = _now()
        extension = ".webm" if "webm" in mime_type else ".bin"
        relative = Path(_safe(session_id)) / "chunks" / f"segment_{int(segment):03d}" / f"chunk_{int(sequence):06d}{extension}"
        path = self.assets_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

        entry = {
            "segment": int(segment),
            "sequence": int(sequence),
            "receivedAt": now,
            "durationMs": int(duration_ms or 0),
            "path": relative.as_posix(),
            "mimeType": mime_type or "video/webm",
            "sizeBytes": len(content),
        }
        chunks = list(session.get("chunks") or [])
        chunks.append(entry)
        session["chunks"] = chunks
        self._prune_old_chunks(session, now)
        self._save_session(session)
        self._materialize_due_clips(session_id)
        return {"ok": True, "chunk": entry}

    def capture_snapshot(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        reason = str(payload.get("reason") or "manual_proctor")
        content = self._decode_optional_image(payload.get("snapshotImage") or payload.get("image"))
        if content is None:
            return {"ok": True, "status": "requested", "reason": reason, "assetId": None, "deferred": True}

        asset_type = "identity_photo" if reason == "identity_verification" else "snapshot"
        asset = self._save_asset_file(
            session,
            asset_type=asset_type,
            content=content,
            suffix=".jpg",
            mime_type="image/jpeg",
            reason=reason,
            violation_id=payload.get("violationId"),
        )
        self._send_asset_webhook(session, asset)
        if reason == "violation":
            self._queue_clip(
                session,
                reason="violation",
                occurred_at=_now(),
                violation_id=payload.get("violationId"),
                segment=int(self._recording(session).get("currentSegment") or 1),
            )
        return {"ok": True, "status": "captured", "reason": reason, "assetId": asset["assetId"]}

    def record_violation(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(str(payload.get("sessionId") or payload.get("session_id")))
        now = _now()
        violation = {
            "id": str(payload.get("violationId") or uuid4().hex),
            "type": str(payload.get("violationType") or payload.get("violation_type") or "unknown"),
            "severity": payload.get("severity") or "warning",
            "confidence": payload.get("confidence"),
            "occurredAt": now,
            "metadata": payload,
        }
        session.setdefault("violations", []).append(violation)
        self._queue_clip(
            session,
            reason=violation["type"],
            occurred_at=now,
            violation_id=violation["id"],
            segment=int(self._recording(session).get("currentSegment") or 1),
        )
        return {"ok": True, "status": "recorded", "violation": violation}

    def get_asset_content(self, asset_id: str) -> tuple[Path, str]:
        asset = self._find_asset(asset_id)
        if asset is None:
            raise KeyError("asset_not_found")
        path = self.assets_root / asset["path"]
        if not path.is_file():
            raise KeyError("asset_content_not_found")
        return path, str(asset.get("mimeType") or mimetypes.guess_type(path.name)[0] or "application/octet-stream")

    def delete_asset(self, asset_id: str) -> bool:
        asset = self._find_asset(asset_id)
        if asset is None:
            return False
        path = self.assets_root / asset["path"]
        if path.exists():
            path.unlink()
        return True

    def _materialize_due_clips(self, session_id: str) -> None:
        session = self.get_session(session_id)
        pending = list(session.get("pendingClips") or [])
        changed = False
        now = _now()
        for clip in pending:
            if clip.get("assetId") or now < int(clip.get("availableAfter") or now):
                continue
            asset = self._materialize_clip(
                session,
                reason=str(clip.get("reason") or "violation"),
                occurred_at=int(clip.get("occurredAt") or now),
                violation_id=clip.get("violationId"),
                segment=int(clip.get("segment") or 1),
                force=False,
            )
            if asset:
                clip["assetId"] = asset["assetId"]
                changed = True
        if changed:
            session["pendingClips"] = pending
            self._save_session(session)

    def _queue_clip(
        self,
        session: dict[str, Any],
        *,
        reason: str,
        occurred_at: int,
        violation_id: Any,
        segment: int,
    ) -> None:
        pending = list(session.get("pendingClips") or [])
        pending.append({
            "reason": reason,
            "occurredAt": int(occurred_at),
            "availableAfter": int(occurred_at) + int(self.settings.clip_post_seconds),
            "violationId": violation_id,
            "segment": int(segment),
            "assetId": None,
        })
        session["pendingClips"] = pending
        self._save_session(session)

    def _materialize_clip(
        self,
        session: dict[str, Any],
        *,
        reason: str,
        occurred_at: int,
        violation_id: Any,
        segment: int,
        force: bool,
    ) -> dict[str, Any] | None:
        pre = int(self.settings.clip_pre_seconds)
        post = int(self.settings.clip_post_seconds)
        start = int(occurred_at) - pre
        end = int(occurred_at) + post
        window_chunks = [
            chunk for chunk in (session.get("chunks") or [])
            if int(chunk.get("segment") or 1) == int(segment)
            and start <= int(chunk.get("receivedAt") or 0) <= end
        ]
        chunks = window_chunks
        if not chunks and force:
            chunks = [
                chunk for chunk in (session.get("chunks") or [])
                if int(chunk.get("segment") or 1) == int(segment)
            ][-6:]
        if not chunks:
            return None
        chunks.sort(key=lambda item: (int(item.get("sequence") or 0), int(item.get("receivedAt") or 0)))
        content = b"".join((self.assets_root / chunk["path"]).read_bytes() for chunk in chunks if (self.assets_root / chunk["path"]).is_file())
        if not content:
            return None
        asset = self._save_asset_file(
            session,
            asset_type="video_clip",
            content=content,
            suffix=".webm",
            mime_type="video/webm",
            reason=reason,
            violation_id=violation_id,
            metadata={
                "clipStartAt": start,
                "clipEndAt": end,
                "recordingSegment": segment,
                "chunkCount": len(chunks),
                "strategy": "rolling_browser_chunks",
            },
        )
        self._send_asset_webhook(session, asset)
        return asset

    def _prune_old_chunks(self, session: dict[str, Any], now: int) -> None:
        cutoff = int(now) - max(
            int(self.settings.media_chunk_retention_seconds),
            int(self.settings.clip_pre_seconds) + int(self.settings.clip_post_seconds) + 60,
        )
        kept = []
        for chunk in session.get("chunks") or []:
            if int(chunk.get("receivedAt") or 0) >= cutoff:
                kept.append(chunk)
                continue
            path = self.assets_root / chunk["path"]
            if path.exists():
                path.unlink()
        session["chunks"] = kept

    def _delete_temporary_chunks(self, session: dict[str, Any]) -> None:
        for chunk in session.get("chunks") or []:
            path = self.assets_root / chunk["path"]
            if path.exists():
                path.unlink()
        session["chunks"] = []
        self._save_session(session)

    def _save_asset_file(
        self,
        session: dict[str, Any],
        *,
        asset_type: str,
        content: bytes,
        suffix: str,
        mime_type: str,
        reason: str,
        violation_id: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        asset_id = f"{asset_type}-{uuid4().hex}"
        relative = Path(_safe(str(session["id"]))) / "assets" / f"{asset_id}{suffix}"
        path = self.assets_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        asset = {
            "assetId": asset_id,
            "type": asset_type,
            "reason": reason,
            "violationId": violation_id,
            "path": relative.as_posix(),
            "mimeType": mime_type,
            "sizeBytes": len(content),
            "checksum": hashlib.sha256(content).hexdigest(),
            "capturedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "availableAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": metadata or {},
        }
        session.setdefault("assets", []).append(asset)
        self._save_session(session)
        return asset

    def _send_asset_webhook(self, session: dict[str, Any], asset: dict[str, Any]) -> None:
        self._send_webhook(session, {
            "eventId": f"asset-{asset['assetId']}",
            "eventType": "asset.captured",
            "sessionId": session["id"],
            "moodleSessionId": session.get("moodleSessionId"),
            "companyId": session.get("companyId"),
            "attemptId": session.get("attemptId"),
            "userId": session.get("userId"),
            "asset": asset,
        })

    def _finalize_session(self, session: dict[str, Any], *, result: str, reason: str) -> None:
        if session.get("completedAt"):
            return
        now = _now()
        session["status"] = "completed" if result == "passed" else "failed"
        session["completedAt"] = now
        session["result"] = result
        self._save_session(session)
        event_type = "session.completed" if result == "passed" else "session.failed"
        self._send_webhook(session, {
            "eventId": f"final-{session['id']}",
            "eventType": event_type,
            "sessionId": session["id"],
            "moodleSessionId": session.get("moodleSessionId"),
            "companyId": session.get("companyId"),
            "attemptId": session.get("attemptId"),
            "userId": session.get("userId"),
            "result": result,
            "reasonCode": reason,
            "completedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        })

    def _send_webhook(self, session: dict[str, Any], event: dict[str, Any]) -> None:
        url = str(session.get("callbackUrl") or self.settings.moodle_webhook_url or "").strip()
        secret = str(self.settings.moodle_webhook_secret or "").strip()
        if not url or not secret:
            self._append_delivery(session, event, "skipped", "webhook_not_configured")
            return
        body = json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        signature = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, content=body, headers={
                    "Content-Type": "application/json",
                    "X-ProctorCore-Signature": f"sha256={signature}",
                })
            status = "delivered" if 200 <= response.status_code < 300 else "failed"
            self._append_delivery(session, event, status, f"{response.status_code}: {response.text[:300]}")
        except Exception as exc:
            self._append_delivery(session, event, "failed", str(exc))

    def _append_delivery(self, session: dict[str, Any], event: dict[str, Any], status: str, message: str) -> None:
        try:
            current = self.get_session(str(session["id"]))
        except KeyError:
            current = session
        current.setdefault("webhookDeliveries", []).append({
            "eventId": event.get("eventId"),
            "eventType": event.get("eventType"),
            "status": status,
            "message": message,
            "time": _now(),
        })
        self._save_session(current)

    def _find_asset(self, asset_id: str) -> dict[str, Any] | None:
        for path in self.sessions_root.glob("*.json"):
            try:
                session = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for asset in session.get("assets") or []:
                if str(asset.get("assetId")) == str(asset_id):
                    return asset
        return None

    def _decode_optional_image(self, value: Any) -> bytes | None:
        if not isinstance(value, str) or not value.strip():
            return None
        raw = value.strip()
        if "," in raw and raw.lower().startswith("data:"):
            raw = raw.split(",", 1)[1]
        try:
            return base64.b64decode(raw, validate=True)
        except Exception:
            return None

    def _recording(self, session: dict[str, Any]) -> dict[str, Any]:
        recording = session.get("recording")
        return recording if isinstance(recording, dict) else {"state": "not_started", "currentSegment": 0, "segments": {}}

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_root / f"{_safe(session_id)}.json"

    def _save_session(self, session: dict[str, Any]) -> None:
        path = self._session_path(str(session["id"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(session, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    def _payload_user_id(self, payload: dict[str, Any]) -> int | None:
        user = payload.get("user")
        if isinstance(user, dict) and user.get("id"):
            return int(user["id"])
        value = payload.get("userId") or payload.get("user_id")
        return int(value) if value else None

    def _require_upload_token(self, session: dict[str, Any], token: str) -> None:
        digest = hashlib.sha256(str(token).encode("utf-8")).hexdigest()
        expires = int(session.get("uploadTokenExpiresAt") or 0)
        if not session.get("uploadTokenHash") or expires < _now() or not hmac.compare_digest(digest, session["uploadTokenHash"]):
            raise PermissionError("invalid_upload_token")

    def _livekit_token(self, *, room: str, identity: str, name: str) -> str:
        now = _now()
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "iss": self.settings.livekit_api_key,
            "sub": identity,
            "name": name,
            "nbf": now - 10,
            "exp": now + 3600,
            "video": {
                "room": room,
                "roomJoin": True,
                "canPublish": True,
                "canPublishData": True,
                "canSubscribe": False,
            },
        }
        signing_input = ".".join([
            _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8")),
            _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
        ])
        signature = hmac.new(
            self.settings.livekit_api_secret.encode("utf-8"),
            signing_input.encode("ascii"),
            hashlib.sha256,
        ).digest()
        return signing_input + "." + _b64url(signature)
