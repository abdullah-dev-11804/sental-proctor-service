from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.config import get_settings
from app.services.job_queue import JobQueue
from app.services.livekit_egress import LiveKitEgress
from app.services.object_store import ObjectStore
from app.services.state_store import StateStore


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
        self.assets_root = self.root / "assets"
        self.assets_root.mkdir(parents=True, exist_ok=True)
        self.state = StateStore(self.settings)
        self.objects = ObjectStore(self.settings)
        self.jobs = JobQueue(self.settings)
        self.egress = LiveKitEgress(self.settings)

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(uuid4())
        now = _now()
        requested_retention = payload.get("retention") if isinstance(payload.get("retention"), dict) else {}
        appeal_days = max(1, int(requested_retention.get("appealDays") or self.settings.default_appeal_period_days))
        video_days = max(
            appeal_days,
            int(requested_retention.get("videoDays") or self.settings.default_video_retention_days),
        )
        report_days = max(183, int(requested_retention.get("reportDays") or self.settings.default_report_retention_days))
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
            # Production uses the configured trusted target. A request-provided
            # callback is only a development fallback when no target is set.
            "callbackUrl": self.settings.moodle_webhook_url or payload.get("callbackUrl"),
            "returnUrl": payload.get("returnUrl"),
            "createdAt": now,
            "startedAt": None,
            "completedAt": None,
            "lastHeartbeatAt": None,
            "uploadTokenHash": None,
            "recording": {
                "state": "not_started",
                "provider": "livekit_egress" if self.settings.livekit_egress_enabled else "browser_fallback",
                "currentSegment": 0,
                "segments": {},
            },
            "processing": {"state": "idle", "jobId": None, "error": None},
            "retention": {
                "held": False,
                "holdReason": None,
                "videoDays": video_days,
                "reportDays": report_days,
                "appealDays": appeal_days,
                "appealUntil": None,
                "videoExpiresAt": None,
                "reportExpiresAt": None,
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
        return self.state.get_session(session_id)

    def start_session(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        self._require_scope(session, payload)
        now = _now()
        session["status"] = "active"
        session["startedAt"] = session.get("startedAt") or now
        session["lastHeartbeatAt"] = now
        session["lastStartPayload"] = payload
        self._save_session(session)
        return session

    def heartbeat(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        self._require_scope(session, payload)
        session["lastHeartbeatAt"] = _now()
        session["lastHeartbeatPayload"] = payload
        self._save_session(session)
        return session

    def create_media_token(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        self._require_scope(session, payload)
        token = self._livekit_token(
            room=str(session.get("roomId") or f"room-{session_id}"),
            identity=str(payload.get("participantIdentity") or f"user-{session.get('userId') or 'unknown'}"),
            name=str(payload.get("participantName") or ""),
        )
        upload_token = uuid4().hex + uuid4().hex
        session["participantIdentity"] = str(
            payload.get("participantIdentity") or f"user-{session.get('userId') or 'unknown'}"
        )
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
        self._require_scope(session, payload)
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
        provider = "browser_fallback"
        egress_result: dict[str, Any] = {}
        if self.settings.livekit_egress_enabled:
            prefix = (
                f"temp/{int(session.get('companyId') or 0)}/{_safe(session_id)}"
                f"/recording/segment_{segment:03d}"
            )
            try:
                egress_result = self.egress.start_participant(
                    str(session.get("roomId") or f"room-{session_id}"),
                    str(session.get("participantIdentity") or f"user-{session.get('userId') or 'unknown'}"),
                    prefix,
                )
                provider = "livekit_egress" if egress_result.get("egressId") else "browser_fallback"
                recording_id = str(egress_result.get("egressId") or recording_id)
            except Exception as exc:
                egress_result = {"state": "failed", "error": str(exc)[:500]}
        recording["state"] = "active"
        recording["provider"] = provider
        recording["currentSegment"] = segment
        recording["recordingId"] = recording_id
        recording.setdefault("segments", {})[str(segment)] = {
            "segment": segment,
            "recordingId": recording_id,
            "startedAt": now,
            "stoppedAt": None,
            "reason": payload.get("reason") or "attempt_page_connected",
            "egress": egress_result,
        }
        session["recording"] = recording
        session["status"] = "active"
        session["startedAt"] = session.get("startedAt") or now
        self._save_session(session)
        return {
            "ok": True,
            "status": "active",
            "recordingId": recording_id,
            "segment": segment,
            "provider": provider,
            "fallback": provider != "livekit_egress",
        }

    def stop_recording(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        self._require_scope(session, payload)
        recording = self._recording(session)
        segment = int(payload.get("segment") or recording.get("currentSegment") or 1)
        now = _now()
        reason = str(payload.get("reason") or "submitted")
        interrupted = reason in {
            "connection_lost",
            "media_track_ended",
            "browser_offline",
            "media_device_error",
            "media_connection_disconnected",
        }
        if recording.get("state") != "active" and interrupted:
            return {"ok": True, "status": str(recording.get("state") or "interrupted"), "duplicate": True}
        if recording.get("state") in {"stopping", "completed"} and not interrupted:
            return {"ok": True, "status": str(recording.get("state")), "duplicate": True}

        recording["state"] = "interrupted" if interrupted else "stopping"
        segment_entry = recording.setdefault("segments", {}).setdefault(str(segment), {"segment": segment})
        segment_entry["stoppedAt"] = now
        segment_entry["stopReason"] = reason
        egress_id = str((segment_entry.get("egress") or {}).get("egressId") or "")
        if egress_id:
            try:
                segment_entry["egressStop"] = self.egress.stop(egress_id)
            except Exception as exc:
                segment_entry["egressStop"] = {"state": "failed", "error": str(exc)[:500]}
        session["recording"] = recording
        if interrupted:
            session["status"] = "interrupted"
            session["interruptedAt"] = now
            self._save_session(session)
            return {
                "ok": True,
                "status": "interrupted",
                "segment": segment,
                "partialMediaPreserved": True,
            }

        session["status"] = "processing"
        session["finalResult"] = "failed" if str(payload.get("result") or "").lower() == "failed" else "passed"
        session["processing"] = {"state": "queued", "jobId": None, "error": None}
        self._save_session(session)
        try:
            job = self.jobs.enqueue(
                "app.workers.jobs.finalize_session_media",
                session_id,
                reason,
                str(session["finalResult"]),
                job_id=f"finalize-{_safe(session_id)}-{segment}",
                retry=True,
            )
            session = self.get_session(session_id)
            session["processing"] = {"state": "queued", "jobId": job["jobId"], "error": None}
            self._save_session(session)
        except Exception as exc:
            session["processing"] = {"state": "queue_failed", "jobId": None, "error": str(exc)[:500]}
            self._save_session(session)
            if self.settings.storage_require_ready:
                raise
        return {"ok": True, "status": "processing", "segment": segment, "processing": session["processing"]}

    def interrupt_session(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        self._require_scope(session, payload)
        session["status"] = "interrupted"
        session["interruptedAt"] = _now()
        session["interruptPayload"] = payload
        self._save_session(session)
        return {"ok": True, "session": session}

    def resume_session(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        self._require_scope(session, payload)
        if session.get("completedAt") or session.get("status") in {"completed", "failed"}:
            raise KeyError("session_closed")
        session["status"] = "active"
        session["resumedAt"] = _now()
        session["resumePayload"] = payload
        self._save_session(session)
        return {"ok": True, "session": session}

    def fail_session(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        self._require_scope(session, payload)
        return self._queue_session_finalization(
            session,
            reason=str(payload.get("reason") or "server_failed"),
            result="failed",
        )

    def finish_session(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        result = "failed" if str(payload.get("result") or payload.get("reason") or "").lower() == "failed" else "passed"
        return self._queue_session_finalization(
            session,
            reason=str(payload.get("reason") or "completed"),
            result=result,
        )

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
        relative = (
            f"temp/{int(session.get('companyId') or 0)}/{_safe(session_id)}/browser/"
            f"segment_{int(segment):03d}/chunk_{int(sequence):06d}{extension}"
        )
        stored = self.objects.put_bytes(
            self.settings.s3_bucket_temp,
            relative,
            content,
            mime_type or "video/webm",
        )

        entry = {
            "segment": int(segment),
            "sequence": int(sequence),
            "receivedAt": now,
            "durationMs": int(duration_ms or 0),
            "bucket": stored.bucket,
            "objectKey": stored.key,
            "mimeType": mime_type or "video/webm",
            "sizeBytes": len(content),
        }
        chunks = list(session.get("chunks") or [])
        chunks.append(entry)
        session["chunks"] = chunks
        self._prune_old_chunks(session, now)
        self._save_session(session)
        return {"ok": True, "chunk": entry}

    def capture_snapshot(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        self._require_scope(session, payload)
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
            # The asset webhook appends delivery state. Reload before adding the
            # violation so that neither update can overwrite the other.
            session = self.get_session(session_id)
            occurred_at = self._timestamp(payload.get("occurredAt")) or _now()
            violation_id = str(payload.get("violationId") or uuid4().hex)
            violation_type = _safe(str(payload.get("violationType") or "violation"), 64)
            violations = list(session.get("violations") or [])
            if not any(str(item.get("id")) == violation_id for item in violations):
                violations.append({
                    "id": violation_id,
                    "type": violation_type,
                    "severity": payload.get("severity") or "warning",
                    "occurredAt": occurred_at,
                    "metadata": {"source": "moodle_violation_snapshot"},
                })
                session["violations"] = violations
            self._queue_clip(
                session,
                reason=violation_type,
                occurred_at=occurred_at,
                violation_id=violation_id,
                segment=int(self._recording(session).get("currentSegment") or 1),
            )
        return {"ok": True, "status": "captured", "reason": reason, "assetId": asset["assetId"]}

    def record_violation(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.get_session(str(payload.get("sessionId") or payload.get("session_id")))
        self._require_scope(session, payload)
        now = self._timestamp(payload.get("occurredAt") or payload.get("occurred_at")) or _now()
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

    def get_asset_content(self, asset_id: str) -> tuple[bytes, str]:
        asset = self._find_asset(asset_id)
        if asset is None:
            raise KeyError("asset_not_found")
        if asset.get("status") == "deleted":
            raise KeyError("asset_content_not_found")
        try:
            content = self.objects.get_bytes(str(asset["bucket"]), str(asset["objectKey"]))
        except Exception as exc:
            raise KeyError("asset_content_not_found") from exc
        return content, str(asset.get("mimeType") or "application/octet-stream")

    def delete_asset(self, asset_id: str) -> bool:
        asset = self._find_asset(asset_id)
        if asset is None:
            raise KeyError("asset_not_found")
        deleted = self.objects.delete(str(asset["bucket"]), str(asset["objectKey"]))
        asset["status"] = "deleted"
        asset["deletedAt"] = _now()
        self.state.index_asset(asset)
        session = self.get_session(str(asset["sessionId"]))
        for item in session.get("assets") or []:
            if str(item.get("assetId")) == asset_id:
                item.update({"status": "deleted", "deletedAt": asset["deletedAt"]})
                break
        self._save_session(session)
        return deleted

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
        parts = []
        for chunk in chunks:
            try:
                parts.append(self.objects.get_bytes(str(chunk["bucket"]), str(chunk["objectKey"])))
            except Exception:
                continue
        content = b"".join(parts)
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
            self.objects.delete(str(chunk["bucket"]), str(chunk["objectKey"]))
        session["chunks"] = kept

    def _delete_temporary_chunks(self, session: dict[str, Any]) -> None:
        for chunk in session.get("chunks") or []:
            self.objects.delete(str(chunk["bucket"]), str(chunk["objectKey"]))
        session["chunks"] = []
        self._save_session(session)

    def _delete_temporary_media(self, session: dict[str, Any]) -> None:
        self._delete_temporary_chunks(session)
        for entry in (session.get("recording") or {}).get("segments", {}).values():
            prefix = str((entry.get("egress") or {}).get("objectPrefix") or "")
            if prefix:
                self.objects.delete_prefix(self.settings.s3_bucket_temp, prefix)

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
        company_id = int(session.get("companyId") or 0)
        object_key = (
            f"evidence/{company_id}/{_safe(str(session['id']))}/"
            f"{self._asset_directory(asset_type)}/{asset_id}{suffix}"
        )
        stored = self.objects.put_bytes(
            self.settings.s3_bucket_evidence,
            object_key,
            content,
            mime_type,
            {"session-id": str(session["id"]), "company-id": str(company_id)},
        )
        asset = {
            "assetId": asset_id,
            "type": asset_type,
            "reason": reason,
            "violationId": violation_id,
            "bucket": stored.bucket,
            "objectKey": stored.key,
            "mimeType": mime_type,
            "sizeBytes": len(content),
            "checksum": hashlib.sha256(content).hexdigest(),
            "capturedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "availableAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": metadata or {},
            "status": "active",
            "held": bool((session.get("retention") or {}).get("held")),
            "expiresAt": (session.get("retention") or {}).get(
                "videoExpiresAt" if asset_type in {"video_clip", "full_recording"} else "reportExpiresAt"
            ),
            "sessionId": session["id"],
            "companyId": company_id,
        }
        session.setdefault("assets", []).append(asset)
        self._save_session(session)
        self.state.index_asset(asset)
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
        retention = dict(session.get("retention") or {})
        appeal_days = max(1, int(retention.get("appealDays") or self.settings.default_appeal_period_days))
        video_days = max(appeal_days, int(retention.get("videoDays") or self.settings.default_video_retention_days))
        report_days = max(183, int(retention.get("reportDays") or self.settings.default_report_retention_days))
        retention.update({
            "appealUntil": now + (appeal_days * 86400),
            "videoExpiresAt": now + (video_days * 86400),
            "reportExpiresAt": now + (report_days * 86400),
        })
        session["retention"] = retention
        for asset in session.get("assets") or []:
            asset["expiresAt"] = retention[
                "videoExpiresAt" if asset.get("type") in {"video_clip", "full_recording"} else "reportExpiresAt"
            ]
            asset["held"] = bool(retention.get("held"))
            self.state.index_asset(asset)
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
            "mediaStatus": (session.get("processing") or {}).get("state"),
            "assetCount": len(session.get("assets") or []),
            "completedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        })

    def _send_webhook(self, session: dict[str, Any], event: dict[str, Any]) -> None:
        url = str(session.get("callbackUrl") or self.settings.moodle_webhook_url or "").strip()
        secret = str(self.settings.moodle_webhook_secret or "").strip()
        if not url or not secret:
            self._append_delivery(session, event, "skipped", "webhook_not_configured")
            return
        try:
            self.jobs.enqueue(
                "app.workers.jobs.deliver_webhook",
                str(session["id"]),
                event,
                # The event id is deterministic for receiver deduplication. The
                # queue id is unique so reconciliation can deliberately resend it.
                job_id=f"webhook-{_safe(str(event.get('eventId') or 'event'))}-{uuid4().hex[:12]}",
                retry=True,
            )
            self._append_delivery(session, event, "queued", "durable_delivery_queued")
        except Exception as exc:
            self._append_delivery(session, event, "queue_failed", str(exc))
            if self.settings.storage_require_ready:
                raise

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
        try:
            return self.state.get_asset(asset_id)
        except KeyError:
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

    @staticmethod
    def _timestamp(value: Any) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)) or str(value).isdigit():
            timestamp = int(value)
            return timestamp // 1000 if timestamp > 100000000000 else timestamp
        try:
            from datetime import datetime

            return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
        except (TypeError, ValueError):
            return None

    def _queue_session_finalization(
        self,
        session: dict[str, Any],
        *,
        reason: str,
        result: str,
    ) -> dict[str, Any]:
        if session.get("completedAt"):
            return {"ok": True, "status": session.get("status"), "duplicate": True}
        recording = self._recording(session)
        recording["state"] = "stopping"
        session["recording"] = recording
        session["status"] = "processing"
        session["finalResult"] = result
        session["processing"] = {"state": "queued", "jobId": None, "error": None}
        self._save_session(session)
        job = self.jobs.enqueue(
            "app.workers.jobs.finalize_session_media",
            str(session["id"]),
            reason,
            result,
            job_id=f"finalize-{_safe(str(session['id']))}-terminal",
            retry=True,
        )
        session = self.get_session(str(session["id"]))
        session["processing"] = {"state": "queued", "jobId": job["jobId"], "error": None}
        self._save_session(session)
        return {"ok": True, "status": "processing", "processing": session["processing"]}

    def _recording(self, session: dict[str, Any]) -> dict[str, Any]:
        recording = session.get("recording")
        return recording if isinstance(recording, dict) else {"state": "not_started", "currentSegment": 0, "segments": {}}

    def _save_session(self, session: dict[str, Any]) -> None:
        self.state.save_session(session)

    @staticmethod
    def _asset_directory(asset_type: str) -> str:
        if asset_type == "video_clip":
            return "clips"
        if asset_type == "full_recording":
            return "recording"
        if asset_type == "report_pdf":
            return "reports"
        return "snapshots"

    def register_asset_bytes(
        self,
        session_id: str,
        asset_type: str,
        content: bytes,
        suffix: str,
        mime_type: str,
        reason: str,
        violation_id: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        asset = self._save_asset_file(
            session,
            asset_type=asset_type,
            content=content,
            suffix=suffix,
            mime_type=mime_type,
            reason=reason,
            violation_id=violation_id,
            metadata=metadata,
        )
        self._send_asset_webhook(session, asset)
        return asset

    def set_evidence_hold(self, session_id: str, held: bool, reason: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        retention = dict(session.get("retention") or {})
        retention["held"] = bool(held)
        retention["holdReason"] = reason if held else None
        retention["changedAt"] = _now()
        session["retention"] = retention
        for asset in session.get("assets") or []:
            asset["held"] = bool(held)
            self.state.index_asset(asset)
        self._save_session(session)
        return retention

    def reconcile_session(self, session_id: str, resend_webhooks: bool = False) -> dict[str, Any]:
        session = self.get_session(session_id)
        missing = []
        available = []
        for asset in session.get("assets") or []:
            exists = self.objects.exists(str(asset["bucket"]), str(asset["objectKey"]))
            asset["status"] = "active" if exists else "missing"
            self.state.index_asset(asset)
            (available if exists else missing).append(asset["assetId"])
            if exists and resend_webhooks:
                self._send_asset_webhook(session, asset)
        session["reconciliation"] = {
            "checkedAt": _now(),
            "available": len(available),
            "missing": len(missing),
        }
        self._save_session(session)
        return {"available": available, "missing": missing, **session["reconciliation"]}

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

    @staticmethod
    def _require_scope(session: dict[str, Any], payload: dict[str, Any]) -> None:
        checks = (
            ("companyId", "company_id", "companyId"),
            ("userId", "user_id", "userId"),
            ("attemptId", "quiz_attempt_id", "attemptId"),
            ("moodleSessionId", "moodle_session_id", "moodleSessionId"),
        )
        for camel, snake, session_key in checks:
            supplied = payload.get(camel)
            if supplied is None:
                supplied = payload.get(snake)
            expected = session.get(session_key)
            if supplied is not None and expected is not None and int(supplied) != int(expected):
                raise PermissionError("tenant_scope_mismatch")

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
