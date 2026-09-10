from __future__ import annotations

import hashlib
import hmac
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from app.core.config import get_settings
from app.services.media_store import MediaStore
from app.services.object_store import ObjectStore


def deliver_webhook(session_id: str, event: dict[str, Any]) -> dict[str, Any]:
    """Deliver one deterministic event; retryable failures are raised for RQ."""
    settings = get_settings()
    store = MediaStore()
    session = store.get_session(session_id)
    url = str(session.get("callbackUrl") or settings.moodle_webhook_url or "").strip()
    secret = str(settings.moodle_webhook_secret or "").strip()
    if not url or not secret:
        _record_delivery(store, session_id, event, "skipped", "webhook_not_configured")
        return {"status": "skipped"}

    body = json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    try:
        response = httpx.post(
            url,
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-ProctorCore-Signature": f"sha256={signature}",
                "X-ProctorCore-Event-Id": str(event.get("eventId") or ""),
            },
            timeout=httpx.Timeout(15.0, connect=5.0),
        )
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        _record_delivery(store, session_id, event, "retrying", str(exc))
        raise

    message = f"{response.status_code}: {response.text[:300]}"
    if 200 <= response.status_code < 300:
        _record_delivery(store, session_id, event, "delivered", message)
        return {"status": "delivered", "httpCode": response.status_code}
    if response.status_code in {408, 429} or response.status_code >= 500:
        _record_delivery(store, session_id, event, "retrying", message)
        response.raise_for_status()
    _record_delivery(store, session_id, event, "dead_letter", message)
    return {"status": "dead_letter", "httpCode": response.status_code}


def _record_delivery(
    store: MediaStore,
    session_id: str,
    event: dict[str, Any],
    status: str,
    message: str,
) -> None:
    with store.state.session_lock(session_id):
        store._append_delivery(store.get_session(session_id), event, status, message)


def finalize_session_media(session_id: str, reason: str = "submitted", result: str = "passed") -> dict[str, Any]:
    """Finalize Egress/browser media, create clips, then publish the final result."""
    settings = get_settings()
    store = MediaStore()
    objects = ObjectStore(settings)
    with store.state.session_lock(session_id):
        session = store.get_session(session_id)
        session["processing"] = {"state": "running", "jobId": _current_job_id(), "error": None}
        store._save_session(session)

    try:
        with tempfile.TemporaryDirectory(prefix=f"proctor-{session_id[:12]}-") as temporary:
            work = Path(temporary)
            recording, strategy, segment_media = _build_recording(session, objects, work)
            if recording is None:
                raise RuntimeError("media_finalization_failed: no durable recording was available")
            recording_asset = store.register_asset_bytes(
                session_id,
                "full_recording",
                recording.read_bytes(),
                ".mp4",
                "video/mp4",
                "full_session",
                metadata={"strategy": strategy, "segmentCount": len(segment_media)},
            )
            _create_violation_clips(
                store,
                session_id,
                session,
                recording,
                recording_asset["assetId"],
                work,
                segment_media,
            )

        with store.state.session_lock(session_id):
            session = store.get_session(session_id)
            session["processing"] = {"state": "completed", "jobId": _current_job_id(), "error": None}
            recording = dict(session.get("recording") or {})
            recording["state"] = "completed"
            session["recording"] = recording
            store._save_session(session)
            store._delete_temporary_media(session)
            store._finalize_session(
                session,
                result="failed" if result == "failed" else "passed",
                reason=reason,
            )
        return {"status": "completed", "sessionId": session_id}
    except Exception as exc:
        with store.state.session_lock(session_id):
            session = store.get_session(session_id)
            retries_left = _current_retries_left()
            state = "retrying" if retries_left > 0 else "dead_letter"
            session["processing"] = {
                "state": state,
                "jobId": _current_job_id(),
                "error": str(exc)[:1000],
                "retriesLeft": retries_left,
            }
            recording = dict(session.get("recording") or {})
            recording["state"] = state
            session["recording"] = recording
            store._save_session(session)
            if retries_left <= 0:
                store._finalize_session(session, result="failed", reason="media_processing_failed")
        raise


def _build_recording(
    session: dict[str, Any],
    objects: ObjectStore,
    work: Path,
) -> tuple[Path | None, str, dict[int, dict[str, Any]]]:
    entries = list((session.get("recording") or {}).get("segments", {}).values())
    chunks = list(session.get("chunks") or [])
    segment_numbers = {int(item.get("segment") or 1) for item in entries}
    segment_numbers.update(int(item.get("segment") or 1) for item in chunks)
    segment_media: dict[int, dict[str, Any]] = {}
    strategies = []

    for segment in sorted(segment_numbers):
        entry = next((item for item in entries if int(item.get("segment") or 1) == segment), {})
        output = _build_egress_segment(entry, segment, objects, work)
        strategy = "livekit_egress_hls"
        if output is None:
            output = _build_browser_segment(chunks, segment, objects, work)
            strategy = "browser_chunk_fallback"
        if output is None:
            continue
        segment_media[segment] = {
            "path": output,
            "startedAt": int(entry.get("startedAt") or session.get("startedAt") or session.get("createdAt") or 0),
            "strategy": strategy,
        }
        strategies.append(strategy)

    if not segment_media:
        return None, "no_media", {}

    paths = [segment_media[number]["path"] for number in sorted(segment_media)]
    full = work / "recording.mp4"
    if len(paths) == 1:
        full.write_bytes(paths[0].read_bytes())
    else:
        concat_file = work / "recording_segments.txt"
        concat_file.write_text("".join(f"file '{path.as_posix()}'\n" for path in paths), encoding="utf-8")
        _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(full)])
    strategy = "mixed_segment_reconciliation" if len(set(strategies)) > 1 else strategies[0]
    return full, strategy, segment_media


def _build_egress_segment(
    entry: dict[str, Any],
    segment: int,
    objects: ObjectStore,
    work: Path,
) -> Path | None:
    prefix = str((entry.get("egress") or {}).get("objectPrefix") or "")
    if not prefix or not _wait_for_object(objects, objects.settings.s3_bucket_temp, f"{prefix}/index.m3u8", 90):
        return None
    local_root = work / "egress" / f"segment_{segment:03d}"
    for key in objects.list_keys(objects.settings.s3_bucket_temp, prefix):
        relative = key[len(prefix):].lstrip("/")
        objects.download_file(objects.settings.s3_bucket_temp, key, local_root / relative)
    playlist = local_root / "index.m3u8"
    output = work / f"segment_{segment:03d}_egress.mp4"
    _run_ffmpeg(["-i", str(playlist), "-map", "0:v?", "-map", "0:a?", "-c", "copy", str(output)])
    return output if output.is_file() and output.stat().st_size > 0 else None


def _build_browser_segment(
    chunks: list[dict[str, Any]],
    segment: int,
    objects: ObjectStore,
    work: Path,
) -> Path | None:
    selected = sorted(
        [item for item in chunks if int(item.get("segment") or 1) == segment],
        key=lambda item: int(item.get("sequence") or 0),
    )
    if not selected:
        return None
    chunk_dir = work / "browser" / f"segment_{segment:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    local_chunks = []
    for index, chunk in enumerate(selected):
        target = chunk_dir / f"chunk_{index:06d}.webm"
        try:
            objects.download_file(str(chunk["bucket"]), str(chunk["objectKey"]), target)
            local_chunks.append(target)
        except Exception:
            continue
    if not local_chunks:
        return None
    concat_file = chunk_dir / "chunks.txt"
    concat_file.write_text("".join(f"file '{path.as_posix()}'\n" for path in local_chunks), encoding="utf-8")
    joined = work / f"segment_{segment:03d}_browser.webm"
    _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(joined)])
    output = work / f"segment_{segment:03d}_browser.mp4"
    _run_ffmpeg([
        "-i", str(joined), "-map", "0:v?", "-map", "0:a?", "-c:v", "libx264",
        "-preset", "veryfast", "-c:a", "aac", "-movflags", "+faststart", str(output),
    ])
    return output if output.is_file() and output.stat().st_size > 0 else None


def _create_violation_clips(
    store: MediaStore,
    session_id: str,
    session: dict[str, Any],
    recording: Path,
    source_asset_id: str,
    work: Path,
    segment_media: dict[int, dict[str, Any]],
) -> None:
    settings = get_settings()
    started_at = int(session.get("startedAt") or session.get("createdAt") or 0)
    pending = list(session.get("pendingClips") or [])
    violations = {str(item.get("id")): item for item in session.get("violations") or []}
    candidates = pending or [
        {
            "violationId": item.get("id"),
            "reason": item.get("type"),
            "occurredAt": item.get("occurredAt"),
            "assetId": None,
        }
        for item in violations.values()
    ]
    for index, candidate in enumerate(candidates):
        if candidate.get("assetId"):
            continue
        violation = violations.get(str(candidate.get("violationId")), {})
        occurred_at = int(candidate.get("occurredAt") or violation.get("occurredAt") or started_at)
        segment = int(candidate.get("segment") or 1)
        media = segment_media.get(segment)
        source = Path(media["path"]) if media else recording
        source_started_at = int(media.get("startedAt") or started_at) if media else started_at
        start_offset = max(0, occurred_at - source_started_at - int(settings.clip_pre_seconds))
        duration = int(settings.clip_pre_seconds) + int(settings.clip_post_seconds)
        clip_path = work / f"clip_{index:04d}.mp4"
        _run_ffmpeg([
            "-ss", str(start_offset), "-i", str(source), "-t", str(duration),
            "-map", "0:v?", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "aac", "-movflags", "+faststart", str(clip_path),
        ])
        if not clip_path.is_file() or clip_path.stat().st_size == 0:
            continue
        asset = store.register_asset_bytes(
            session_id,
            "video_clip",
            clip_path.read_bytes(),
            ".mp4",
            "video/mp4",
            str(candidate.get("reason") or violation.get("type") or "violation"),
            violation_id=candidate.get("violationId") or violation.get("id"),
            metadata={
                "sourceAssetId": source_asset_id,
                "clipStartSeconds": start_offset,
                "clipDurationSeconds": duration,
                "recordingSegment": segment,
                "strategy": "ffmpeg_timestamp_extract",
            },
        )
        candidate["assetId"] = asset["assetId"]

    current = store.get_session(session_id)
    current["pendingClips"] = candidates
    store._save_session(current)


def _run_ffmpeg(arguments: list[str]) -> None:
    completed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg_failed: {completed.stderr[-800:]}")


def _wait_for_object(objects: ObjectStore, bucket: str, key: str, timeout: int) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if objects.exists(bucket, key):
            return True
        time.sleep(2)
    return False


def _current_segment(session: dict[str, Any]) -> dict[str, Any]:
    recording = session.get("recording") or {}
    segment = str(recording.get("currentSegment") or 1)
    return (recording.get("segments") or {}).get(segment) or {}


def _current_job_id() -> str | None:
    try:
        from rq import get_current_job

        job = get_current_job()
        return job.id if job else None
    except Exception:
        return None


def _current_retries_left() -> int:
    try:
        from rq import get_current_job

        job = get_current_job()
        return max(0, int(job.retries_left or 0)) if job else 0
    except Exception:
        return 0
