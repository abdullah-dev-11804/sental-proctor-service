from __future__ import annotations

import hashlib
import hmac
import json
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import median
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
    """Build temporary full media, retain key evidence, then publish the result."""
    settings = get_settings()
    store = MediaStore()
    objects = ObjectStore(settings)
    with store.state.session_lock(session_id):
        session = store.get_session(session_id)
        session["processing"] = {"state": "running", "jobId": _current_job_id(), "error": None}
        store._save_session(session)

    try:
        _wait_for_audio_analysis(store, session_id, settings)
        session = store.get_session(session_id)
        with tempfile.TemporaryDirectory(prefix=f"proctor-{session_id[:12]}-") as temporary:
            work = Path(temporary)
            recording, strategy, segment_media = _build_recording(session, objects, work)
            if recording is None:
                raise RuntimeError("media_finalization_failed: no durable recording was available")
            temporary_recording = objects.put_file(
                settings.s3_bucket_temp,
                (
                    f"temp/{int(session.get('companyId') or 0)}/{session_id}/"
                    "finalized/full-session.mp4"
                ),
                recording,
                "video/mp4",
                {
                    "session-id": session_id,
                    "company-id": str(int(session.get("companyId") or 0)),
                    "temporary": "true",
                },
            )
            _create_violation_clips(
                store,
                session_id,
                session,
                recording,
                work,
                segment_media,
            )

        cleanup_not_before = int(time.time()) + max(1, int(settings.media_reconciliation_grace_seconds))
        with store.state.session_lock(session_id):
            session = store.get_session(session_id)
            session["processing"] = {
                "state": "completed",
                "jobId": _current_job_id(),
                "error": None,
                "recordingStrategy": strategy,
            }
            recording = dict(session.get("recording") or {})
            recording["state"] = "completed"
            session["recording"] = recording
            required_assets = [
                str(asset["assetId"])
                for asset in session.get("assets") or []
                if asset.get("status") == "active"
            ]
            session["temporaryMedia"] = {
                "state": "awaiting_reconciliation",
                "cleanupNotBefore": cleanup_not_before,
                "fullRecording": {
                    "bucket": temporary_recording.bucket,
                    "objectKey": temporary_recording.key,
                    "sizeBytes": temporary_recording.size,
                    "strategy": strategy,
                    "segmentCount": len(segment_media),
                },
                "requiredAssetIds": required_assets,
                "requiredEventIds": [
                    *[f"asset-{asset_id}" for asset_id in required_assets],
                    f"final-{session_id}",
                ],
            }
            store._save_session(session)
            store._finalize_session(
                session,
                result="failed" if result == "failed" else "passed",
                reason=reason,
            )
        cleanup = store.jobs.enqueue(
            "app.workers.jobs.cleanup_temporary_media",
            session_id,
            job_id=f"cleanup-{session_id}",
            retry=True,
            delay_seconds=max(1, int(settings.media_reconciliation_grace_seconds)),
        )
        with store.state.session_lock(session_id):
            session = store.get_session(session_id)
            temporary = dict(session.get("temporaryMedia") or {})
            temporary["cleanupJobId"] = cleanup["jobId"]
            session["temporaryMedia"] = temporary
            store._save_session(session)
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


def _wait_for_audio_analysis(store: MediaStore, session_id: str, settings) -> None:
    """Allows the live analyzer to flush its final active speech/noise event."""
    session = store.get_session(session_id)
    requested = bool((session.get("audioAnalysis") or {}).get("enabled"))
    if not requested or not settings.audio_analysis_enabled:
        return
    deadline = time.monotonic() + max(1, int(settings.audio_finalize_wait_seconds))
    while time.monotonic() < deadline:
        session = store.get_session(session_id)
        state = session.get("audioAnalysisState") if isinstance(session.get("audioAnalysisState"), dict) else {}
        if state.get("status") in {"stopped", "failed", "degraded"}:
            return
        time.sleep(0.25)
    with store.state.session_lock(session_id):
        session = store.get_session(session_id)
        state = session.get("audioAnalysisState") if isinstance(session.get("audioAnalysisState"), dict) else {}
        state.update({
            "status": "degraded",
            "error": "audio_analysis_flush_timeout",
            "updatedAt": int(time.time()),
        })
        session["audioAnalysisState"] = state
        store._save_session(session)


def cleanup_temporary_media(session_id: str) -> dict[str, Any]:
    """Reconcile durable evidence and remove full-session temporary source media."""
    store = MediaStore()
    try:
        receipt = store.reconcile_and_cleanup_temporary_media(session_id)
        return {"status": "deleted", "sessionId": session_id, "receipt": receipt}
    except Exception as exc:
        with store.state.session_lock(session_id):
            session = store.get_session(session_id)
            temporary = dict(session.get("temporaryMedia") or {})
            retries_left = _current_retries_left()
            temporary.update({
                "state": "retrying" if retries_left > 0 else "manual_reconciliation_required",
                "lastError": str(exc)[:1000],
                "retriesLeft": retries_left,
                "lastCheckedAt": int(time.time()),
            })
            session["temporaryMedia"] = temporary
            store._save_session(session)
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
        egress_output = _build_egress_segment(entry, segment, objects, work)
        browser_output = _build_browser_segment(chunks, segment, objects, work)
        output, strategy = _prefer_complete_segment(egress_output, browser_output)
        if output is None:
            continue
        segment_media[segment] = {
            "path": output,
            "startedAt": int(entry.get("startedAt") or session.get("startedAt") or session.get("createdAt") or 0),
            "stoppedAt": int(entry.get("stoppedAt") or 0),
            "strategy": strategy,
        }
        strategies.append(strategy)

    if not segment_media:
        return None, "no_media", {}

    paths = [segment_media[number]["path"] for number in sorted(segment_media)]
    assembled = work / "recording_source.mp4"
    if len(paths) == 1:
        assembled.write_bytes(paths[0].read_bytes())
    else:
        concat_file = work / "recording_segments.txt"
        concat_file.write_text("".join(f"file '{path.as_posix()}'\n" for path in paths), encoding="utf-8")
        _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(assembled)])
    full = work / "recording.mp4"
    _normalize_media_timeline(assembled, full)
    strategy = "mixed_segment_reconciliation" if len(set(strategies)) > 1 else strategies[0]
    return full, strategy, segment_media


def _prefer_complete_segment(egress: Path | None, browser: Path | None) -> tuple[Path | None, str]:
    """Prefer Egress unless the browser safety copy has materially more captured media."""
    if egress is None:
        return browser, "browser_chunk_fallback"
    if browser is None:
        return egress, "livekit_egress_hls"
    # Container duration includes HLS timestamp holes. Measure packet payload so
    # a 30-second Egress file containing 17 seconds of missing frames does not
    # beat a genuinely more complete browser safety recording.
    if _media_content_duration_seconds(browser) > _media_content_duration_seconds(egress) + 2.0:
        return browser, "browser_chunk_fallback"
    return egress, "livekit_egress_hls"


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
    chunk_contents = []
    for chunk in selected:
        try:
            chunk_contents.append(objects.get_bytes(str(chunk["bucket"]), str(chunk["objectKey"])))
        except Exception:
            continue
    streams = _group_webm_chunks(chunk_contents)
    if not streams:
        return None

    chunk_dir = work / "browser" / f"segment_{segment:03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    for index, content in enumerate(streams):
        joined = chunk_dir / f"stream_{index:04d}.webm"
        joined.write_bytes(content)
        converted_path = chunk_dir / f"stream_{index:04d}.mp4"
        _run_ffmpeg([
            "-i", str(joined), "-map", "0:v?", "-map", "0:a?", "-c:v", "libx264",
            "-preset", "veryfast", "-c:a", "aac", "-movflags", "+faststart", str(converted_path),
        ])
        if _media_duration_seconds(converted_path) > 0:
            converted.append(converted_path)
    if not converted:
        return None

    output = work / f"segment_{segment:03d}_browser.mp4"
    if len(converted) == 1:
        output.write_bytes(converted[0].read_bytes())
    else:
        concat_file = chunk_dir / "streams.txt"
        concat_file.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in converted),
            encoding="utf-8",
        )
        _run_ffmpeg([
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", "-movflags", "+faststart", str(output),
        ])
    return output if output.is_file() and output.stat().st_size > 0 else None


def _group_webm_chunks(contents: list[bytes]) -> list[bytes]:
    """Reassemble MediaRecorder chunks, splitting at each new EBML stream."""
    ebml_header = b"\x1a\x45\xdf\xa3"
    groups: list[bytearray] = []
    for content in contents:
        if not content:
            continue
        if not groups or content.startswith(ebml_header):
            groups.append(bytearray())
        groups[-1].extend(content)
    return [bytes(group) for group in groups if group]


def _create_violation_clips(
    store: MediaStore,
    session_id: str,
    session: dict[str, Any],
    recording: Path,
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
    source_durations: dict[Path, float] = {}
    for index, candidate in enumerate(candidates):
        if candidate.get("assetId"):
            continue
        violation = violations.get(str(candidate.get("violationId")), {})
        occurred_at = int(candidate.get("occurredAt") or violation.get("occurredAt") or started_at)
        segment = int(candidate.get("segment") or 1)
        media = segment_media.get(segment)
        source = Path(media["path"]) if media else recording
        source_started_at = int(media.get("startedAt") or started_at) if media else started_at
        duration = int(settings.clip_pre_seconds) + int(settings.clip_post_seconds)
        if source not in source_durations:
            source_durations[source] = _media_duration_seconds(source)
        source_duration = source_durations[source]
        start_offset, event_offset, timeline_mapping = _violation_clip_start_offset(
            occurred_at=occurred_at,
            source_started_at=source_started_at,
            source_duration=source_duration,
            clip_duration=duration,
            pre_seconds=int(settings.clip_pre_seconds),
            media=media,
        )
        clip_path = work / f"clip_{index:04d}.mp4"
        frame_rate = _media_video_frame_rate(source)
        _run_ffmpeg([
            "-ss", str(start_offset), "-i", str(source), "-t", str(duration),
            "-map", "0:v:0?", "-map", "0:a:0?",
            # Egress preserves timestamp discontinuities when the browser
            # republishes tracks during Moodle page navigation. Rebuild time
            # from actual decoded frames/samples so clips do not start with
            # delayed video or contain multi-second frozen gaps.
            "-vf", f"setpts=N/({frame_rate:g}*TB)",
            "-af", "asetpts=N/SR/TB", "-fps_mode:v", "passthrough",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "48000", "-movflags", "+faststart", str(clip_path),
        ])
        if _media_duration_seconds(clip_path) <= 0:
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
                "source": "temporary_full_session",
                "clipStartSeconds": start_offset,
                "clipDurationSeconds": duration,
                "eventOffsetSeconds": event_offset,
                "timelineMapping": timeline_mapping,
                "recordingSegment": segment,
                "strategy": "ffmpeg_continuous_timeline_extract",
            },
        )
        candidate["assetId"] = asset["assetId"]

    current = store.get_session(session_id)
    current["pendingClips"] = candidates
    store._save_session(current)
    missing = [str(candidate.get("violationId") or "unknown") for candidate in candidates if not candidate.get("assetId")]
    if missing:
        raise RuntimeError("violation_clip_creation_failed: " + ",".join(missing[:10]))


def _violation_clip_start_offset(
    *,
    occurred_at: int,
    source_started_at: int,
    source_duration: float,
    clip_duration: int,
    pre_seconds: int,
    media: dict[str, Any] | None,
) -> tuple[float, float, str]:
    """Map a wall-clock violation onto Egress or compact browser media."""
    elapsed = float(max(0, occurred_at - source_started_at))
    event_offset = elapsed
    mapping = "wall_clock"
    if media and media.get("strategy") == "browser_chunk_fallback":
        stopped_at = int(media.get("stoppedAt") or 0)
        wall_duration = max(0, stopped_at - source_started_at)
        if wall_duration > 0 and source_duration > 0:
            event_offset = min(source_duration, elapsed * source_duration / wall_duration)
            mapping = "scaled_browser_timeline"

    start_offset = max(0.0, event_offset - pre_seconds)
    if source_duration > 0:
        # If media was absent around the exact wall-clock point, retain the
        # nearest complete evidence window instead of producing an empty file
        # and keeping the entire report in an endless retry cycle.
        latest_start = max(0.0, source_duration - clip_duration)
        if start_offset > latest_start:
            start_offset = latest_start
            mapping += "_boundary_clamped"
    return round(start_offset, 3), round(event_offset, 3), mapping


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


def _normalize_media_timeline(source: Path, output: Path) -> None:
    """Remove timestamp holes introduced by LiveKit track republishing."""
    frame_rate = _media_video_frame_rate(source)
    _run_ffmpeg([
        "-i", str(source), "-map", "0:v:0?", "-map", "0:a:0?",
        "-vf", f"setpts=N/({frame_rate:g}*TB)",
        "-af", "asetpts=N/SR/TB", "-fps_mode:v", "passthrough",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-movflags", "+faststart", str(output),
    ])


def _media_video_frame_rate(path: Path, default: float = 15.0) -> float:
    """Return a sane nominal frame rate without treating timestamp gaps as low FPS."""
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return default
    value = completed.stdout.strip()
    try:
        numerator, denominator = value.split("/", 1)
        frame_rate = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return default
    return frame_rate if 1.0 <= frame_rate <= 60.0 else default


def _media_content_duration_seconds(path: Path) -> float:
    """Measure captured A/V payload duration while ignoring timestamp holes."""
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "packet=stream_index,duration_time",
            "-of", "csv=p=0", str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        return 0.0
    packet_durations: dict[int, list[float]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split(",")
        if len(fields) < 2:
            continue
        try:
            stream = int(fields[0])
            duration = float(fields[1])
        except (TypeError, ValueError):
            continue
        if duration > 0:
            packet_durations.setdefault(stream, []).append(duration)
    if not packet_durations:
        return _media_duration_seconds(path)
    # A packet immediately before a timestamp discontinuity can itself report
    # the entire missing interval as its duration. Median packet duration times
    # packet count measures decoded payload without counting that hole.
    stream_durations = [median(values) * len(values) for values in packet_durations.values()]
    # A complete proctoring recording needs both streams, so its usable length
    # is bounded by the stream with the least actual payload.
    return min(stream_durations)


def _media_duration_seconds(path: Path | None) -> float:
    """Return zero for empty, corrupt, or streamless media."""
    if path is None or not path.is_file() or path.stat().st_size == 0:
        return 0.0
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return 0.0
    try:
        duration = float(completed.stdout.strip())
    except (TypeError, ValueError):
        return 0.0
    return duration if duration > 0 else 0.0


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
