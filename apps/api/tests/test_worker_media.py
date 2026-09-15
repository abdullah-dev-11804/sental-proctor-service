from types import SimpleNamespace

import pytest

import app.workers.jobs as jobs
from app.workers.jobs import _group_webm_chunks


def test_mediarecorder_chunks_are_reassembled_and_page_streams_stay_separate() -> None:
    header = b"\x1a\x45\xdf\xa3"

    streams = _group_webm_chunks([
        header + b"page-one-first",
        b"page-one-second",
        header + b"page-two-first",
        b"page-two-second",
    ])

    assert streams == [
        header + b"page-one-firstpage-one-second",
        header + b"page-two-firstpage-two-second",
    ]


def test_missing_violation_clip_keeps_temporary_source_for_retry(monkeypatch, tmp_path) -> None:
    session = {
        "startedAt": 100,
        "pendingClips": [{"violationId": "v1", "occurredAt": 110, "segment": 1}],
        "violations": [{"id": "v1", "type": "tab_hidden", "occurredAt": 110}],
    }

    class Store:
        def get_session(self, _session_id):
            return session

        def _save_session(self, updated):
            session.update(updated)

    monkeypatch.setattr(jobs, "get_settings", lambda: SimpleNamespace(clip_pre_seconds=15, clip_post_seconds=15))
    monkeypatch.setattr(jobs, "_run_ffmpeg", lambda _arguments: None)
    monkeypatch.setattr(jobs, "_media_duration_seconds", lambda _path: 0.0)

    with pytest.raises(RuntimeError, match="violation_clip_creation_failed: v1"):
        jobs._create_violation_clips(
            Store(),
            "session-1",
            session,
            tmp_path / "temporary-recording.mp4",
            tmp_path,
            {},
        )
