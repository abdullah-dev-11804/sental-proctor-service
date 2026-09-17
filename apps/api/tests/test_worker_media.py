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


def test_violation_clip_rebuilds_continuous_audio_video_timestamps(monkeypatch, tmp_path) -> None:
    session = {
        "startedAt": 100,
        "pendingClips": [{"violationId": "v1", "occurredAt": 120, "segment": 1}],
        "violations": [{"id": "v1", "type": "speech_detected", "occurredAt": 120}],
    }
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    calls = []

    class Store:
        def get_session(self, _session_id):
            return session

        def _save_session(self, updated):
            session.update(updated)

        def register_asset_bytes(self, *_args, **kwargs):
            return {"assetId": "asset-1"}

    def run_ffmpeg(arguments):
        calls.append(arguments)
        (tmp_path / "clip_0000.mp4").write_bytes(b"clip")

    monkeypatch.setattr(jobs, "get_settings", lambda: SimpleNamespace(clip_pre_seconds=15, clip_post_seconds=15))
    monkeypatch.setattr(jobs, "_run_ffmpeg", run_ffmpeg)
    monkeypatch.setattr(jobs, "_media_duration_seconds", lambda _path: 12.0)
    monkeypatch.setattr(jobs, "_media_video_frame_rate", lambda _path: 15.0)

    jobs._create_violation_clips(
        Store(),
        "session-1",
        session,
        source,
        tmp_path,
        {1: {"path": source, "startedAt": 100}},
    )

    command = calls[0]
    assert command[command.index("-vf") + 1] == "setpts=N/(15*TB)"
    assert command[command.index("-af") + 1] == "asetpts=N/SR/TB"
    assert command[command.index("-fps_mode:v") + 1] == "passthrough"


def test_browser_recording_wins_when_egress_duration_is_mostly_timestamp_holes(
    monkeypatch, tmp_path
) -> None:
    egress = tmp_path / "egress.mp4"
    browser = tmp_path / "browser.mp4"
    durations = {egress: 12.8, browser: 27.0}
    monkeypatch.setattr(jobs, "_media_content_duration_seconds", lambda path: durations[path])

    selected, strategy = jobs._prefer_complete_segment(egress, browser)

    assert selected == browser
    assert strategy == "browser_chunk_fallback"
