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
    monkeypatch.setattr(jobs, "_media_av_payload_is_aligned", lambda _path: True)

    jobs._create_violation_clips(
        Store(),
        "session-1",
        session,
        source,
        tmp_path,
        {1: {"path": source, "startedAt": 100}},
    )

    command = calls[0]
    assert command[command.index("-vf") + 1] == "setpts=PTS-STARTPTS"
    assert command[command.index("-af") + 1] == "aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS"
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


def test_browser_violation_time_is_mapped_onto_compacted_media() -> None:
    start, event, mapping = jobs._violation_clip_start_offset(
        occurred_at=160,
        source_started_at=100,
        source_duration=60.0,
        clip_duration=30,
        pre_seconds=15,
        media={"strategy": "browser_chunk_fallback", "stoppedAt": 220},
    )

    assert event == 30.0
    assert start == 15.0
    assert mapping == "scaled_compact_timeline"


def test_violation_after_available_media_uses_nearest_complete_window() -> None:
    start, event, mapping = jobs._violation_clip_start_offset(
        occurred_at=250,
        source_started_at=100,
        source_duration=60.0,
        clip_duration=30,
        pre_seconds=15,
        media={"strategy": "livekit_egress_hls", "stoppedAt": 220},
    )

    assert event == 60.0
    assert start == 30.0
    assert mapping == "scaled_compact_timeline_boundary_clamped"


def test_overlapping_violation_windows_are_grouped_once() -> None:
    candidates = [
        {"violationId": "v1", "occurredAt": 100, "segment": 1, "assetId": None},
        {"violationId": "v2", "occurredAt": 112, "segment": 1, "assetId": None},
        {"violationId": "v3", "occurredAt": 180, "segment": 1, "assetId": None},
    ]
    violations = {
        "v1": {"id": "v1", "type": "speech_detected", "occurredAt": 100},
        "v2": {"id": "v2", "type": "multiple_faces", "occurredAt": 112},
        "v3": {"id": "v3", "type": "tab_hidden", "occurredAt": 180},
    }

    groups = jobs._group_clip_candidates(
        candidates,
        violations,
        started_at=90,
        pre_seconds=15,
        post_seconds=15,
    )

    assert [[item["candidate"]["violationId"] for item in group] for group in groups] == [
        ["v1", "v2"],
        ["v3"],
    ]


def test_overlapping_violations_share_one_registered_asset(monkeypatch, tmp_path) -> None:
    session = {
        "startedAt": 100,
        "pendingClips": [
            {"violationId": "v1", "occurredAt": 120, "segment": 1, "assetId": None},
            {"violationId": "v2", "occurredAt": 130, "segment": 1, "assetId": None},
        ],
        "violations": [
            {"id": "v1", "type": "speech_detected", "occurredAt": 120},
            {"id": "v2", "type": "multiple_faces", "occurredAt": 130},
        ],
    }
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    registrations = []

    class Store:
        def get_session(self, _session_id):
            return session

        def _save_session(self, updated):
            session.update(updated)

        def register_asset_bytes(self, *args, **kwargs):
            registrations.append((args, kwargs))
            return {"assetId": "shared-asset"}

    def run_ffmpeg(_arguments):
        (tmp_path / "clip_0000.mp4").write_bytes(b"clip")

    monkeypatch.setattr(jobs, "get_settings", lambda: SimpleNamespace(clip_pre_seconds=15, clip_post_seconds=15))
    monkeypatch.setattr(jobs, "_run_ffmpeg", run_ffmpeg)
    monkeypatch.setattr(jobs, "_media_duration_seconds", lambda _path: 60.0)
    monkeypatch.setattr(jobs, "_media_av_payload_is_aligned", lambda _path: True)

    jobs._create_violation_clips(
        Store(),
        "session-1",
        session,
        source,
        tmp_path,
        {1: {"path": source, "startedAt": 100, "stoppedAt": 160, "strategy": "browser_chunk_fallback"}},
    )

    assert len(registrations) == 1
    assert registrations[0][1]["metadata"]["relatedViolationIds"] == ["v1", "v2"]
    assert {item["assetId"] for item in session["pendingClips"]} == {"shared-asset"}


def test_media_concat_decodes_inputs_instead_of_stream_copy(monkeypatch, tmp_path) -> None:
    calls = []
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    monkeypatch.setattr(jobs, "_run_ffmpeg", calls.append)

    jobs._concat_media_files([first, second], tmp_path / "joined.mp4")

    command = calls[0]
    assert "-filter_complex" in command
    assert "concat=n=2:v=1:a=1[vout][aout]" in command[command.index("-filter_complex") + 1]
    assert not ("-c" in command and command[command.index("-c") + 1] == "copy")


def test_corrupt_repeated_audio_payload_is_rejected(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        jobs,
        "_media_payload_durations",
        lambda _path: {0: 21.38, 1: 0.045},
    )

    assert jobs._media_av_payload_is_aligned(tmp_path / "broken.mp4") is False
