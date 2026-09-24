import base64
from pathlib import Path

import pytest

import app.workers.jobs as worker_jobs
from app.core.config import Settings
from app.services.media_store import MediaStore


class FakeJobs:
    def __init__(self) -> None:
        self.calls = []

    def enqueue(self, function, *args, **kwargs):
        self.calls.append((function, args, kwargs))
        return {"jobId": kwargs.get("job_id") or "job-1", "status": "queued"}


class FakeEgress:
    def start_participant(self, room, identity, prefix, **kwargs):
        return {"egressId": "egress-1", "objectPrefix": prefix, "screenShare": kwargs.get("screen_share", False)}

    def stop(self, egress_id):
        return {"state": "stopping", "egressId": egress_id}


@pytest.fixture()
def store(tmp_path: Path, monkeypatch) -> MediaStore:
    settings = Settings(
        _env_file=None,
        local_storage_root=tmp_path,
        storage_backend="local",
        storage_require_ready=False,
        livekit_egress_enabled=True,
        moodle_webhook_url="",
        moodle_webhook_secret="",
    )
    monkeypatch.setattr("app.services.media_store.get_settings", lambda: settings)
    service = MediaStore()
    service.jobs = FakeJobs()
    service.egress = FakeEgress()
    return service


def _payload(company_id: int = 7) -> dict:
    return {
        "companyId": company_id,
        "moodleSessionId": 11,
        "attemptId": 22,
        "user": {"id": 33},
        "retention": {"videoDays": 10, "reportDays": 183, "appealDays": 14},
    }


def _scope() -> dict:
    return {"companyId": 7, "moodleSessionId": 11, "attemptId": 22, "userId": 33}


def test_interrupt_preserves_segment_without_finalizing(store: MediaStore) -> None:
    session = store.create_session(_payload())
    store.create_media_token(session["id"], _scope())
    started = store.start_recording(session["id"], {**_scope(), "segment": 1})

    stopped = store.stop_recording(
        session["id"],
        {**_scope(), "segment": 1, "reason": "connection_lost", "result": "failed"},
    )
    current = store.get_session(session["id"])

    assert started["provider"] == "livekit_egress"
    assert stopped["status"] == "interrupted"
    assert current["status"] == "interrupted"
    assert current["completedAt"] is None
    assert store.jobs.calls == []


def test_resume_uses_a_new_segment_then_submission_queues_finalization(store: MediaStore) -> None:
    session = store.create_session(_payload())
    store.start_recording(session["id"], {**_scope(), "segment": 1})
    store.stop_recording(session["id"], {**_scope(), "segment": 1, "reason": "browser_offline"})
    store.resume_session(session["id"], _scope())
    store.start_recording(session["id"], {**_scope(), "segment": 2})
    result = store.stop_recording(session["id"], {**_scope(), "segment": 2, "reason": "submitted"})

    assert result["status"] == "processing"
    assert store.get_session(session["id"])["processing"]["state"] == "queued"
    assert store.jobs.calls[0][0] == "app.workers.jobs.finalize_session_media"


def test_reopened_fallback_segment_continues_chunk_sequence(store: MediaStore) -> None:
    session = store.create_session(_payload())
    store.egress = type("FailedEgress", (), {
        "start_participant": lambda *_args: {"state": "failed"},
        "stop": lambda *_args: {"state": "stopped"},
    })()
    token = store.create_media_token(session["id"], _scope())
    started = store.start_recording(session["id"], {**_scope(), "segment": 1})
    store.save_media_chunk(
        session["id"],
        token["uploadToken"],
        b"chunk",
        segment=1,
        sequence=1,
        duration_ms=5000,
        mime_type="video/webm",
    )

    resumed = store.start_recording(session["id"], {**_scope(), "segment": 1})

    assert started["fallback"] is True
    assert resumed["duplicate"] is True
    assert resumed["fallback"] is True
    assert resumed["nextSequence"] == 2


def test_screen_capture_uses_independent_token_chunks_and_egress(store: MediaStore) -> None:
    session = store.create_session(_payload())
    camera = store.create_media_token(session["id"], {**_scope(), "mediaRole": "camera"})
    screen = store.create_media_token(session["id"], {
        **_scope(),
        "mediaRole": "screen",
        "participantIdentity": "candidate-screen",
    })

    started = store.start_screen_recording(session["id"], {**_scope(), "displaySurface": "monitor"})
    saved = store.save_media_chunk(
        session["id"],
        screen["uploadToken"],
        b"screen-chunk",
        segment=started["segment"],
        sequence=0,
        duration_ms=5000,
        mime_type="video/webm",
        stream_type="screen",
    )

    assert started["provider"] == "livekit_egress"
    assert saved["chunk"]["stream"] == "screen"
    assert len(store.get_session(session["id"])["screenChunks"]) == 1
    assert store.get_session(session["id"])["chunks"] == []
    with pytest.raises(PermissionError, match="invalid_upload_token"):
        store.save_media_chunk(
            session["id"], camera["uploadToken"], b"wrong-stream", segment=1, sequence=1,
            mime_type="video/webm", stream_type="screen",
        )


def test_retention_begins_at_completion_and_classifies_assets(store: MediaStore) -> None:
    session = store.create_session(_payload())
    assert session["retention"]["videoExpiresAt"] is None
    assert session["retention"]["reportExpiresAt"] is None

    snapshot = store.register_asset_bytes(session["id"], "snapshot", b"snapshot", ".jpg", "image/jpeg", "violation")
    clip = store.register_asset_bytes(session["id"], "video_clip", b"clip", ".mp4", "video/mp4", "violation")
    current = store.get_session(session["id"])
    store._finalize_session(current, result="passed", reason="submitted")
    completed = store.get_session(session["id"])
    assets = {item["assetId"]: item for item in completed["assets"]}

    assert completed["retention"]["videoExpiresAt"] >= completed["retention"]["appealUntil"]
    assert assets[snapshot["assetId"]]["expiresAt"] == completed["retention"]["reportExpiresAt"]
    assert assets[clip["assetId"]]["expiresAt"] == completed["retention"]["videoExpiresAt"]


def test_snapshot_data_url_is_decoded_and_tenant_scope_is_enforced(store: MediaStore) -> None:
    session = store.create_session(_payload())
    encoded = "data:image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode("ascii")
    captured = store.capture_snapshot(session["id"], {**_scope(), "reason": "submission", "snapshotImage": encoded})

    assert captured["status"] == "captured"
    with pytest.raises(PermissionError, match="tenant_scope_mismatch"):
        store.start_session(session["id"], {**_scope(), "companyId": 8})


def test_verified_identity_frame_is_stored_as_identity_evidence(store: MediaStore) -> None:
    session = store.create_session(_payload())
    accepted_frame = b"accepted-verification-frame"
    encoded = "data:image/jpeg;base64," + base64.b64encode(accepted_frame).decode("ascii")

    captured = store.capture_snapshot(session["id"], {
        **_scope(),
        "reason": "identity_verification",
        "snapshotImage": encoded,
    })
    asset = next(
        item for item in store.get_session(session["id"])["assets"]
        if item["assetId"] == captured["assetId"]
    )
    content, mime_type = store.get_asset_content(captured["assetId"])

    assert asset["type"] == "identity_photo"
    assert asset["reason"] == "identity_verification"
    assert content == accepted_frame
    assert mime_type == "image/jpeg"


def test_duplicate_violation_does_not_queue_duplicate_clip(store: MediaStore) -> None:
    session = store.create_session(_payload())
    payload = {
        **_scope(),
        "sessionId": session["id"],
        "violationId": "violation-1",
        "violationType": "tab_hidden",
        "occurredAt": 100,
    }

    store.record_violation(payload)
    store.record_violation(payload)

    pending = store.get_session(session["id"])["pendingClips"]
    assert len(pending) == 1


def test_finalization_retains_only_key_evidence_and_schedules_source_cleanup(
    store: MediaStore, monkeypatch
) -> None:
    session = store.create_session(_payload())

    def build_recording(_session, _objects, work):
        recording = work / "recording.mp4"
        recording.write_bytes(b"temporary-full-session")
        return recording, "browser_chunk_fallback", {}

    def create_clips(service, session_id, *_args):
        service.register_asset_bytes(
            session_id,
            "video_clip",
            b"clip",
            ".mp4",
            "video/mp4",
            "tab_hidden",
            violation_id="violation-1",
        )

    monkeypatch.setattr(worker_jobs, "get_settings", lambda: store.settings)
    monkeypatch.setattr(worker_jobs, "MediaStore", lambda: store)
    monkeypatch.setattr(worker_jobs, "ObjectStore", lambda _settings: store.objects)
    monkeypatch.setattr(worker_jobs, "_build_recording", build_recording)
    monkeypatch.setattr(worker_jobs, "_create_violation_clips", create_clips)

    result = worker_jobs.finalize_session_media(session["id"])
    current = store.get_session(session["id"])

    assert result["status"] == "completed"
    assert [asset["type"] for asset in current["assets"]] == ["video_clip"]
    assert current["temporaryMedia"]["state"] == "awaiting_reconciliation"
    full_recording = current["temporaryMedia"]["fullRecording"]
    assert store.objects.exists(full_recording["bucket"], full_recording["objectKey"])
    assert store.jobs.calls[-1][0] == "app.workers.jobs.cleanup_temporary_media"
    assert store.jobs.calls[-1][2]["delay_seconds"] == store.settings.media_reconciliation_grace_seconds


def test_deleting_one_asset_does_not_hide_other_assets(store: MediaStore) -> None:
    session = store.create_session(_payload())
    first = store.register_asset_bytes(session["id"], "snapshot", b"first", ".jpg", "image/jpeg", "one")
    second = store.register_asset_bytes(session["id"], "snapshot", b"second", ".jpg", "image/jpeg", "two")

    store.delete_asset(first["assetId"])
    assets = {item["assetId"]: item for item in store.get_session(session["id"])["assets"]}

    assert assets[first["assetId"]]["status"] == "deleted"
    assert assets[second["assetId"]]["status"] == "active"


def test_evidence_hold_updates_every_indexed_asset(store: MediaStore) -> None:
    session = store.create_session(_payload())
    first = store.register_asset_bytes(session["id"], "snapshot", b"first", ".jpg", "image/jpeg", "one")
    second = store.register_asset_bytes(session["id"], "video_clip", b"second", ".mp4", "video/mp4", "two")

    retention = store.set_evidence_hold(session["id"], True, "appeal_open")

    assert retention["held"] is True
    assert store.state.get_asset(first["assetId"])["held"] is True
    assert store.state.get_asset(second["assetId"])["held"] is True


def test_reconciliation_marks_only_the_missing_physical_object(store: MediaStore) -> None:
    session = store.create_session(_payload())
    missing = store.register_asset_bytes(session["id"], "snapshot", b"first", ".jpg", "image/jpeg", "one")
    available = store.register_asset_bytes(session["id"], "snapshot", b"second", ".jpg", "image/jpeg", "two")
    store.objects.delete(missing["bucket"], missing["objectKey"])

    result = store.reconcile_session(session["id"])

    assert result["missing"] == [missing["assetId"]]
    assert result["available"] == [available["assetId"]]
    assert store.state.get_asset(missing["assetId"])["status"] == "missing"
    assert store.state.get_asset(available["assetId"])["status"] == "active"


def test_temporary_media_waits_for_moodle_delivery_then_is_deleted(store: MediaStore) -> None:
    session = store.create_session(_payload())
    clip = store.register_asset_bytes(
        session["id"], "video_clip", b"clip", ".mp4", "video/mp4", "tab_hidden"
    )
    temporary = store.objects.put_bytes(
        store.settings.s3_bucket_temp,
        f"temp/7/{session['id']}/browser/segment_001/chunk_000001.webm",
        b"temporary",
        "video/webm",
    )
    full_recording = store.objects.put_bytes(
        store.settings.s3_bucket_temp,
        f"temp/7/{session['id']}/finalized/full-session.mp4",
        b"temporary-full-session",
        "video/mp4",
    )
    session = store.get_session(session["id"])
    session["chunks"] = [{"bucket": temporary.bucket, "objectKey": temporary.key}]
    session["temporaryMedia"] = {
        "state": "awaiting_reconciliation",
        "cleanupNotBefore": 0,
        "fullRecording": {"bucket": full_recording.bucket, "objectKey": full_recording.key},
        "requiredAssetIds": [clip["assetId"]],
        "requiredEventIds": [f"asset-{clip['assetId']}", f"final-{session['id']}"],
    }
    store._save_session(session)

    with pytest.raises(RuntimeError, match="webhooks_pending"):
        store.reconcile_and_cleanup_temporary_media(session["id"])
    assert store.objects.exists(temporary.bucket, temporary.key)

    session = store.get_session(session["id"])
    session["webhookDeliveries"] = [
        {"eventId": f"asset-{clip['assetId']}", "status": "delivered"},
        {"eventId": f"final-{session['id']}", "status": "delivered"},
    ]
    store._save_session(session)

    receipt = store.reconcile_and_cleanup_temporary_media(session["id"])

    assert receipt["deletedBrowserChunks"] == 1
    assert receipt["deletedFullRecording"] is True
    assert not store.objects.exists(temporary.bucket, temporary.key)
    assert not store.objects.exists(full_recording.bucket, full_recording.key)
    assert store.get_session(session["id"])["temporaryMedia"]["state"] == "deleted"


def test_asset_webhook_uses_a_deterministic_receiver_event_id(store: MediaStore) -> None:
    store.settings.moodle_webhook_url = "https://moodle.example/webhook"
    store.settings.moodle_webhook_secret = "secret"
    session = store.create_session(_payload())

    asset = store.register_asset_bytes(session["id"], "snapshot", b"first", ".jpg", "image/jpeg", "one")

    _function, args, _kwargs = store.jobs.calls[-1]
    assert args[1]["eventId"] == f"asset-{asset['assetId']}"
