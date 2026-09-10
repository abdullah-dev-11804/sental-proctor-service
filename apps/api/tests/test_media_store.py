import base64
from pathlib import Path

import pytest

from app.core.config import Settings
from app.services.media_store import MediaStore


class FakeJobs:
    def __init__(self) -> None:
        self.calls = []

    def enqueue(self, function, *args, **kwargs):
        self.calls.append((function, args, kwargs))
        return {"jobId": kwargs.get("job_id") or "job-1", "status": "queued"}


class FakeEgress:
    def start_participant(self, room, identity, prefix):
        return {"egressId": "egress-1", "objectPrefix": prefix}

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


def test_asset_webhook_uses_a_deterministic_receiver_event_id(store: MediaStore) -> None:
    store.settings.moodle_webhook_url = "https://moodle.example/webhook"
    store.settings.moodle_webhook_secret = "secret"
    session = store.create_session(_payload())

    asset = store.register_asset_bytes(session["id"], "snapshot", b"first", ".jpg", "image/jpeg", "one")

    _function, args, _kwargs = store.jobs.calls[-1]
    assert args[1]["eventId"] == f"asset-{asset['assetId']}"
