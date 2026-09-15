from datetime import timedelta
from types import SimpleNamespace

import redis
import rq

from app.core.config import Settings
from app.services.job_queue import JobQueue


def test_enqueue_uses_rq_explicit_call_contract(monkeypatch) -> None:
    captured = {}

    class FakeQueue:
        def __init__(self, name, connection, default_timeout):
            self.name = name

        def enqueue_call(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(id=kwargs["job_id"], get_status=lambda refresh=False: "queued")

    monkeypatch.setattr(redis.Redis, "from_url", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(rq, "Queue", FakeQueue)
    settings = Settings(_env_file=None, app_env="test", storage_require_ready=False)

    result = JobQueue(settings).enqueue(
        "app.workers.jobs.deliver_webhook",
        "session-1",
        {"eventId": "event-1"},
        job_id="webhook-event-1",
        retry=True,
    )

    assert captured["func"] == "app.workers.jobs.deliver_webhook"
    assert captured["args"] == ("session-1", {"eventId": "event-1"})
    assert captured["kwargs"] == {}
    assert result == {"jobId": "webhook-event-1", "queue": settings.queue_name, "status": "queued"}


def test_enqueue_can_schedule_reconciliation_cleanup(monkeypatch) -> None:
    captured = {}

    class FakeQueue:
        def __init__(self, name, connection, default_timeout):
            self.name = name

        def enqueue_in(self, delay, function, *args, **kwargs):
            captured.update({"delay": delay, "function": function, "args": args, "kwargs": kwargs})
            return SimpleNamespace(id=kwargs["job_id"], get_status=lambda refresh=False: "scheduled")

    monkeypatch.setattr(redis.Redis, "from_url", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(rq, "Queue", FakeQueue)
    settings = Settings(_env_file=None, app_env="test", storage_require_ready=False)

    result = JobQueue(settings).enqueue(
        "app.workers.jobs.cleanup_temporary_media",
        "session-1",
        job_id="cleanup-session-1",
        retry=True,
        delay_seconds=300,
    )

    assert captured["delay"] == timedelta(seconds=300)
    assert captured["function"] == "app.workers.jobs.cleanup_temporary_media"
    assert captured["args"] == ("session-1",)
    assert result["status"] == "scheduled"
