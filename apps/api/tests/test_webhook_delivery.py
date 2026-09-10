from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest

import app.workers.jobs as jobs


class _Store:
    def __init__(self):
        self.deliveries = []
        self.state = self

    @contextmanager
    def session_lock(self, session_id):
        yield

    def get_session(self, session_id):
        return {"id": session_id, "callbackUrl": "https://moodle.example/webhook"}

    def _append_delivery(self, session, event, status, message):
        self.deliveries.append((status, message))


def _settings():
    return SimpleNamespace(
        moodle_webhook_url="https://moodle.example/webhook",
        moodle_webhook_secret="test-secret",
    )


def _response(status_code):
    return httpx.Response(
        status_code,
        text="response",
        request=httpx.Request("POST", "https://moodle.example/webhook"),
    )


def test_retryable_webhook_response_is_raised_for_rq(monkeypatch) -> None:
    store = _Store()
    monkeypatch.setattr(jobs, "get_settings", _settings)
    monkeypatch.setattr(jobs, "MediaStore", lambda: store)
    monkeypatch.setattr(jobs.httpx, "post", lambda *args, **kwargs: _response(503))

    with pytest.raises(httpx.HTTPStatusError):
        jobs.deliver_webhook("session-1", {"eventId": "event-1"})

    assert store.deliveries[-1][0] == "retrying"


def test_permanent_webhook_response_goes_to_dead_letter(monkeypatch) -> None:
    store = _Store()
    monkeypatch.setattr(jobs, "get_settings", _settings)
    monkeypatch.setattr(jobs, "MediaStore", lambda: store)
    monkeypatch.setattr(jobs.httpx, "post", lambda *args, **kwargs: _response(400))

    result = jobs.deliver_webhook("session-1", {"eventId": "event-1"})

    assert result == {"status": "dead_letter", "httpCode": 400}
    assert store.deliveries[-1][0] == "dead_letter"
