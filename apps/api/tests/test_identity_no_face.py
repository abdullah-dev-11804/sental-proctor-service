import asyncio
import cv2
import numpy as np
import pytest
from fastapi import HTTPException

import app.api.identity as identity_api
from app.services.face_matcher import FaceMatchResult, FaceQuality


class _Storage:
    def save_identity_image(self, company_id, session_id, role, content, suffix):
        return f"evidence/{company_id}/{session_id}/{role}{suffix}"


class _NoFaceMatcher:
    def verify(self, live_bytes, reference_bytes):
        quality = FaceQuality(brightness=255.0, blur=0.0, face_count=0, width=320, height=240)
        return FaceMatchResult(
            status="needs_retry",
            decision="retry",
            allowed=False,
            score=0.0,
            reason="no_face",
            live_quality=quality,
            reference_quality=quality,
            engine="scrfd_adaface",
        )


class _Upload:
    content_type = "image/jpeg"

    def __init__(self, content):
        self.content = content

    async def read(self):
        return self.content


def test_identity_requires_face(monkeypatch) -> None:
    monkeypatch.setattr(identity_api, "LocalStorage", _Storage)
    monkeypatch.setattr(identity_api, "get_face_matcher", lambda: _NoFaceMatcher())
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    body = asyncio.run(
        identity_api.verify_identity(
            company_id=7,
            user_id=123,
            session_id="session-test",
            live_image=_Upload(encoded.tobytes()),
            reference_image=_Upload(encoded.tobytes()),
            x_proctorcore_company=7,
        )
    )
    assert body.identity_status == "needs_retry"
    assert body.access_decision == "retry"
    assert body.access_allowed is False
    assert body.reason == "no_face"


def test_identity_rejects_missing_tenant_header(monkeypatch) -> None:
    monkeypatch.setattr(identity_api, "LocalStorage", _Storage)
    monkeypatch.setattr(identity_api, "get_face_matcher", lambda: _NoFaceMatcher())
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            identity_api.verify_identity(
                company_id=7,
                user_id=123,
                session_id="session-test",
                live_image=_Upload(encoded.tobytes()),
                reference_image=_Upload(encoded.tobytes()),
                x_proctorcore_company=None,
            )
        )
    assert exc.value.status_code == 404
