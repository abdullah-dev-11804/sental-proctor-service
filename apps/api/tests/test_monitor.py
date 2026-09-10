import base64
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

import app.api.monitor as monitor_api
from app.services.face_matcher import FaceQuality


class _Store:
    def get_session(self, session_id):
        return {"id": session_id, "companyId": 7}


class _Matcher:
    engine = "scrfd_adaface"

    def __init__(self):
        self.embedding_requests = []

    def analyse_frame(self, content, include_embedding=False):
        self.embedding_requests.append(include_embedding)
        quality = FaceQuality(
            brightness=100.0,
            blur=100.0,
            face_count=1,
            width=640,
            height=480,
            yaw=0.1,
            antispoof_score=0.99,
            antispoof_passed=True,
        )
        embedding = np.ones((1, 4), dtype=np.float32) if include_embedding else None
        return embedding, quality

    def _estimate_yaw(self, quality):
        return quality.yaw

    def _similarity(self, left, right):
        return 1.0


def _payload(reference_image=None):
    encoded = base64.b64encode(b"test-frame-bytes").decode("ascii")
    return monitor_api.FrameAnalysisRequest(
        sessionId="session-1",
        companyId=7,
        frameImage=encoded,
        referenceImage=reference_image,
    )


def test_monitoring_skips_adaface_when_periodic_identity_is_not_requested(monkeypatch) -> None:
    matcher = _Matcher()
    monkeypatch.setattr(monitor_api, "MediaStore", _Store)
    monkeypatch.setattr(monitor_api, "get_face_matcher", lambda: matcher)
    monkeypatch.setattr(
        monitor_api,
        "get_settings",
        lambda: SimpleNamespace(monitor_lookaway_yaw_threshold=0.42, identity_pass_threshold=0.85),
    )

    result = monitor_api.analyse_frame(_payload(), x_proctorcore_company=7)

    assert result["faceCount"] == 1
    assert result["lookingAway"] is False
    assert matcher.embedding_requests == [False]


def test_monitoring_rejects_cross_company_request() -> None:
    with pytest.raises(HTTPException) as exc:
        monitor_api.analyse_frame(_payload(), x_proctorcore_company=8)
    assert exc.value.status_code == 404
