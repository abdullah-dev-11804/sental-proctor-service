import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


def test_identity_requires_face() -> None:
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    client = TestClient(app)
    response = client.post(
        "/v1/identity/verify",
        headers={"Authorization": "Bearer dev-secret-change-me"},
        data={
            "company_id": "7",
            "user_id": "123",
            "session_id": "session-test",
        },
        files={
            "live_image": ("live.jpg", encoded.tobytes(), "image/jpeg"),
            "reference_image": ("reference.jpg", encoded.tobytes(), "image/jpeg"),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["identity_status"] == "needs_retry"
    assert body["access_decision"] == "retry"
    assert body["access_allowed"] is False
    assert body["reason"] == "no_face"
