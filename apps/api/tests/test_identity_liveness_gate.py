import base64

from app.api import identity as identity_api


IMAGE = base64.b64encode(b"jpeg" * 100).decode()


class FakeLiveness:
    result = "fail"

    def __init__(self, _matcher):
        pass

    def required_components(self, _enrollment):
        return {"passivePad": True, "headPose": True, "illumination": True}

    def validate(self, *_args, **_kwargs):
        return {
            "overall": self.result,
            "reason": "print_attack" if self.result == "fail" else "ok",
            "challengeId": "challenge-123",
            "passivePad": {
                "result": self.result,
                "aggregate": {"live": 0.03, "print": 0.94, "replay": 0.03},
            },
            "illumination": {"result": self.result},
        }


class FakeMatcher:
    engine = "scrfd_adaface"

    def __init__(self):
        self.enrollment_calls = 0

    def select_enrollment_reference(self, frames, _left, _right, policy):
        self.enrollment_calls += 1
        assert frames == [b"jpeg" * 100]
        assert policy["skipPassivePad"] is True
        return {
            "ok": True,
            "result": "enrolled",
            "accessAllowed": True,
            "similarityScore": 1.0,
            "bestReferenceBytes": frames[0],
            "template": {"embeddings": [[0.1, 0.2]], "meanEmbedding": [0.1, 0.2], "quality": {}},
            "quality": {},
            "reason": "ok",
            "engine": "scrfd_adaface-enrollment",
        }


class FakeStorage:
    saved_template = None

    def delete_face_reference(self, _company_id, _user_id):
        return []

    def save_face_template(self, _company_id, _user_id, template, reference):
        FakeStorage.saved_template = (template, reference)
        return "reference-id", "template-key", "image-key"


def payload():
    return identity_api.FaceReferenceEnrollRequest(
        transactionId="transaction-123",
        companyId=3,
        userId=245,
        fullName="Test User",
        confirmedAt=123456,
        centerImages=[IMAGE],
        contextId="quiz:10",
        challengeId="challenge-123",
        challengeNonce="nonce-12345678",
        livenessEvidence=[{"image": IMAGE, "capturedAtMs": 123456, "elapsedMs": 100}],
    )


def test_adaface_template_builder_is_not_called_when_liveness_fails(monkeypatch):
    matcher = FakeMatcher()
    FakeLiveness.result = "fail"
    monkeypatch.setattr(identity_api, "LivenessService", FakeLiveness)
    monkeypatch.setattr(identity_api, "get_face_matcher", lambda: matcher)

    result = identity_api.enroll_face_reference(payload(), x_proctorcore_company=3)

    assert result["result"] == "spoof_detected"
    assert matcher.enrollment_calls == 0


def test_successful_liveness_preserves_template_enrollment(monkeypatch):
    matcher = FakeMatcher()
    FakeLiveness.result = "pass"
    FakeStorage.saved_template = None
    monkeypatch.setattr(identity_api, "LivenessService", FakeLiveness)
    monkeypatch.setattr(identity_api, "get_face_matcher", lambda: matcher)
    monkeypatch.setattr(identity_api, "LocalStorage", FakeStorage)

    result = identity_api.enroll_face_reference(payload(), x_proctorcore_company=3)

    assert result["result"] == "enrolled"
    assert matcher.enrollment_calls == 1
    assert FakeStorage.saved_template[0]["meanEmbedding"] == [0.1, 0.2]
    assert FakeStorage.saved_template[1] == b"jpeg" * 100


def test_uncorroborated_passive_failure_is_not_reported_as_proven_spoof(monkeypatch):
    monkeypatch.setattr(identity_api, "get_face_matcher", lambda: FakeMatcher())
    liveness = {
        "overall": "fail",
        "reason": "headpose_sequence_failed",
        "passivePad": {
            "result": "fail",
            "aggregate": {"live": 0.03, "print": 0.94, "replay": 0.03},
        },
        "illumination": {"result": "pass"},
    }

    result = identity_api._liveness_retry_response(payload(), 0.85, "enrollment", liveness)

    assert result["result"] == "liveness_failed"
