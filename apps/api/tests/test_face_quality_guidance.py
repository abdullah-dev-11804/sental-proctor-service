from types import SimpleNamespace

from app.services.face_matcher import FaceMatcher, FaceQuality


def _matcher() -> FaceMatcher:
    matcher = FaceMatcher.__new__(FaceMatcher)
    matcher.settings = SimpleNamespace(identity_require_passive_antispoof=False)
    return matcher


def _quality(**overrides) -> FaceQuality:
    values = {
        "brightness": 90.0,
        "blur": 90.0,
        "face_count": 1,
        "width": 1280,
        "height": 720,
        "confidence": 0.90,
    }
    values.update(overrides)
    return FaceQuality(**values)


def test_quality_reason_prioritises_multiple_faces() -> None:
    reason = _matcher()._quality_retry_reason(_quality(face_count=2, brightness=10), 45, 45, 0.75)
    assert reason == "multiple_faces"


def test_quality_reason_reports_light_before_detector_confidence() -> None:
    reason = _matcher()._quality_retry_reason(_quality(brightness=20, confidence=0.68), 45, 45, 0.75)
    assert reason == "low_light"


def test_quality_reason_reports_blur_before_detector_confidence() -> None:
    reason = _matcher()._quality_retry_reason(_quality(blur=20, confidence=0.68), 45, 45, 0.75)
    assert reason == "blurry"


def test_quality_reason_uses_confidence_when_image_quality_passes() -> None:
    reason = _matcher()._quality_retry_reason(_quality(confidence=0.68), 45, 45, 0.75)
    assert reason == "low_face_confidence"
