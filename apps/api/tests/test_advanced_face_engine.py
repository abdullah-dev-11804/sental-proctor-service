import numpy as np
from types import SimpleNamespace

from app.services.advanced_face_engine import AdvancedFaceEngine, AdvancedFaceQuality


class _Input:
    name = "input"


class _AntiSpoofSession:
    def __init__(self, output) -> None:
        self.output = np.asarray(output, dtype=np.float32)

    def get_inputs(self):
        return [_Input()]

    def run(self, _outputs, _inputs):
        return [self.output]


def _antispoof_engine(output, live_class_index=0) -> AdvancedFaceEngine:
    engine = AdvancedFaceEngine.__new__(AdvancedFaceEngine)
    engine.settings = SimpleNamespace(
        identity_antispoof_crop_scale=2.7,
        identity_antispoof_input_size=80,
        identity_antispoof_live_class_index=live_class_index,
    )
    engine.antispoof = _AntiSpoofSession(output)
    return engine


class _Detector:
    def detect(self, _image, threshold, max_num, metric):
        assert threshold == 0.65
        assert max_num == 0
        assert metric == "default"
        return np.asarray([[40, 30, 80, 90, 0.96]], dtype=np.float32), None


def test_quality_analysis_does_not_run_adaface(monkeypatch) -> None:
    engine = AdvancedFaceEngine.__new__(AdvancedFaceEngine)
    quality = AdvancedFaceQuality(
        brightness=100.0,
        blur=90.0,
        face_count=1,
        width=640,
        height=480,
    )
    monkeypatch.setattr(engine, "_detect_and_measure", lambda image: (None, None, quality))
    monkeypatch.setattr(
        engine,
        "_adaface_embedding",
        lambda image: (_ for _ in ()).throw(AssertionError("AdaFace must not run during monitoring")),
    )

    result = engine.analyse(np.zeros((480, 640, 3), dtype=np.uint8))

    assert result is quality


def test_antispoof_does_not_apply_softmax_twice_to_probabilities() -> None:
    engine = _antispoof_engine([[0.91, 0.05, 0.04]])
    image = np.full((120, 120, 3), 127, dtype=np.uint8)
    bbox = np.asarray([30, 30, 90, 90, 0.99], dtype=np.float32)

    assert engine._antispoof_score(image, bbox) == np.float32(0.91)


def test_antispoof_applies_softmax_to_logits() -> None:
    engine = _antispoof_engine([[4.0, 1.0, -1.0]])
    image = np.full((120, 120, 3), 127, dtype=np.uint8)
    bbox = np.asarray([30, 30, 90, 90, 0.99], dtype=np.float32)

    expected = np.exp(4.0) / (np.exp(4.0) + np.exp(1.0) + np.exp(-1.0))
    assert np.isclose(engine._antispoof_score(image, bbox), expected)


def test_upstream_minifasnet_uses_class_one_for_live_face() -> None:
    engine = _antispoof_engine([[0.03, 0.94, 0.03]], live_class_index=1)
    image = np.full((120, 120, 3), 127, dtype=np.uint8)
    bbox = np.asarray([30, 30, 90, 90, 0.99], dtype=np.float32)

    scores = engine._antispoof_scores(image, bbox)

    assert np.isclose(scores["live"], 0.94)
    assert np.isclose(scores["print"], 0.03)
    assert np.isclose(scores["replay"], 0.03)


def test_antispoof_preserves_single_probability_output() -> None:
    engine = _antispoof_engine([[0.93]])
    image = np.full((120, 120, 3), 127, dtype=np.uint8)
    bbox = np.asarray([30, 30, 90, 90, 0.99], dtype=np.float32)

    assert np.isclose(engine._antispoof_score(image, bbox), 0.93)


def test_detected_face_quality_ignores_dark_background(monkeypatch) -> None:
    engine = AdvancedFaceEngine.__new__(AdvancedFaceEngine)
    engine.settings = SimpleNamespace(
        identity_min_face_confidence=0.65,
        identity_antispoof_threshold=0.78,
    )
    engine.detector = _Detector()
    image = np.full((120, 120, 3), 5, dtype=np.uint8)
    image[25:95, 35:85] = 180
    monkeypatch.setattr(engine, "_antispoof_score", lambda _image, _bbox: None)
    monkeypatch.setattr(engine, "_headpose_yaw", lambda _image, _bbox: None)

    _bbox, _keypoints, quality = engine._detect_and_measure(image)

    assert quality.brightness > 100
    assert quality.face_count == 1
