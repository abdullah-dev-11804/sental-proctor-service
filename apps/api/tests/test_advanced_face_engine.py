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
        identity_antispoof_input_range="raw_255",
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


class _ThresholdDetector(_Detector):
    def __init__(self) -> None:
        self.threshold = None

    def detect(self, _image, threshold, max_num, metric):
        self.threshold = threshold
        return np.asarray([[40, 30, 80, 90, 0.36]], dtype=np.float32), None


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


def test_antispoof_uses_upstream_raw_bgr_pixel_range() -> None:
    engine = _antispoof_engine([[0.03, 0.94, 0.03]], live_class_index=1)
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 20
    image[:, :, 2] = 30

    tensor = engine._antispoof_image_tensor(image)

    assert tensor.shape == (1, 3, 80, 80)
    assert tensor.dtype == np.float32
    assert tensor[0, :, 0, 0].tolist() == [10.0, 20.0, 30.0]


def test_antispoof_unit_range_remains_available_for_other_exports() -> None:
    engine = _antispoof_engine([[0.03, 0.94, 0.03]], live_class_index=1)
    engine.settings.identity_antispoof_input_range = "unit_1"
    image = np.full((80, 80, 3), 255, dtype=np.uint8)

    tensor = engine._antispoof_image_tensor(image)

    assert np.allclose(tensor, 1.0)


def test_antispoof_crop_preserves_the_detected_box_aspect_ratio() -> None:
    engine = AdvancedFaceEngine.__new__(AdvancedFaceEngine)
    image = np.full((300, 300, 3), 127, dtype=np.uint8)
    bbox = np.asarray([100, 75, 150, 175, 0.99], dtype=np.float32)

    crop = engine._crop_bbox_scaled(image, bbox, scale=2.0)

    assert crop.shape[0] == 201
    assert crop.shape[1] == 101


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


def test_headpose_detection_can_use_its_lower_movement_threshold(monkeypatch) -> None:
    engine = AdvancedFaceEngine.__new__(AdvancedFaceEngine)
    engine.settings = SimpleNamespace(
        identity_min_face_confidence=0.65,
        identity_headpose_min_face_confidence=0.30,
        identity_antispoof_threshold=0.78,
    )
    engine.detector = _ThresholdDetector()
    monkeypatch.setattr(engine, "_antispoof_scores", lambda _image, _bbox: None)
    monkeypatch.setattr(engine, "_headpose_yaw", lambda _image, _bbox: -18.0)

    quality = engine.analyse_headpose(np.full((120, 120, 3), 127, dtype=np.uint8))

    assert engine.detector.threshold == 0.30
    assert quality.face_count == 1
    assert quality.headpose_yaw_degrees == -18.0
