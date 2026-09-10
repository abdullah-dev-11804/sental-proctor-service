import numpy as np

from app.services.advanced_face_engine import AdvancedFaceEngine, AdvancedFaceQuality


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
