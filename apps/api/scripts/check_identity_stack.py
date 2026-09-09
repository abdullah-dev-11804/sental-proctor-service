#!/usr/bin/env python3
from __future__ import annotations

import json

from app.core.config import get_settings
from app.services.face_matcher import FaceMatcher


def main() -> None:
    settings = get_settings()
    matcher = FaceMatcher()
    required = required_models(settings, matcher.engine)
    print(json.dumps({
        "ok": True,
        "engine": matcher.engine,
        "requiredModels": required,
        "modelsReady": all(item["exists"] for item in required),
        "passThreshold": settings.identity_pass_threshold,
        "reviewThreshold": settings.identity_review_threshold,
        "activeLiveness": settings.identity_require_active_liveness,
        "passiveAntispoof": settings.identity_require_passive_antispoof,
        "headposeLiveness": settings.identity_require_headpose_liveness,
    }, indent=2))


def required_models(settings, engine: str) -> list[dict]:
    if engine.strip().lower() != "scrfd_adaface":
        raise RuntimeError("Only the SCRFD + AdaFace identity engine is supported.")
    values = [
        ("scrfd_detector", settings.identity_scrfd_model, True),
        ("adaface_recognizer", settings.identity_adaface_model, True),
        ("passive_antispoof", settings.identity_antispoof_model, settings.identity_require_passive_antispoof),
        ("head_pose", settings.identity_headpose_model, settings.identity_require_headpose_liveness),
    ]
    models = []
    for role, value, required in values:
        if not required and not str(value).strip():
            continue
        path = settings.identity_model_root / value
        models.append({
            "role": role,
            "path": str(path),
            "required": bool(required),
            "exists": path.is_file(),
        })
    return models


if __name__ == "__main__":
    main()
