import app.main as main
from app.core.config import Settings


class _ReadyService:
    def status(self):
        return {"ready": True, "workers": 1, "private": True}


class _ReadyState:
    def redis_ready(self):
        return True


class _Matcher:
    def describe_runtime(self):
        return {"identity_engine_loaded": "scrfd_adaface"}


def test_health_reports_required_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_face_matcher", lambda: _Matcher())
    monkeypatch.setattr(main, "_identity_models", lambda: [{"required": True, "exists": True}])
    monkeypatch.setattr(main, "ObjectStore", lambda _settings: _ReadyService())
    monkeypatch.setattr(main, "JobQueue", lambda _settings: _ReadyService())
    monkeypatch.setattr(main, "StateStore", lambda _settings: _ReadyState())
    monkeypatch.setattr(main, "LiveKitEgress", lambda _settings: _ReadyService())
    monkeypatch.setattr(main.settings, "moodle_webhook_url", "https://moodle.example/webhook")
    monkeypatch.setattr(main.settings, "moodle_webhook_secret", "test-secret")
    monkeypatch.setattr(main, "_production_checks", lambda _storage, _egress: {"configured": True})

    body = main.health()

    assert body["ok"] is True
    assert body["features"]["worker"]["workers"] == 1


def test_production_checks_reject_local_storage(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "storage_require_ready", True)
    monkeypatch.setattr(main.settings, "reference_encryption_key_file", "/run/secrets/reference_encryption_key")
    monkeypatch.setattr(main.settings, "api_shared_secret", "a" * 32)
    monkeypatch.setattr(main.settings, "livekit_api_secret", "b" * 32)
    monkeypatch.setattr(main.settings, "livekit_url", "wss://proctoring.example")
    monkeypatch.setattr(main.settings, "moodle_webhook_url", "https://moodle.example/webhook")
    monkeypatch.setattr(main.settings, "cors_origins_raw", "https://moodle.example")
    monkeypatch.setattr(main.settings, "identity_temporal_passive_pad_enabled", True)
    monkeypatch.setattr(main.settings, "identity_require_passive_antispoof", True)
    monkeypatch.setattr(main.settings, "identity_active_liveness_enabled", True)
    monkeypatch.setattr(main.settings, "identity_headpose_challenge_enabled", True)
    monkeypatch.setattr(main.settings, "identity_illumination_challenge_enabled", True)

    checks = main._production_checks(
        {"ready": True, "private": True, "backend": "local"},
        {"ready": True, "enabled": True},
    )

    assert checks["private_object_storage"] is False
    assert all(value for key, value in checks.items() if key != "private_object_storage")


def test_production_checks_reject_disabled_active_liveness(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "storage_require_ready", True)
    monkeypatch.setattr(main.settings, "reference_encryption_key_file", "/run/secrets/reference_encryption_key")
    monkeypatch.setattr(main.settings, "api_shared_secret", "a" * 32)
    monkeypatch.setattr(main.settings, "livekit_api_secret", "b" * 32)
    monkeypatch.setattr(main.settings, "livekit_url", "wss://proctoring.example")
    monkeypatch.setattr(main.settings, "moodle_webhook_url", "https://moodle.example/webhook")
    monkeypatch.setattr(main.settings, "cors_origins_raw", "https://moodle.example")
    monkeypatch.setattr(main.settings, "identity_temporal_passive_pad_enabled", True)
    monkeypatch.setattr(main.settings, "identity_require_passive_antispoof", True)
    monkeypatch.setattr(main.settings, "identity_active_liveness_enabled", False)
    monkeypatch.setattr(main.settings, "identity_headpose_challenge_enabled", True)
    monkeypatch.setattr(main.settings, "identity_illumination_challenge_enabled", True)

    checks = main._production_checks(
        {"ready": True, "private": True, "backend": "minio"},
        {"ready": True, "enabled": True},
    )

    assert checks["identity_liveness_enabled"] is False


def test_production_environment_always_enforces_readiness() -> None:
    settings = Settings(_env_file=None, app_env="production", storage_require_ready=False)
    assert settings.production_readiness_required is True


def test_liveness_feature_flags_make_pad_and_headpose_models_required(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "identity_engine", "scrfd_adaface")
    monkeypatch.setattr(main.settings, "identity_temporal_passive_pad_enabled", True)
    monkeypatch.setattr(main.settings, "identity_require_passive_antispoof", False)
    monkeypatch.setattr(main.settings, "identity_active_liveness_enabled", True)
    monkeypatch.setattr(main.settings, "identity_headpose_challenge_enabled", True)
    monkeypatch.setattr(main.settings, "identity_require_headpose_liveness", False)

    models = {model["role"]: model for model in main._identity_models()}

    assert models["passive_antispoof"]["required"] is True
    assert models["head_pose"]["required"] is True
