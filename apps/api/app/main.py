from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.identity import compat_router as identity_compat_router
from app.api.identity import router as identity_router
from app.api.identity import get_face_matcher
from app.api.sessions import compat_router as sessions_compat_router
from app.api.sessions import router as sessions_router
from app.api.staging import router as staging_router
from app.core.config import get_settings


settings = get_settings()

app = FastAPI(
    title="SENTAL Proctor Service",
    version="0.1.0",
    description="Server B / AI-media service for hybrid Moodle proctoring.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

app.include_router(identity_router)
app.include_router(identity_compat_router)
app.include_router(sessions_router)
app.include_router(sessions_compat_router)
app.include_router(staging_router)


@app.get("/health")
def health() -> dict:
    matcher = get_face_matcher()
    identity_models = _identity_models()
    identity_runtime = matcher.describe_runtime()
    return {
        "ok": True,
        "status": "healthy",
        "service": "sental-proctor-service",
        "environment": settings.app_env,
        "identity": identity_runtime,
        "features": {
            "identity_verification": True,
            "identity_engine": settings.identity_engine,
            "identity_models_ready": all(model["exists"] for model in identity_models if model["required"]),
            "identity_models": identity_models,
            "identity_runtime": identity_runtime,
            "sessions": "staged",
            "violations": "staged",
            "clips": "staged",
            "reports": "staged",
            "moodle_webhooks": "staged",
        },
    }


@app.get("/api/health")
def api_health() -> dict:
    """Compatibility health route for the Moodle local_proctorcore client."""
    return health()


@app.on_event("startup")
def warm_identity_engine() -> None:
    try:
        matcher = get_face_matcher()
        runtime = matcher.describe_runtime()
        print(f"[startup] identity runtime: {runtime}")
    except Exception as exc:
        print(f"[startup] identity engine warmup failed: {exc}")


def _identity_models() -> list[dict]:
    engine = settings.identity_engine.strip().lower()
    if engine in ("scrfd_adaface", "production_face"):
        models = [
            _model_status(settings.identity_scrfd_model, True, "scrfd_detector"),
            _model_status(settings.identity_adaface_model, True, "adaface_recognizer"),
        ]
        if settings.identity_antispoof_model.strip() or settings.identity_require_passive_antispoof:
            models.append(_model_status(settings.identity_antispoof_model, settings.identity_require_passive_antispoof, "passive_antispoof"))
        if settings.identity_headpose_model.strip() or settings.identity_require_headpose_liveness:
            models.append(_model_status(settings.identity_headpose_model, settings.identity_require_headpose_liveness, "head_pose"))
        return models
    return [
        _model_status(settings.identity_yunet_model, engine == "opencv_sface", "yunet_detector"),
        _model_status(settings.identity_sface_model, engine == "opencv_sface", "sface_recognizer"),
    ]


def _model_status(value: str, required: bool, role: str) -> dict:
    path = settings.identity_model_root / value if value.strip() else settings.identity_model_root
    return {
        "role": role,
        "path": str(path),
        "required": required,
        "exists": path.is_file() if value.strip() else False,
    }
