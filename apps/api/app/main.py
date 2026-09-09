from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.assets import router as assets_router
from app.api.identity import compat_router as identity_compat_router
from app.api.identity import router as identity_router
from app.api.identity import get_face_matcher
from app.api.monitor import router as monitor_router
from app.api.sessions import compat_router as sessions_compat_router
from app.api.sessions import router as sessions_router
from app.api.staging import router as staging_router
from app.core.config import get_settings
from app.services.job_queue import JobQueue
from app.services.livekit_egress import LiveKitEgress
from app.services.object_store import ObjectStore
from app.services.state_store import StateStore


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
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-ProctorCore-Upload-Token"],
)

app.include_router(identity_router)
app.include_router(identity_compat_router)
app.include_router(monitor_router)
app.include_router(sessions_router)
app.include_router(sessions_compat_router)
app.include_router(assets_router)
app.include_router(staging_router)


@app.get("/health")
def health() -> dict:
    matcher = get_face_matcher()
    identity_models = _identity_models()
    identity_runtime = matcher.describe_runtime()
    storage = ObjectStore(settings).status()
    queue = JobQueue(settings).status()
    redis_ready = StateStore(settings).redis_ready()
    models_ready = all(model["exists"] for model in identity_models if model["required"])
    egress = LiveKitEgress(settings).status()
    webhook_ready = bool(settings.moodle_webhook_url.strip() and settings.moodle_webhook_secret.strip())
    required_ready = bool(
        storage.get("ready")
        and redis_ready
        and models_ready
        and queue.get("ready")
        and int(queue.get("workers") or 0) > 0
        and egress.get("ready")
        and webhook_ready
    )
    return {
        "ok": required_ready or not settings.storage_require_ready,
        "status": "healthy" if required_ready else "degraded",
        "service": "sental-proctor-service",
        "environment": settings.app_env,
        "identity": identity_runtime,
        "features": {
            "identity_verification": True,
            "identity_engine": settings.identity_engine,
            "identity_models_ready": models_ready,
            "identity_models": identity_models,
            "identity_runtime": identity_runtime,
            "sessions": "redis_with_durable_manifest",
            "violations": "realtime_moodle_events",
            "clips": "livekit_egress_ffmpeg",
            "reports": "moodle_pdf",
            "moodle_webhooks": "signed_durable_outbox",
            "storage": storage,
            "redis": {"ready": redis_ready},
            "worker": queue,
            "egress": egress,
            "webhook": {"ready": webhook_ready, "urlConfigured": bool(settings.moodle_webhook_url.strip())},
        },
    }


@app.get("/api/health")
def api_health() -> dict:
    """Compatibility health route for the Moodle local_proctorcore client."""
    return health()


@app.on_event("startup")
def warm_identity_engine() -> None:
    try:
        storage = ObjectStore(settings)
        storage.ensure_ready()
        if settings.storage_require_ready and not StateStore(settings).redis_ready():
            raise RuntimeError("Redis is required but unavailable.")
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
    return []


def _model_status(value: str, required: bool, role: str) -> dict:
    path = settings.identity_model_root / value if value.strip() else settings.identity_model_root
    return {
        "role": role,
        "path": str(path),
        "required": required,
        "exists": path.is_file() if value.strip() else False,
    }
