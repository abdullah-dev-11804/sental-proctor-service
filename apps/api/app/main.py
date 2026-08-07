from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.identity import compat_router as identity_compat_router
from app.api.identity import router as identity_router
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
    detector_model = settings.identity_model_root / settings.identity_yunet_model
    recognizer_model = settings.identity_model_root / settings.identity_sface_model
    return {
        "ok": True,
        "status": "healthy",
        "service": "sental-proctor-service",
        "environment": settings.app_env,
        "features": {
            "identity_verification": True,
            "identity_engine": settings.identity_engine,
            "identity_models_ready": detector_model.is_file() and recognizer_model.is_file(),
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
