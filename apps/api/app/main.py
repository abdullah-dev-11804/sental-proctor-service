from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.identity import router as identity_router
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
app.include_router(sessions_router)
app.include_router(staging_router)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "service": "sental-proctor-service",
        "environment": settings.app_env,
        "features": {
            "identity_verification": True,
            "sessions": "staged",
            "violations": "staged",
            "clips": "staged",
            "reports": "staged",
            "moodle_webhooks": "staged",
        },
    }
