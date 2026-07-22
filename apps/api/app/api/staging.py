from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import require_api_auth


router = APIRouter(prefix="/v1", tags=["staging"])


class ViolationEventRequest(BaseModel):
    session_id: str
    company_id: int = Field(ge=0)
    violation_type: str
    severity: str = "warning"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    snapshot_ref: str | None = None


@router.post("/violations", dependencies=[Depends(require_api_auth)])
def record_violation(payload: ViolationEventRequest) -> dict:
    return {
        "ok": True,
        "status": "staged",
        "violation": payload.model_dump(),
        "next": "Persist to Redis/database, publish over WebSocket, and queue clip generation.",
    }


@router.get("/clips/staging", dependencies=[Depends(require_api_auth)])
def clip_worker_status() -> dict:
    return {
        "ok": True,
        "status": "staged",
        "next": "Add delayed ffmpeg worker for T-15/T+15 violation clips.",
    }


@router.get("/reports/staging", dependencies=[Depends(require_api_auth)])
def report_worker_status() -> dict:
    return {
        "ok": True,
        "status": "staged",
        "next": "Add final report JSON/PDF generation and Moodle webhook retries.",
    }
