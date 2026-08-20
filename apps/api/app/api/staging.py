from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import require_api_auth
from app.services.media_store import MediaStore


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
    data = payload.model_dump()
    data["sessionId"] = data.pop("session_id")
    data["companyId"] = data.pop("company_id")
    data["violationType"] = data.pop("violation_type")
    return MediaStore().record_violation(data)


@router.get("/clips/staging", dependencies=[Depends(require_api_auth)])
def clip_worker_status() -> dict:
    return {
        "ok": True,
        "status": "rolling_browser_chunks",
        "next": "Replace browser chunk concatenation with LiveKit egress plus ffmpeg-normalized clips.",
    }


@router.get("/reports/staging", dependencies=[Depends(require_api_auth)])
def report_worker_status() -> dict:
    return {
        "ok": True,
        "status": "moodle_pdf",
        "next": "Add optional Server B generated PDF if Moodle-side PDF generation is not enough.",
    }
