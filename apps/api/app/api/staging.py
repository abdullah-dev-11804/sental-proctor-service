from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field

from app.core.security import require_api_auth, require_company_scope
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
def record_violation(
    payload: ViolationEventRequest,
    x_proctorcore_company: int | None = Header(None),
) -> dict:
    require_company_scope(payload.company_id, x_proctorcore_company)
    data = payload.model_dump()
    data["sessionId"] = data.pop("session_id")
    data["companyId"] = data.pop("company_id")
    data["violationType"] = data.pop("violation_type")
    return MediaStore().record_violation(data)
