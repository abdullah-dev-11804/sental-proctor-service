from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.security import require_api_auth
from app.models.identity import IdentityVerifyResponse
from app.services.face_matcher import FaceMatcher
from app.services.storage import LocalStorage


router = APIRouter(prefix="/v1/identity", tags=["identity"])


@router.post(
    "/verify",
    response_model=IdentityVerifyResponse,
    dependencies=[Depends(require_api_auth)],
)
async def verify_identity(
    company_id: int = Form(..., ge=0),
    user_id: int = Form(..., ge=1),
    session_id: str = Form(..., min_length=3, max_length=128),
    live_image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
) -> IdentityVerifyResponse:
    live_bytes = await live_image.read()
    reference_bytes = await reference_image.read()

    if not live_bytes or not reference_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "missing_image", "message": "Both live_image and reference_image are required."},
        )

    storage = LocalStorage()
    live_key = storage.save_identity_image(company_id, session_id, "live", live_bytes, _suffix(live_image))
    reference_key = storage.save_identity_image(company_id, session_id, "reference", reference_bytes, _suffix(reference_image))

    try:
        result = FaceMatcher().verify(live_bytes, reference_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": str(exc), "message": "One or both images could not be decoded."},
        ) from exc

    return IdentityVerifyResponse(
        identity_status=result.status,
        access_decision=result.decision,
        access_allowed=result.allowed,
        match_score=result.score,
        reason=result.reason,
        live_snapshot_key=live_key,
        reference_snapshot_key=reference_key,
        quality={
            "live": result.live_quality.__dict__,
            "reference": result.reference_quality.__dict__,
        },
        engine=result.engine,
    )


def _suffix(file: UploadFile) -> str:
    content_type = (file.content_type or "").lower()
    if content_type == "image/png":
        return ".png"
    if content_type == "image/webp":
        return ".webp"
    return ".jpg"
