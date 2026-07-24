import base64
import binascii

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.core.security import require_api_auth
from app.models.identity import IdentityVerifyResponse
from app.services.face_matcher import FaceMatcher
from app.services.storage import LocalStorage


router = APIRouter(prefix="/v1/identity", tags=["identity"])
compat_router = APIRouter(prefix="/api/v1/identity", tags=["moodle-compat"])


class MoodleIdentityVerifyRequest(BaseModel):
    transactionId: str = Field(min_length=8, max_length=128)
    companyId: int = Field(default=0, ge=0)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    referenceImage: str = Field(min_length=16)
    centerImage: str = Field(min_length=16)
    leftImage: str | None = None
    rightImage: str | None = None
    centerImages: list[str] | None = None
    leftImages: list[str] | None = None
    rightImages: list[str] | None = None


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


@compat_router.post("/verify", dependencies=[Depends(require_api_auth)])
def verify_moodle_identity(payload: MoodleIdentityVerifyRequest) -> dict:
    """JSON/base64 identity route used by current Moodle local_proctorcore.

    Moodle performs the browser challenge and sends three images. Server B
    compares the center frame to the Moodle profile reference and checks that
    the left/right frames contain enough head movement for active liveness.
    """
    reference_bytes = _decode_base64_image(payload.referenceImage)
    center_frames = _decode_base64_images(payload.centerImages, payload.centerImage)
    left_frames = _decode_optional_base64_images(payload.leftImages, payload.leftImage)
    right_frames = _decode_optional_base64_images(payload.rightImages, payload.rightImage)

    try:
        matcher = FaceMatcher()
        if left_frames and right_frames:
            result = matcher.verify_sequence_challenge(
                center_frames,
                left_frames,
                right_frames,
                reference_bytes,
                payload.threshold,
            )
        else:
            result = matcher.verify_center_sequence(center_frames, reference_bytes, payload.threshold)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": str(exc), "message": "One or both images could not be decoded."},
        ) from exc

    threshold = payload.threshold if payload.threshold is not None else 0.72
    result["transactionId"] = payload.transactionId
    result["threshold"] = threshold
    return result


def _suffix(file: UploadFile) -> str:
    content_type = (file.content_type or "").lower()
    if content_type == "image/png":
        return ".png"
    if content_type == "image/webp":
        return ".webp"
    return ".jpg"


def _decode_base64_image(value: str) -> bytes:
    clean = value.strip()
    if "," in clean:
        clean = clean.split(",", 1)[1]
    try:
        content = base64.b64decode(clean, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_base64_image", "message": "Image is not valid base64."},
        ) from exc
    if len(content) < 256 or len(content) > 8 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_image_size", "message": "Image size is outside the allowed range."},
        )
    return content


def _decode_base64_images(values: list[str] | None, fallback: str | None) -> list[bytes]:
    source = values if values else ([fallback] if fallback else [])
    frames = [_decode_base64_image(value) for value in source if value]
    if not frames:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "missing_image", "message": "At least one challenge frame is required."},
        )
    return frames


def _decode_optional_base64_images(values: list[str] | None, fallback: str | None) -> list[bytes]:
    source = values if values else ([fallback] if fallback else [])
    return [_decode_base64_image(value) for value in source if value]
