import base64

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.identity import get_face_matcher
from app.core.config import get_settings
from app.core.security import require_api_auth


router = APIRouter(prefix="/api/v1/monitor", tags=["monitor"])


class FrameAnalysisRequest(BaseModel):
    sessionId: str = Field(min_length=1)
    companyId: int = Field(default=0, ge=0)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    frameImage: str = Field(min_length=16)
    referenceImage: str | None = None


@router.post("/analyse", dependencies=[Depends(require_api_auth)])
def analyse_frame(payload: FrameAnalysisRequest) -> dict:
    matcher = get_face_matcher()
    settings = get_settings()
    try:
        frame = _decode_base64(payload.frameImage)
        image = matcher._decode_image(frame)
        face, quality = matcher._extract_primary_face(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid_frame") from exc

    yaw = matcher._estimate_yaw(quality)
    looking_away = yaw is not None and abs(float(yaw)) >= float(settings.monitor_lookaway_yaw_threshold)
    identity_result = "not_checked"
    similarity_score = None

    if payload.referenceImage:
        try:
            reference = _decode_base64(payload.referenceImage)
            reference_image = matcher._decode_image(reference)
            reference_face, reference_quality = matcher._extract_primary_face(reference_image)
            if quality.face_count == 1 and reference_quality.face_count == 1:
                similarity_score = matcher._similarity(face, reference_face)
                threshold = payload.threshold if payload.threshold is not None else settings.identity_pass_threshold
                identity_result = "matched" if similarity_score >= threshold else "not_matched"
            else:
                identity_result = "not_checked"
        except Exception:
            identity_result = "not_checked"

    return {
        "ok": True,
        "sessionId": payload.sessionId,
        "companyId": payload.companyId,
        "faceCount": int(quality.face_count),
        "lookingAway": bool(looking_away),
        "yaw": float(yaw) if yaw is not None else None,
        "identityResult": identity_result,
        "similarityScore": float(round(similarity_score, 4)) if similarity_score is not None else None,
        "quality": quality.__dict__,
        "engine": matcher.engine,
    }


def _decode_base64(value: str) -> bytes:
    raw = value.strip()
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw, validate=True)
