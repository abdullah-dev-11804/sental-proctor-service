from uuid import uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import require_api_auth


router = APIRouter(prefix="/v1/sessions", tags=["sessions"])
compat_router = APIRouter(prefix="/api/v1/sessions", tags=["moodle-compat"])


class SessionStartRequest(BaseModel):
    company_id: int = Field(ge=0)
    moodle_session_id: int | None = Field(default=None, ge=1)
    quiz_attempt_id: int | None = Field(default=None, ge=1)
    user_id: int = Field(ge=1)
    course_id: int | None = Field(default=None, ge=1)
    quiz_id: int | None = Field(default=None, ge=1)


class FinishSessionRequest(BaseModel):
    reason: str = "completed"


class SnapshotUploadUrlRequest(BaseModel):
    purpose: str = "identity"
    violation_id: str | None = None
    content_type: str = "image/jpeg"


class MoodleSessionCreateRequest(BaseModel):
    moodleSessionId: int | None = Field(default=None, ge=1)
    companyId: int = Field(default=0, ge=0)
    courseId: int | None = Field(default=None, ge=1)
    cmid: int | None = Field(default=None, ge=1)
    quizId: int | None = Field(default=None, ge=1)
    attemptId: int | None = Field(default=None, ge=1)
    attemptNumber: int | None = Field(default=None, ge=1)
    user: dict | None = None
    callbackUrl: str | None = None
    returnUrl: str | None = None
    timer: dict | None = None
    source: dict | None = None


class MoodleSessionStartRequest(BaseModel):
    moodleSessionId: int | None = Field(default=None, ge=1)
    companyId: int = Field(default=0, ge=0)
    attemptId: int | None = Field(default=None, ge=1)
    userId: int | None = Field(default=None, ge=1)
    startedAt: str | None = None
    source: str | None = None


@router.post("/start", dependencies=[Depends(require_api_auth)])
def start_session(payload: SessionStartRequest) -> dict:
    session_id = str(uuid4())
    return {
        "status": "pending_preflight",
        "session_id": session_id,
        "company_id": payload.company_id,
        "moodle_session_id": payload.moodle_session_id,
        "quiz_attempt_id": payload.quiz_attempt_id,
        "ws_token": "staged-ws-token",
        "livekit_url": "staged-livekit-url",
        "livekit_token": "staged-livekit-token",
        "next": "Wire this route to Redis session state and LiveKit room creation.",
    }


@compat_router.post("", dependencies=[Depends(require_api_auth)])
def create_moodle_session(payload: MoodleSessionCreateRequest) -> dict:
    """Moodle local_proctorcore compatibility route.

    Moodle creates the official attempt record first, then asks Server B for an
    external proctoring session id. This staged response keeps that contract in
    place until Redis/LiveKit-backed session state is implemented.
    """
    session_id = str(uuid4())
    room_id = f"room-{session_id}"
    session = {
        "id": session_id,
        "sessionId": session_id,
        "roomId": room_id,
        "status": "created",
        "companyId": payload.companyId,
        "moodleSessionId": payload.moodleSessionId,
        "attemptId": payload.attemptId,
        "quizId": payload.quizId,
        "courseId": payload.courseId,
        "media": {
            "livekitUrl": "staged-livekit-url",
            "roomId": room_id,
        },
    }
    return {
        "ok": True,
        "status": "created",
        "session": session,
    }


@compat_router.get("/{session_id}", dependencies=[Depends(require_api_auth)])
def get_moodle_session(session_id: str) -> dict:
    return {
        "ok": True,
        "session": {
            "id": session_id,
            "sessionId": session_id,
            "status": "staged",
        },
    }


@compat_router.post("/{session_id}/start", dependencies=[Depends(require_api_auth)])
def start_moodle_session(session_id: str, payload: MoodleSessionStartRequest) -> dict:
    return {
        "ok": True,
        "session": {
            "id": session_id,
            "sessionId": session_id,
            "status": "active",
            "companyId": payload.companyId,
            "moodleSessionId": payload.moodleSessionId,
            "attemptId": payload.attemptId,
        },
    }


@router.post("/{session_id}/snapshot-upload-url", dependencies=[Depends(require_api_auth)])
def snapshot_upload_url(session_id: str, payload: SnapshotUploadUrlRequest) -> dict:
    object_key = f"evidence/staged/{session_id}/snapshots/{payload.violation_id or uuid4()}.jpg"
    return {
        "upload_url": "staged-local-upload-url",
        "object_key": object_key,
        "expires_in_seconds": 60,
        "next": "Replace with a single-purpose MinIO/S3 pre-signed PUT URL.",
    }


@router.post("/{session_id}/finish", dependencies=[Depends(require_api_auth)])
def finish_session(session_id: str, payload: FinishSessionRequest) -> dict:
    return {
        "ok": True,
        "session_id": session_id,
        "state": "finalizing",
        "reason": payload.reason,
        "next": "Queue report generation and signed Moodle webhook delivery.",
    }
