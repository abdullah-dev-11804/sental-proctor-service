from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.security import require_api_auth
from app.services.media_store import MediaStore


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
    retention: dict | None = None
    source: dict | None = None


class MoodleSessionStartRequest(BaseModel):
    moodleSessionId: int | None = Field(default=None, ge=1)
    companyId: int = Field(default=0, ge=0)
    attemptId: int | None = Field(default=None, ge=1)
    userId: int | None = Field(default=None, ge=1)
    startedAt: str | None = None
    source: str | None = None


class RecordingRequest(BaseModel):
    moodleSessionId: int | None = Field(default=None, ge=1)
    companyId: int = Field(default=0, ge=0)
    attemptId: int | None = Field(default=None, ge=1)
    userId: int | None = Field(default=None, ge=1)
    segment: int = Field(default=1, ge=1)
    reason: str = "attempt_page_connected"
    result: str | None = Field(default=None, pattern="^(passed|failed)$")
    startedAt: str | None = None
    stoppedAt: str | None = None
    idempotencyKey: str | None = None


class SnapshotRequest(BaseModel):
    moodleSessionId: int | None = Field(default=None, ge=1)
    companyId: int = Field(default=0, ge=0)
    attemptId: int | None = Field(default=None, ge=1)
    userId: int | None = Field(default=None, ge=1)
    reason: str = "manual_proctor"
    capturedAt: str | None = None
    violationId: int | str | None = None
    violationType: str | None = None
    occurredAt: int | str | None = None
    idempotencyKey: str | None = None
    snapshotImage: str | None = None


class MediaTokenRequest(BaseModel):
    moodleSessionId: int | None = Field(default=None, ge=1)
    attemptId: int | None = Field(default=None, ge=1)
    quizId: int | None = Field(default=None, ge=1)
    courseId: int | None = Field(default=None, ge=1)
    companyId: int = Field(default=0, ge=0)
    userId: int | None = Field(default=None, ge=1)
    participantIdentity: str | None = None
    participantName: str | None = None
    permissions: dict | None = None
    requestedAt: str | None = None


class InterruptionRequest(BaseModel):
    moodleSessionId: int | None = Field(default=None, ge=1)
    companyId: int = Field(default=0, ge=0)
    attemptId: int | None = Field(default=None, ge=1)
    userId: int | None = Field(default=None, ge=1)
    reason: str = "connection_lost"
    message: str | None = None


class FailureRequest(InterruptionRequest):
    reason: str = "server_failed"


class EvidenceHoldRequest(BaseModel):
    companyId: int = Field(ge=0)
    reason: str = Field(min_length=3, max_length=500)


class ReconcileRequest(BaseModel):
    companyId: int = Field(ge=0)
    resendWebhooks: bool = False


@router.post("/start", dependencies=[Depends(require_api_auth)])
def start_session(payload: SessionStartRequest) -> dict:
    session = MediaStore().create_session(payload.model_dump())
    return {
        "status": "pending_preflight",
        "session_id": session["id"],
        "company_id": payload.company_id,
        "moodle_session_id": payload.moodle_session_id,
        "quiz_attempt_id": payload.quiz_attempt_id,
    }


@compat_router.post("", dependencies=[Depends(require_api_auth)])
def create_moodle_session(payload: MoodleSessionCreateRequest) -> dict:
    """Moodle local_proctorcore compatibility route."""
    store = MediaStore()
    session = store.create_session(payload.model_dump())
    session = {
        "id": session["id"],
        "sessionId": session["sessionId"],
        "roomId": session["roomId"],
        "status": session["status"],
        "companyId": session["companyId"],
        "moodleSessionId": session["moodleSessionId"],
        "attemptId": session["attemptId"],
        "quizId": session["quizId"],
        "courseId": session["courseId"],
        "media": {
            "livekitUrl": store.settings.livekit_url,
            "roomId": session["roomId"],
        },
    }
    return {
        "ok": True,
        "status": "created",
        "session": session,
    }


@compat_router.get("/{session_id}", dependencies=[Depends(require_api_auth)])
def get_moodle_session(session_id: str, x_proctorcore_company: int | None = Header(None)) -> dict:
    try:
        session = MediaStore().get_session(session_id)
        if x_proctorcore_company is None or int(session.get("companyId") or 0) != x_proctorcore_company:
            raise KeyError("session_not_found")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc
    return {
        "ok": True,
        "session": {
            "id": session["id"],
            "sessionId": session["sessionId"],
            "status": session["status"],
            "recording": session.get("recording", {}),
            "assets": session.get("assets", []),
            "violations": session.get("violations", []),
            "processing": session.get("processing", {}),
            "retention": session.get("retention", {}),
            "webhookDeliveries": session.get("webhookDeliveries", []),
        },
    }


@compat_router.post("/{session_id}/start", dependencies=[Depends(require_api_auth)])
def start_moodle_session(session_id: str, payload: MoodleSessionStartRequest) -> dict:
    try:
        session = MediaStore().start_session(session_id, payload.model_dump())
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc
    return {
        "ok": True,
        "session": {
            "id": session["id"],
            "sessionId": session["sessionId"],
            "status": "active",
            "companyId": session["companyId"],
            "moodleSessionId": session["moodleSessionId"],
            "attemptId": session["attemptId"],
        },
    }


@compat_router.post("/{session_id}/heartbeat", dependencies=[Depends(require_api_auth)])
def heartbeat_moodle_session(session_id: str, payload: MoodleSessionStartRequest) -> dict:
    try:
        session = MediaStore().heartbeat(session_id, payload.model_dump())
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc
    return {"ok": True, "status": session["status"], "lastHeartbeatAt": session["lastHeartbeatAt"]}


@compat_router.post("/{session_id}/media-token", dependencies=[Depends(require_api_auth)])
def create_moodle_media_token(session_id: str, payload: MediaTokenRequest) -> dict:
    try:
        return MediaStore().create_media_token(session_id, payload.model_dump())
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc


@compat_router.post("/{session_id}/recording/start", dependencies=[Depends(require_api_auth)])
def start_moodle_recording(session_id: str, payload: RecordingRequest) -> dict:
    try:
        return MediaStore().start_recording(session_id, payload.model_dump())
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc


@compat_router.post("/{session_id}/recording/stop", dependencies=[Depends(require_api_auth)])
def stop_moodle_recording(session_id: str, payload: RecordingRequest) -> dict:
    try:
        return MediaStore().stop_recording(session_id, payload.model_dump())
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc


@compat_router.post("/{session_id}/snapshots", dependencies=[Depends(require_api_auth)])
def capture_moodle_snapshot(session_id: str, payload: SnapshotRequest) -> dict:
    try:
        return MediaStore().capture_snapshot(session_id, payload.model_dump())
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc


@compat_router.post("/{session_id}/interrupt", dependencies=[Depends(require_api_auth)])
def interrupt_moodle_session(session_id: str, payload: InterruptionRequest) -> dict:
    try:
        return MediaStore().interrupt_session(session_id, payload.model_dump())
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc


@compat_router.post("/{session_id}/resume", dependencies=[Depends(require_api_auth)])
def resume_moodle_session(session_id: str, payload: InterruptionRequest) -> dict:
    try:
        return MediaStore().resume_session(session_id, payload.model_dump())
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc


@compat_router.post("/{session_id}/fail", dependencies=[Depends(require_api_auth)])
def fail_moodle_session(session_id: str, payload: FailureRequest) -> dict:
    try:
        return MediaStore().fail_session(session_id, payload.model_dump())
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc


@compat_router.post("/{session_id}/media/chunks")
async def upload_moodle_media_chunk(
    session_id: str,
    asset: UploadFile = File(...),
    segment: int = Form(1),
    sequence: int = Form(0),
    durationMs: int | None = Form(None),
    x_proctorcore_upload_token: str = Header(""),
) -> dict:
    try:
        content = await asset.read()
        return MediaStore().save_media_chunk(
            session_id,
            x_proctorcore_upload_token,
            content,
            segment=segment,
            sequence=sequence,
            duration_ms=durationMs,
            mime_type=asset.content_type or "video/webm",
        )
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail="invalid_upload_token") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{session_id}/snapshot-upload-url", dependencies=[Depends(require_api_auth)])
def snapshot_upload_url(session_id: str, payload: SnapshotUploadUrlRequest) -> dict:
    object_key = f"evidence/{session_id}/snapshots/{payload.violation_id or 'manual'}.jpg"
    return {
        "upload_url": "staged-local-upload-url",
        "object_key": object_key,
        "expires_in_seconds": 60,
        "next": "Replace with a single-purpose MinIO/S3 pre-signed PUT URL.",
    }


@compat_router.post("/{session_id}/evidence/hold", dependencies=[Depends(require_api_auth)])
def hold_session_evidence(session_id: str, payload: EvidenceHoldRequest) -> dict:
    store = MediaStore()
    try:
        session = store.get_session(session_id)
        store._require_scope(session, payload.model_dump())
        retention = store.set_evidence_hold(session_id, True, payload.reason)
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc
    return {"ok": True, "retention": retention}


@compat_router.post("/{session_id}/evidence/release", dependencies=[Depends(require_api_auth)])
def release_session_evidence(session_id: str, payload: EvidenceHoldRequest) -> dict:
    store = MediaStore()
    try:
        session = store.get_session(session_id)
        store._require_scope(session, payload.model_dump())
        retention = store.set_evidence_hold(session_id, False, payload.reason)
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc
    return {"ok": True, "retention": retention}


@compat_router.post("/{session_id}/reconcile", dependencies=[Depends(require_api_auth)])
def reconcile_moodle_session(session_id: str, payload: ReconcileRequest) -> dict:
    store = MediaStore()
    try:
        session = store.get_session(session_id)
        store._require_scope(session, payload.model_dump())
        result = store.reconcile_session(session_id, payload.resendWebhooks)
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc
    return {"ok": True, "reconciliation": result}


@router.post("/{session_id}/finish", dependencies=[Depends(require_api_auth)])
def finish_session(session_id: str, payload: FinishSessionRequest) -> dict:
    try:
        MediaStore().finish_session(session_id, payload.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc
    return {
        "ok": True,
        "session_id": session_id,
        "state": "finalized",
        "reason": payload.reason,
    }
