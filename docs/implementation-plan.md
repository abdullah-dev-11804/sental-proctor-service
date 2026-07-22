# Implementation Plan

## Phase 1 - Identity Access Slice

Status: started.

- FastAPI service skeleton.
- Authenticated `/v1/identity/verify`.
- Local evidence image persistence.
- OpenCV baseline face detection, quality checks, and face similarity scoring.
- Access decision returned as `allow`, `deny`, `review`, or `retry`.

Next hardening:

- Add YuNet/SFace ONNX model support.
- Store identity result in Redis/database.
- Connect Moodle launch/session token to identity verification.
- Add signed reference-image fetch from Moodle or controlled storage.

## Phase 2 - Session And Storage Foundations

Status: staged.

- Replace staged session response with Redis-backed session state.
- Add MinIO bucket bootstrap.
- Generate server-side object keys.
- Generate short-lived single-purpose upload URLs.

## Phase 3 - LiveKit And Rolling Buffer

Status: staged.

- Create LiveKit room per session.
- Generate student LiveKit token.
- Start LiveKit egress HLS segments.
- Track segment timestamps in Redis.
- Cleanup temporary HLS segments.

## Phase 4 - Violations And Clip Worker

Status: staged.

- Add authenticated WebSocket hub.
- Validate violation event tokens.
- Persist violation spans.
- Queue delayed clip creation job.
- Merge HLS segments with ffmpeg.

## Phase 5 - Reports And Moodle Webhooks

Status: staged.

- Generate report JSON.
- Generate report PDF.
- Store report in private reports bucket.
- Send signed final webhook to Moodle.
- Retry with same event id.
