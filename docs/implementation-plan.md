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

Status: first working slice implemented.

- File-backed session state under `storage/media/sessions`.
- Moodle-compatible session create/start/get/heartbeat/interruption/fail endpoints.
- Signed Moodle webhooks for asset and final session events.
- Protected Server B asset content endpoint for Moodle-side report views.

Next hardening:

- Replace file-backed session state with Redis/database state.
- Add MinIO bucket bootstrap.
- Store evidence objects in MinIO/S3 instead of local disk.
- Add durable webhook retry queue.

## Phase 3 - LiveKit And Rolling Buffer

Status: browser rolling-chunk prototype implemented.

- Generate student LiveKit token.
- Browser uploads short WebM chunks to Server B with a scoped upload token.
- Old temporary chunks are pruned by rolling retention.
- Finalization deletes temporary chunks after key clips are materialized.

Next hardening:

- Create LiveKit room lifecycle explicitly.
- Replace browser chunk upload with LiveKit egress HLS/WebM segments.
- Track segment timestamps in Redis/database.
- Normalize generated clips with ffmpeg.

## Phase 4 - Violations And Clip Worker

Status: first working slice implemented.

- Moodle browser-event violations trigger snapshot requests.
- Server B snapshots can queue key-moment clips from the rolling buffer.
- External `/v1/violations` workers can queue key-moment clips.
- Clips are generated as WebM chunk windows around the violation/submission.

Next hardening:

- Add authenticated WebSocket/data-channel hub.
- Persist violation spans in database.
- Add delayed worker instead of materializing clips opportunistically.
- Merge/repair clips with ffmpeg for stronger playback compatibility.

## Phase 5 - Reports And Moodle Webhooks

Status: Moodle report integration implemented.

- Server B sends `asset.captured` webhooks for snapshots and clips.
- Server B sends `session.completed` / `session.failed` final webhooks.
- Moodle stores assets, retention dates, and generates PDF reports.

Next hardening:

- Add optional Server B report JSON/PDF generation.
- Store report artifacts in private reports bucket.
- Retry webhook delivery with durable backoff.
