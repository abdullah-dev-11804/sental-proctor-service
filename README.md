# SENTAL Proctor Service

Server B / AI-media service for the SENTAL Moodle proctoring system.

This service is intentionally separate from Moodle:

- Moodle stores the official exam/proctoring record.
- This service owns identity verification, media/evidence storage, rolling buffers, violation clips, report generation, and final Moodle webhooks.

## Current Slice

Implemented now:

- FastAPI service skeleton.
- Authenticated health and staging APIs.
- Face-recognition access endpoint using OpenCV Zoo YuNet + SFace ONNX models.
- Local evidence/reference storage for development.
- Staging routes for sessions, snapshot upload URLs, violations, finish-session, clips, reports, and webhooks.
- Docker Compose scaffold for Redis, MinIO, LiveKit, and the API.

Not implemented yet:

- LiveKit room creation and egress.
- Real S3/MinIO signed upload URLs.
- WebSocket violation hub.
- HLS rolling buffer and clip worker.
- PDF report worker.
- Signed final webhook delivery to Moodle.

## Quick Start

Use Python 3.11 or 3.12 for local development. Do not use Python 3.14 for this service yet, because OpenCV/NumPy wheels may not be available and pip may try to compile them from source.

```bash
cd sental-proctor-service
cp .env.example .env
apps/api/scripts/create_venv.sh
source .venv/bin/activate
pip install --upgrade pip
pip install -r apps/api/requirements.txt
apps/api/scripts/download_face_models.sh
uvicorn app.main:app --app-dir apps/api --reload --host 127.0.0.1 --port 8091
```

Health check:

```bash
curl http://127.0.0.1:8091/health
```

Identity check with two images:

```bash
curl -X POST http://127.0.0.1:8091/v1/identity/verify \
  -H "Authorization: Bearer dev-secret-change-me" \
  -F "company_id=7" \
  -F "user_id=123" \
  -F "session_id=session-demo-1" \
  -F "live_image=@/path/to/live.jpg" \
  -F "reference_image=@/path/to/profile.jpg"
```

## Docker Staging

```bash
cd sental-proctor-service
cp .env.example .env
apps/api/scripts/download_face_models.sh
docker compose up --build
```

The Docker setup is a staging scaffold. For the first identity-verification slice, the API can run by itself without Redis, MinIO, or LiveKit.

Health output includes `identity_engine` and `identity_models_ready`. For staging/production,
`identity_models_ready` must be `true`; do not enable final identity decisions while the
service is using the development-only legacy matcher fallback.

## Identity Calibration

Before using `block` or `fail` identity mismatch modes, collect consented webcam-like
same-person and different-person pairs from the client environment and run:

```bash
apps/api/scripts/calibrate_identity_thresholds.py /path/to/pairs.csv
```

CSV format:

```csv
label,image_a,image_b
1,/samples/user1_a.jpg,/samples/user1_b.jpg
0,/samples/user1_a.jpg,/samples/user2_a.jpg
```

Use the output false-match and false-non-match rates to set the final pass/review thresholds.

Kazakh server deployment notes live in `docs/deploy-kz-server.md`.

## Install Location

Keep this outside Moodle plugin folders:

```text
proctoring/
  local/proctorcore/              # Moodle local plugin
  quizaccess_proctorcore/         # Moodle quiz access rule workspace
  sental-proctor-service/         # Server B / AI-media service
```
