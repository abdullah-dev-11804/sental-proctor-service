# SENTAL Proctor Service

Kazakhstan-hosted AI and media service for the SENTAL Moodle proctoring system.

This service is intentionally separate from Moodle:

- Moodle stores the official exam/proctoring record.
- This service owns identity verification, media/evidence storage, rolling buffers, violation clips, report generation, and final Moodle webhooks.

## Implemented Runtime

- SCRFD detection, AdaFace 1:1 verification, encrypted multi-frame templates,
  MiniFASNet passive anti-spoofing, and SixDRepNet head pose.
- Authenticated, company-scoped Moodle integration endpoints.
- LiveKit participant Egress to private MinIO HLS segments.
- Low-bitrate browser MediaRecorder safety chunks alongside LiveKit Egress.
- Redis/RQ media finalization and signed webhook jobs with retries and dead-letter visibility.
- Temporary FFmpeg full-session assembly and timestamped violation clips.
- Indexed, company-scoped clips, snapshots, identity evidence, and retention state; complete source media is deleted after Moodle reconciliation.
- Session interruption/resume, partial-media preservation, evidence hold/release, and reconciliation.
- Detailed health and per-session diagnostics for deployment validation.
- Moodle-side PDF reports; the Proctoring Server supplies indexed evidence and final status.
- Optional real-time LiveKit microphone analysis with RMS/dBFS noise detection, Silero VAD,
  SpeechBrain ECAPA-TDNN temporal speaker clustering, and cautious suspicious-speech rules.

## Quick Start

Use Python 3.11 or 3.12 for local development. Do not use Python 3.14 for this service yet, because OpenCV/NumPy wheels may not be available and pip may try to compile them from source.

```bash
cd sental-proctor-service
cp .env.example .env
apps/api/scripts/create_venv.sh
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchaudio==2.8.0
pip install -r apps/api/requirements.txt
# Place the approved face models in models/, then install the pinned audio models if enabled.
python apps/api/scripts/download_audio_models.py --model-root models/audio
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
  -H "X-ProctorCore-Company: 7" \
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
# Place the approved face models in models/ and install audio models when enabling audio analysis.
python apps/api/scripts/download_audio_models.py --model-root models/audio
docker compose up --build
```

The local configuration can use file storage and browser recording fallback. The Kazakhstan
production configuration requires Redis, MinIO, LiveKit Egress, the worker, identity models,
biometric encryption secret, and the signed Moodle webhook target to pass `/api/health`.

Health output includes `identity_engine` and `identity_models_ready`. For staging/production,
`identity_models_ready` must be `true`. The service fails closed if the configured engine is
not SCRFD + AdaFace; YuNet, SFace and Haar fallbacks are not supported.

For the stronger face stack, set `IDENTITY_ENGINE=scrfd_adaface` and place the
required SCRFD/AdaFace ONNX files under `models/`. Details are in
`docs/identity-production-stack.md`.

Audio model licensing, settings, health checks, calibration and manual testing are documented in
`docs/audio-production-pipeline.md`. Audio analysis remains disabled until both the Server B
environment and the Moodle administrator setting enable it.

The randomized temporal liveness flow, configuration, manual attack matrix, and
calibration procedure are documented in `docs/liveness-validation.md`.

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
