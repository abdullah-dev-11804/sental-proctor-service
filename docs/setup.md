# Setup Commands

Run these commands from:

```bash
cd /home/abdullahamin/Projects/Moodle/proctoring/sental-proctor-service
```

## 1. Install System Dependencies

Use Python 3.11 or 3.12. Avoid Python 3.14 for now because OpenCV/NumPy wheels may not be available and pip may try to compile large native packages from source.

Automatic helper:

```bash
bash apps/api/scripts/install_system_deps.sh
```

Ubuntu/Debian manual install:

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3-pip ffmpeg libglib2.0-0 libgl1
```

Fedora manual install:

```bash
sudo dnf install -y python3.12 python3.12-devel python3-pip ffmpeg glib2 mesa-libGL
```

## 2. Create Environment

```bash
cp .env.example .env
apps/api/scripts/create_venv.sh
source .venv/bin/activate
pip install --upgrade pip
pip install -r apps/api/requirements.txt
```

## 3. Run API Locally

```bash
uvicorn app.main:app --app-dir apps/api --reload --host 127.0.0.1 --port 8091
```

## 4. Test Health

```bash
curl http://127.0.0.1:8091/health
```

## 5. Test Identity Verification

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

Expected decisions:

- `passed` / `allow` when the live face matches the reference confidently.
- `needs_review` / `review` for borderline scores.
- `failed` / `deny` for clear mismatch or multiple faces.
- `needs_retry` / `retry` for no face, bad lighting, blur, or invalid image.

## 6. Docker Staging

Docker is the easiest path when the host Python version is too new:

```bash
cp .env.example .env
docker compose up --build
```

API:

```text
http://127.0.0.1:8091
```

MinIO console:

```text
http://127.0.0.1:9001
```

## 7. Run Tests

```bash
source .venv/bin/activate
pytest apps/api/tests
```
