# Identity Production Stack

The stronger face stack is enabled with:

```text
IDENTITY_ENGINE=scrfd_adaface
```

It uses:

- SCRFD KPS for face detection and 5-point landmarks.
- AdaFace R50 ONNX for aligned face embeddings.
- Optional passive anti-spoof ONNX model.
- Optional 6DRepNet-style ONNX model for head-pose yaw.

Audio monitoring is intentionally outside this phase.

## Required model files

Place these files under the service model directory:

```text
models/scrfd_2.5g_kps.onnx
models/adaface_ir50_ms1mv2.onnx
```

Use `IDENTITY_ADAFACE_COLOR_ORDER=bgr` for ONNX exports from the original
AdaFace repository. If you export from a CVLFace/Hugging Face wrapper that was
validated with RGB input, set it to `rgb`.

Optional hardening files:

```text
models/minifasnet_antispoof.onnx
models/6drepnet_300w_lp.onnx
```

When optional files are ready, set:

```text
IDENTITY_REQUIRE_PASSIVE_ANTISPOOF=true
IDENTITY_ANTISPOOF_MODEL=minifasnet_antispoof.onnx
IDENTITY_ANTISPOOF_LIVE_CLASS_INDEX=1
IDENTITY_ANTISPOOF_CROP_SCALE=2.7
IDENTITY_REQUIRE_HEADPOSE_LIVENESS=true
IDENTITY_HEADPOSE_MODEL=6drepnet_300w_lp.onnx
```

Keep optional requirements disabled until the exact exported model input/output
layout is validated with real webcam samples.

If your uploaded filenames differ, use those exact names instead, for example:

```text
IDENTITY_ANTISPOOF_MODEL=minifasnet_v2.onnx
IDENTITY_HEADPOSE_MODEL=SixDRepNet.onnx
```

## Server check

After updating `.env` and placing models:

```bash
cd /opt/sental-proctor-service
docker compose up -d --build
docker compose exec api python /app/scripts/check_identity_stack.py
curl https://proctoring.sental.kz/api/health
```

The health response must show:

```json
"identity_engine": "scrfd_adaface",
"identity_models_ready": true
```

## Calibration rule

Do not use final `block` or `fail` Moodle behaviour until the client provides a
calibration set from the real exam environment:

- same user, same laptop camera
- same user, phone camera
- different users
- printed photo
- phone-screen replay
- low light
- backlight

Run threshold calibration before setting production pass/review thresholds.
