# Pre-quiz liveness validation

The pre-quiz identity gate uses independent liveness and identity decisions:

1. Moodle requests a one-use challenge from the Proctoring Server.
2. SCRFD applies movement-safe quality limits to liveness frames. The stricter enrollment policy is still applied separately before any reusable reference is stored.
3. MiniFASNet-V2 scores a dedicated neutral frontal burst; median live/print/replay scores produce `pass`, `fail`, or `inconclusive`.
4. SixDRepNet validates a separate randomized temporal head-pose sequence.
5. When an administrator explicitly enables it in Moodle, the illumination validator uses a separate coloured-frame sequence to compare facial chromaticity changes with the server-issued colours.
6. AdaFace runs only after all configured liveness components pass.

The deployed MiniFASNet-V2 file is a direct ONNX export of upstream
`minivision-ai/Silent-Face-Anti-Spoofing` weights. The upstream inference code
keeps resized BGR pixels as float values in the `0..255` range and treats class
index `1` as the genuine/live class. Production must therefore use
`IDENTITY_ANTISPOOF_INPUT_RANGE=raw_255` and
`IDENTITY_ANTISPOOF_LIVE_CLASS_INDEX=1` for this approved file. Its approved
SHA-256 is `d7b3cd9ba8a7ceb13baa8c4720902e27ca3112eff52f926c08804af6b6eecc7b`;
set this as `IDENTITY_ANTISPOOF_MODEL_SHA256` so startup rejects a different
binary. Validate all three conventions before using a differently wrapped or
exported model.

Challenges are tenant/user/quiz/transaction bound, stored in Redis, short-lived, and consumed once. Raw images and embeddings are not written by the liveness validator. Numeric component diagnostics are returned to trusted Moodle code and logged in summarized form.

Challenge evidence uses the movement-safe liveness quality limits. The stricter enrollment limits are applied only to the final straight reference frames before a reusable template is stored.

## API contract

- `POST /api/v1/identity/liveness/challenges` issues the challenge.
- `POST /api/v1/identity/liveness/challenges/quality` checks one starting frame without consuming the challenge.
- `POST /api/v1/identity/liveness/challenges/pose` analyses a chronological burst of up to eight movement frames without consuming the challenge.
- `POST /api/v1/identity/references/enroll` and `POST /api/v1/identity/references/verify` accept `contextId`, `challengeId`, `challengeNonce`, and timestamped `livenessEvidence`.

The issue and quality endpoints require the existing API bearer token and matching `X-ProctorCore-Company` header. The final enrollment/verification request consumes the challenge before validation, so a challenge cannot be accepted twice. Every evidence frame is tagged as `passive`, `headpose`, or `illumination`; a signal is evaluated only from its matching stream.

## Configuration

Use `.env.kazakhstan.example` as the complete production-oriented example. New installations must explicitly enable the temporal and active checks. Existing configurations remain disabled by default. `IDENTITY_ILLUMINATION_CHALLENGE_ENABLED` only makes the server capability available; the screen-light test is requested only when an administrator enables **Require screen-light liveness challenge** in Moodle. That Moodle setting is off by default.

The following values require calibration from client-environment validation data:

- `IDENTITY_ANTISPOOF_THRESHOLD`
- `IDENTITY_ANTISPOOF_SPOOF_THRESHOLD`
- `IDENTITY_PASSIVE_MIN_VALID_FRAMES`
- `IDENTITY_LIVENESS_MAX_INVALID_FRAME_RATIO`
- `IDENTITY_HEADPOSE_TURN_DEGREES`
- `IDENTITY_HEADPOSE_CENTER_DEGREES`
- `IDENTITY_HEADPOSE_MIN_PROGRESS_DEGREES`
- `IDENTITY_HEADPOSE_MIN_FACE_CONFIDENCE`
- `IDENTITY_HEADPOSE_LEFT_SIGN`
- `IDENTITY_ILLUMINATION_MIN_RESPONSE`
- `IDENTITY_ILLUMINATION_PASS_CORRELATION`
- `IDENTITY_ILLUMINATION_FAIL_CORRELATION`

The checked-in values are conservative engineering starting points, not scientifically calibrated production claims.

## Manual validation matrix

Use consented test accounts and record the server-side component result, reason, valid/invalid frame counts, capture duration, device, browser, lighting, and attack type. Do not retain unnecessary raw biometric captures.

Run at least 20 repeated attempts for each genuine condition and each attack condition:

| Scenario | Expected behavior |
| --- | --- |
| Genuine Pixel/Android front camera, normal lighting | Liveness passes consistently, then identity runs |
| Genuine laptop webcam, normal lighting | Liveness passes consistently without device-specific false spoof failures |
| Genuine user with glasses | Liveness passes or returns an explainable retry; identity is evaluated only after pass |
| Low lighting | Quality retry or liveness inconclusive, never a confident spoof solely from poor exposure |
| Strong room lighting and reflections | Genuine sequence remains stable; record PAD and illumination distributions |
| Webcam auto-exposure during colour changes | Illumination remains tolerant or becomes inconclusive, not a false attack |
| Low-quality webcam | Actionable quality retry or inconclusive result |
| Printed photograph | Repeated passive PAD and/or active signals fail |
| Photograph displayed on a phone | Repeated passive PAD and/or active signals fail |
| Prerecorded face video on a phone | Random movement/illumination mismatch fails or becomes inconclusive |
| Correct face, wrong requested movement | Head-pose component fails |
| Challenge replay or expired ID | Request is rejected before identity runs |
| Face disappears during challenge | Inconclusive retry |
| A second face appears | Liveness fails |

## Calibration procedure

1. Keep thresholds unchanged while collecting labelled genuine, print, screen-photo, and video-replay component results.
2. Split data by person so the same identity is not present in calibration and validation sets.
3. Calculate false rejection and false acceptance rates separately for each device and condition.
4. Select thresholds against the client's documented risk target, then validate on the untouched set.
5. Record the dataset version, model checksums, configuration, date, and approver.
6. Re-run calibration whenever a model, preprocessing step, capture timing, or supported device population changes.

Do not tune a threshold from one person's successful or failed attempt.

## Head-pose diagnostics

During a consented test attempt, tail the API logs:

```bash
docker compose logs -f api | grep --line-buffered -E \
  'liveness_challenge_issued|liveness_quality|headpose_'
```

`liveness_challenge_issued` records the actual randomized movement sequence. Every pose frame then produces either `headpose_progress` with yaw, baseline, directed delta, required delta and hold count, or `headpose_frame_rejected` with the exact quality reason and numeric quality measurements.

For a requested turn, `directed_delta` must increase toward `required_delta`. The validated SixDRepNet export and mirrored SENTAL preview use `IDENTITY_HEADPOSE_LEFT_SIGN=-1`: a user's left turn produces a negative raw yaw delta and a positive directed delta. Revalidate this setting if the model export or capture mirroring changes. Do not lower the turn threshold to compensate for an inverted sign.

`IDENTITY_HEADPOSE_MIN_FACE_CONFIDENCE` applies only while the user is turning. It allows SCRFD to retain a partially rotated face without weakening the stricter straight-pose and reusable-reference quality gates.
