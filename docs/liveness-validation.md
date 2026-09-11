# Pre-quiz liveness validation

The pre-quiz identity gate uses independent liveness and identity decisions:

1. Moodle requests a one-use challenge from the Proctoring Server.
2. SCRFD and the existing quality policy validate every captured frame.
3. MiniFASNet-V2 scores multiple frames; median live/print/replay scores produce `pass`, `fail`, or `inconclusive`.
4. SixDRepNet validates a randomized temporal head-pose sequence.
5. The illumination validator compares facial chromaticity changes with the randomized server-issued colour sequence.
6. AdaFace runs only after all configured liveness components pass.

Challenges are tenant/user/quiz/transaction bound, stored in Redis, short-lived, and consumed once. Raw images and embeddings are not written by the liveness validator. Numeric component diagnostics are returned to trusted Moodle code and logged in summarized form.

## API contract

- `POST /api/v1/identity/liveness/challenges` issues the challenge.
- `POST /api/v1/identity/liveness/challenges/quality` checks one starting frame without consuming the challenge.
- `POST /api/v1/identity/references/enroll` and `POST /api/v1/identity/references/verify` accept `contextId`, `challengeId`, `challengeNonce`, and timestamped `livenessEvidence`.

The issue and quality endpoints require the existing API bearer token and matching `X-ProctorCore-Company` header. The final enrollment/verification request consumes the challenge before validation, so a challenge cannot be accepted twice.

## Configuration

Use `.env.kazakhstan.example` as the complete production-oriented example. New installations must explicitly enable the temporal and active checks. Existing configurations remain disabled by default.

The following values require calibration from client-environment validation data:

- `IDENTITY_ANTISPOOF_THRESHOLD`
- `IDENTITY_ANTISPOOF_SPOOF_THRESHOLD`
- `IDENTITY_PASSIVE_MIN_VALID_FRAMES`
- `IDENTITY_LIVENESS_MAX_INVALID_FRAME_RATIO`
- `IDENTITY_HEADPOSE_TURN_DEGREES`
- `IDENTITY_HEADPOSE_CENTER_DEGREES`
- `IDENTITY_HEADPOSE_MIN_PROGRESS_DEGREES`
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
