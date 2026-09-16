# Production audio-violation pipeline

## Scope and event meaning

The dedicated `audio-analyzer` service subscribes to the candidate's existing private LiveKit
microphone track. It converts frames to 16 kHz mono PCM and applies, in order:

1. RMS/dBFS energy measurement for sustained non-speech noise.
2. Silero VAD for speech probability and speech-segment aggregation.
3. SpeechBrain ECAPA-TDNN embeddings only for qualifying speech segments.
4. Rolling speaker clustering and observable speech-pattern rules.
5. Normalized `violation.detected` webhooks to Moodle's existing violation, scoring and report flow.

The four reportable types are `background_noise`, `speech_detected`,
`second_voice_detected`, and `possible_prompting`. The final type means only that sustained,
repeated, or multi-speaker speech activity needs review. There is no transcription, semantic
analysis, speaker identity, or claim that answers were definitely dictated.

Each event carries start/end timestamps, duration, detector confidence, detector-specific
measurements, and a reference to the matching LiveKit recording segment. Finalization uses the
existing FFmpeg clip extractor, so the report can link a short video/audio evidence window without
duplicating a second long audio recording.

## Approved models and licences

The pinned inventory is machine-readable in `docs/audio-models.json`.

- Silero VAD v6.2, MIT licence, official source:
  `https://github.com/snakers4/silero-vad/tree/v6.2`. The pinned ONNX file is approximately 2.3 MB.
- SpeechBrain `spkrec-ecapa-voxceleb`, revision
  `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`, Apache-2.0 licence, official source:
  `https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb`. The embedding checkpoint is
  approximately 83 MB and the model card reports 0.80% EER on the cleaned VoxCeleb1 test set.

Both licences permit commercial use. Model accuracy on an exam-room microphone is not implied by
the published benchmark; thresholds must be calibrated on representative consented recordings.
The service never downloads models at runtime and refuses to load a model whose configured
SHA-256 does not match. Downloaded model artifacts remain read-only at runtime. SpeechBrain's
generated cache files are written under `/app/storage/audio-model-cache`, not into the model mount.
The analyzer overrides the upstream YAML's remote `pretrained_path` with the verified local model
directory and runs with `HF_HUB_OFFLINE=1`, preventing runtime Hub downloads.

## Installation

From the service root:

```bash
docker compose build api
mkdir -p models/audio

docker compose run --rm --no-deps \
  --volume "$(pwd)/models/audio:/audio-models:rw" \
  api python /app/scripts/download_audio_models.py \
  --model-root /audio-models

docker compose run --rm --no-deps \
  --volume "$(pwd)/models/audio:/audio-models:ro" \
  api python /app/scripts/download_audio_models.py \
  --model-root /audio-models \
  --verify-only
```

The one-off download container mounts the host `models/audio` directory at the separate
`/audio-models` path with temporary write access. This avoids conflicting with the service's
inherited read-only `/app/models` mount. The normal API, worker, and analyzer services continue to
read the downloaded files at `/app/models/audio` through their read-only model mount.

Copy the audio variables from `.env.example`, set `AUDIO_ANALYSIS_ENABLED=true`, retain the pinned
checksums, then rebuild and start the API, worker and analyzer:

```bash
docker compose build api worker audio-analyzer
docker compose up -d api worker audio-analyzer
curl -fsS http://127.0.0.1:8091/api/health
```

The health response must show:

```json
{
  "features": {
    "audioAnalysis": {
      "enabled": true,
      "ready": true,
      "pipelineAvailable": true,
      "sampleRate": 16000
    }
  }
}
```

If a required model or analyzer heartbeat is absent, health becomes degraded and every requested
session is marked with a degraded audio-analysis state. The service does not silently claim audio
analysis is running.

In Moodle, go to **Site administration → Plugins → Local plugins → ProctorCore** and open
**Audio violation analysis**. Enable the general switch, then tune the separate noise, speech,
additional-speaker and suspicious-speech controls. The points for all four event types are under
**Violation points and session outcome**. These are configuration-only settings; no Moodle database
upgrade is needed.

## Runtime and capacity

The Docker image installs CPU-only PyTorch 2.8, torchaudio 2.8, SpeechBrain 1.1.1, LiveKit RTC
1.1.18, and uses the existing ONNX Runtime for Silero. Reserve roughly 0.5–1 GB RAM for the analyzer
container after model loading. VAD cost is small; ECAPA inference is the dominant CPU cost and runs
only on speech segments of at least one second. Benchmark concurrent rooms on the production CPU
before choosing capacity. Scale analyzer instances only after ensuring a session is assigned to one
analyzer; duplicate event IDs protect Moodle, but duplicate inference wastes CPU.

## Manual acceptance tests

Use a new proctored attempt for each calibration case and verify the Server B session diagnostics,
Moodle timeline, risk points, and timestamped clip:

1. Remain silent: no audio event.
2. Clap once or tap the desk: no sustained-noise event.
3. Play steady fan/vacuum noise above the configured dBFS threshold: one `background_noise` after
   the minimum duration.
4. Speak one complete sentence: one `speech_detected` after the minimum duration.
5. Speak continuously beyond the sustained threshold: `speech_detected`, then possibly
   `possible_prompting` with review-only wording.
6. Have one person speak from different distances/volumes: no `second_voice_detected`.
7. Alternate several qualifying segments from two clearly different speakers:
   `second_voice_detected` only after the configured repeated evidence count.
8. Continue an alternating conversation: possibly `possible_prompting` once rolling thresholds are
   met.
9. Play TV speech: speech and possibly multiple-speaker evidence may be reported, but the report
   must not claim proven prompting or cheating.
10. Speak continuously: confirm one consolidated speech event rather than one per audio frame.
11. Stop the analyzer or rename a model: `/api/health` must become degraded and the session must
    show degraded audio analysis.
12. Disable the Moodle general audio setting: start another attempt and confirm no audio-analysis
    violations are created.

## Known limitations and calibration risks

- TV, radio, loud laptop audio and reverberant rooms can look like additional speakers. This is why
  the event is reviewable evidence, not an automatic identity claim.
- Very short speech is intentionally filtered and may be missed.
- Similar voices, strong echo, low-bitrate audio, accents, illness and microphone changes can shift
  embedding similarity. Do not lower the repeated-evidence requirement merely to increase recall.
- The first stable voice cluster is only a temporal reference; it is not cryptographically tied to
  the enrolled face and is not a speaker identity model.
- Risk points may automatically fail a session if the administrator configures aggressive totals.
  Use manual-review thresholds while calibrating real exam rooms.
