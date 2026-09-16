from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.core.config import Settings, get_settings


@dataclass(frozen=True)
class AudioPolicy:
    enabled: bool
    sample_rate: int
    noise_enabled: bool
    noise_threshold_dbfs: float
    noise_min_seconds: float
    noise_cooldown_seconds: int
    speech_enabled: bool
    vad_threshold: float
    speech_min_seconds: float
    speech_cooldown_seconds: int
    second_speaker_enabled: bool
    speaker_similarity_threshold: float
    speaker_min_segments: int
    speaker_window_seconds: int
    second_speaker_cooldown_seconds: int
    prompting_enabled: bool
    prompt_sustained_seconds: float
    prompt_rolling_seconds: float
    prompt_window_seconds: int
    prompt_multispeaker: bool
    prompt_cooldown_seconds: int

    @classmethod
    def from_session(cls, session: dict[str, Any], settings: Settings | None = None) -> "AudioPolicy":
        settings = settings or get_settings()
        supplied = session.get("audioAnalysis")
        if not isinstance(supplied, dict):
            supplied = {}
        noise = supplied.get("backgroundNoise") if isinstance(supplied.get("backgroundNoise"), dict) else {}
        speech = supplied.get("speech") if isinstance(supplied.get("speech"), dict) else {}
        speaker = supplied.get("secondSpeaker") if isinstance(supplied.get("secondSpeaker"), dict) else {}
        prompt = supplied.get("possiblePrompting") if isinstance(supplied.get("possiblePrompting"), dict) else {}
        return cls(
            enabled=bool(settings.audio_analysis_enabled and supplied.get("enabled", False)),
            sample_rate=16000,
            noise_enabled=bool(noise.get("enabled", settings.audio_background_noise_enabled)),
            noise_threshold_dbfs=_bounded(noise.get("thresholdDbfs"), settings.audio_noise_threshold_dbfs, -90, 0),
            noise_min_seconds=_bounded(
                noise.get("minimumDurationSeconds"), settings.audio_noise_min_duration_seconds, 0.5, 120
            ),
            noise_cooldown_seconds=int(_bounded(
                noise.get("cooldownSeconds"), settings.audio_noise_cooldown_seconds, 5, 3600
            )),
            speech_enabled=bool(speech.get("enabled", settings.audio_speech_enabled)),
            vad_threshold=_bounded(speech.get("vadThreshold"), settings.audio_vad_threshold, 0.05, 0.99),
            speech_min_seconds=_bounded(
                speech.get("minimumDurationSeconds"), settings.audio_speech_min_duration_seconds, 0.2, 30
            ),
            speech_cooldown_seconds=int(_bounded(
                speech.get("cooldownSeconds"), settings.audio_speech_cooldown_seconds, 5, 3600
            )),
            second_speaker_enabled=bool(speaker.get("enabled", settings.audio_second_speaker_enabled)),
            speaker_similarity_threshold=_bounded(
                speaker.get("similarityThreshold"), settings.audio_speaker_similarity_threshold, 0.1, 0.99
            ),
            speaker_min_segments=int(_bounded(
                speaker.get("minimumSegments"), settings.audio_speaker_min_segments, 2, 12
            )),
            speaker_window_seconds=int(_bounded(
                speaker.get("windowSeconds"), settings.audio_speaker_window_seconds, 15, 600
            )),
            second_speaker_cooldown_seconds=int(_bounded(
                speaker.get("cooldownSeconds"), settings.audio_second_speaker_cooldown_seconds, 10, 3600
            )),
            prompting_enabled=bool(prompt.get("enabled", settings.audio_prompting_enabled)),
            prompt_sustained_seconds=_bounded(
                prompt.get("sustainedSpeechSeconds"), settings.audio_prompt_sustained_speech_seconds, 2, 300
            ),
            prompt_rolling_seconds=_bounded(
                prompt.get("rollingSpeechSeconds"), settings.audio_prompt_rolling_speech_seconds, 2, 600
            ),
            prompt_window_seconds=int(_bounded(
                prompt.get("windowSeconds"), settings.audio_prompt_window_seconds, 15, 900
            )),
            prompt_multispeaker=bool(
                prompt.get("multiSpeakerContribution", settings.audio_prompt_multispeaker_enabled)
            ),
            prompt_cooldown_seconds=int(_bounded(
                prompt.get("cooldownSeconds"), settings.audio_prompt_cooldown_seconds, 10, 3600
            )),
        )


@dataclass
class SpeechSegment:
    start: float
    end: float
    confidence: float
    samples: np.ndarray
    cluster: int | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SpeakerCluster:
    centroid: np.ndarray
    observations: int = 1


class SileroVad:
    """Stateful Silero VAD ONNX inference for 16 kHz, 512-sample frames."""

    def __init__(self, settings: Settings | None = None) -> None:
        import onnxruntime as ort

        self.settings = settings or get_settings()
        self.path = self.settings.audio_model_root / self.settings.audio_silero_model
        _require_model(self.path, self.settings.audio_silero_model_sha256, "Silero VAD")
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.sample_rate = np.array(16000, dtype=np.int64)

    def __call__(self, frame: np.ndarray) -> float:
        samples = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        outputs = self.session.run(None, {"input": samples, "state": self.state, "sr": self.sample_rate})
        self.state = np.asarray(outputs[1], dtype=np.float32)
        return float(np.asarray(outputs[0]).reshape(-1)[0])

    def reset(self) -> None:
        self.state.fill(0)


class SpeechBrainSpeakerEncoder:
    """Local-only SpeechBrain ECAPA-TDNN embedding inference."""

    def __init__(self, settings: Settings | None = None) -> None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self.settings = settings or get_settings()
        self.root = self.settings.audio_model_root / self.settings.audio_speaker_model
        checkpoint = self.root / "embedding_model.ckpt"
        _require_model(checkpoint, self.settings.audio_speaker_model_sha256, "SpeechBrain ECAPA-TDNN")
        cache_key = hashlib.sha256(
            f"{self.settings.audio_speaker_model}:{self.settings.audio_speaker_model_version}".encode("utf-8")
        ).hexdigest()[:16]
        self.cache_root = self.settings.local_storage_root / "audio-model-cache" / cache_key
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.torch = torch
        self.classifier = EncoderClassifier.from_hparams(
            source=str(self.root),
            # SpeechBrain materializes runtime files such as label_encoder.ckpt.
            # Models are intentionally mounted read-only, so keep generated files
            # in the service's writable, persistent storage instead.
            savedir=str(self.cache_root),
            # The upstream YAML names its Hugging Face repository as
            # pretrained_path. Force all pretrainer loadables to resolve from
            # our downloaded, checksum-verified local model directory.
            overrides={"pretrained_path": str(self.root)},
            run_opts={"device": "cpu"},
        )

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        signal = self.torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)
        with self.torch.no_grad():
            embedding = self.classifier.encode_batch(signal)
        vector = embedding.detach().cpu().numpy().reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / max(norm, 1e-9)


class AudioEventEngine:
    """Streaming audio state machine with conservative, observable event rules."""

    FRAME_SAMPLES = 512
    SPEECH_HANGOVER_SECONDS = 0.30
    MIN_SPEAKER_SEGMENT_SECONDS = 1.0
    MAX_SPEAKER_AUDIO_SECONDS = 15.0

    def __init__(
        self,
        policy: AudioPolicy,
        vad: Callable[[np.ndarray], float],
        speaker_encoder: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.policy = policy
        self.vad = vad
        self.speaker_encoder = speaker_encoder
        self.pending = np.empty(0, dtype=np.float32)
        self.pending_started_at: float | None = None
        self.vad_history: deque[float] = deque(maxlen=3)
        self.noise_started_at: float | None = None
        self.noise_dbfs: list[float] = []
        self.speech_started_at: float | None = None
        self.speech_last_at: float | None = None
        self.speech_probabilities: list[float] = []
        self.speech_samples: list[np.ndarray] = []
        self.speech_sample_count = 0
        self.active_prompt_emitted = False
        self.segments: deque[SpeechSegment] = deque()
        self.clusters: list[SpeakerCluster] = []
        self.cluster_observations: deque[tuple[float, int]] = deque()
        self.last_emitted: dict[str, float] = {}
        self.second_speaker_active = False
        self.diagnostic_frames = 0
        self.diagnostic_speech_frames = 0
        self.diagnostic_max_vad = 0.0
        self.diagnostic_max_dbfs = -180.0

    def process(self, samples: np.ndarray, started_at: float) -> list[dict[str, Any]]:
        if not self.policy.enabled:
            return []
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        samples = np.clip(samples, -1.0, 1.0)
        if samples.size == 0:
            return []
        if self.pending_started_at is None:
            self.pending_started_at = float(started_at)
        self.pending = np.concatenate((self.pending, samples))
        events: list[dict[str, Any]] = []
        frame_seconds = self.FRAME_SAMPLES / self.policy.sample_rate
        while self.pending.size >= self.FRAME_SAMPLES:
            frame = self.pending[:self.FRAME_SAMPLES]
            self.pending = self.pending[self.FRAME_SAMPLES:]
            frame_started_at = float(self.pending_started_at)
            self.pending_started_at = frame_started_at + frame_seconds
            events.extend(self._process_frame(frame, frame_started_at, frame_seconds))
        return events

    def flush(self, ended_at: float | None = None) -> list[dict[str, Any]]:
        now = float(ended_at if ended_at is not None else time.time())
        events = self._finish_noise(now)
        events.extend(self._finish_speech(now))
        self.pending = np.empty(0, dtype=np.float32)
        self.pending_started_at = None
        reset = getattr(self.vad, "reset", None)
        if callable(reset):
            reset()
        return events

    def _process_frame(self, frame: np.ndarray, started_at: float, duration: float) -> list[dict[str, Any]]:
        probability = float(np.clip(self.vad(frame), 0.0, 1.0))
        self.vad_history.append(probability)
        smoothed = float(np.mean(self.vad_history))
        speech_active = smoothed >= self.policy.vad_threshold
        dbfs = _dbfs(frame)
        self.diagnostic_frames += 1
        self.diagnostic_speech_frames += int(speech_active)
        self.diagnostic_max_vad = max(self.diagnostic_max_vad, probability)
        self.diagnostic_max_dbfs = max(self.diagnostic_max_dbfs, dbfs)
        events: list[dict[str, Any]] = []

        if self.policy.noise_enabled and dbfs >= self.policy.noise_threshold_dbfs and not speech_active:
            self.noise_started_at = self.noise_started_at if self.noise_started_at is not None else started_at
            self.noise_dbfs.append(dbfs)
        else:
            events.extend(self._finish_noise(started_at))

        if speech_active:
            if self.speech_started_at is None:
                self.speech_started_at = started_at
                self.active_prompt_emitted = False
            self.speech_last_at = started_at + duration
            self.speech_probabilities.append(smoothed)
            self._append_speaker_audio(frame)
            active_duration = (started_at + duration) - self.speech_started_at
            if self.policy.prompting_enabled \
                    and not self.active_prompt_emitted \
                    and active_duration >= self.policy.prompt_sustained_seconds \
                    and self._cooldown_ready(
                        "possible_prompting", started_at + duration, self.policy.prompt_cooldown_seconds
                    ):
                self.active_prompt_emitted = True
                events.append(self._event(
                    "possible_prompting",
                    self.speech_started_at,
                    started_at + duration,
                    confidence=min(0.95, max(0.5, active_duration / self.policy.prompt_sustained_seconds * 0.6)),
                    metadata={
                        "detector": "observable_speech_pattern_rules",
                        "sustainedSpeech": True,
                        "repeatedSpeech": False,
                        "multiSpeakerContribution": False,
                        "latestSpeechDurationSeconds": round(active_duration, 3),
                        "rollingSpeechDurationSeconds": round(active_duration, 3),
                        "analysisWindowSeconds": self.policy.prompt_window_seconds,
                        "reviewSignalOnly": True,
                        "continuousEvent": True,
                    },
                ))
        elif self.speech_started_at is not None:
            self._append_speaker_audio(frame)
            if started_at - float(self.speech_last_at or started_at) >= self.SPEECH_HANGOVER_SECONDS:
                events.extend(self._finish_speech(float(self.speech_last_at or started_at)))
        return events

    def diagnostics(self, reset: bool = False) -> dict[str, float | int]:
        """Returns non-content audio levels for operational pipeline diagnostics."""
        result = {
            "frames": self.diagnostic_frames,
            "speechFrames": self.diagnostic_speech_frames,
            "maxVadProbability": round(self.diagnostic_max_vad, 4),
            "maxDbfs": round(self.diagnostic_max_dbfs, 2),
        }
        if reset:
            self.diagnostic_frames = 0
            self.diagnostic_speech_frames = 0
            self.diagnostic_max_vad = 0.0
            self.diagnostic_max_dbfs = -180.0
        return result

    def _finish_noise(self, ended_at: float) -> list[dict[str, Any]]:
        if self.noise_started_at is None:
            return []
        started_at = self.noise_started_at
        values = self.noise_dbfs
        self.noise_started_at = None
        self.noise_dbfs = []
        duration = max(0.0, ended_at - started_at)
        if duration < self.policy.noise_min_seconds or not self._cooldown_ready(
            "background_noise", ended_at, self.policy.noise_cooldown_seconds
        ):
            return []
        mean_dbfs = float(np.mean(values)) if values else self.policy.noise_threshold_dbfs
        return [self._event(
            "background_noise", started_at, ended_at,
            confidence=_energy_confidence(mean_dbfs, self.policy.noise_threshold_dbfs),
            metadata={
                "meanDbfs": round(mean_dbfs, 2),
                "peakDbfs": round(max(values) if values else mean_dbfs, 2),
                "thresholdDbfs": self.policy.noise_threshold_dbfs,
                "detector": "rms_dbfs",
            },
        )]

    def _finish_speech(self, ended_at: float) -> list[dict[str, Any]]:
        if self.speech_started_at is None:
            return []
        segment = SpeechSegment(
            start=self.speech_started_at,
            end=max(ended_at, self.speech_started_at),
            confidence=float(np.mean(self.speech_probabilities)) if self.speech_probabilities else 0.0,
            samples=np.concatenate(self.speech_samples) if self.speech_samples else np.empty(0, dtype=np.float32),
        )
        self.speech_started_at = None
        self.speech_last_at = None
        self.speech_probabilities = []
        self.speech_samples = []
        self.speech_sample_count = 0
        active_prompt_emitted = self.active_prompt_emitted
        self.active_prompt_emitted = False
        if segment.duration < self.policy.speech_min_seconds:
            return []

        events: list[dict[str, Any]] = []
        self.segments.append(segment)
        self._prune(segment.end)
        if self.policy.speech_enabled and self._cooldown_ready(
            "speech_detected", segment.end, self.policy.speech_cooldown_seconds
        ):
            events.append(self._event(
                "speech_detected", segment.start, segment.end, segment.confidence,
                {"detector": "silero_vad", "vadThreshold": self.policy.vad_threshold},
            ))

        if (self.policy.second_speaker_enabled or self.policy.prompt_multispeaker) \
                and self.speaker_encoder is not None \
                and segment.duration >= self.MIN_SPEAKER_SEGMENT_SECONDS:
            embedding = np.asarray(self.speaker_encoder(segment.samples), dtype=np.float32).reshape(-1)
            segment.cluster, similarity = self._assign_cluster(embedding, segment.end)
            counts = self._cluster_counts()
            stable_clusters = [cluster for cluster, count in counts.items() if count >= self.policy.speaker_min_segments]
            self.second_speaker_active = len(stable_clusters) > 1
            if self.policy.second_speaker_enabled and self.second_speaker_active and self._cooldown_ready(
                "second_voice_detected", segment.end, self.policy.second_speaker_cooldown_seconds
            ):
                events.append(self._event(
                    "second_voice_detected", self._window_start(segment.end), segment.end,
                    confidence=max(0.0, min(1.0, 1.0 - similarity)),
                    metadata={
                        "detector": "speechbrain_ecapa_tdnn_temporal_clustering",
                        "stableSpeakerClusters": len(stable_clusters),
                        "clusterCounts": {str(key): value for key, value in counts.items()},
                        "sameSpeakerSimilarityThreshold": self.policy.speaker_similarity_threshold,
                        "minimumSegments": self.policy.speaker_min_segments,
                        "analysisWindowSeconds": self.policy.speaker_window_seconds,
                    },
                ))

        if self.policy.prompting_enabled:
            rolling_duration = sum(item.duration for item in self.segments)
            sustained = segment.duration >= self.policy.prompt_sustained_seconds
            repeated = rolling_duration >= self.policy.prompt_rolling_seconds
            multispeaker = bool(self.policy.prompt_multispeaker and self.second_speaker_active and repeated)
            if not active_prompt_emitted and (sustained or repeated or multispeaker) and self._cooldown_ready(
                "possible_prompting", segment.end, self.policy.prompt_cooldown_seconds
            ):
                events.append(self._event(
                    "possible_prompting", self._window_start(segment.end), segment.end,
                    confidence=_prompt_confidence(
                        segment.duration,
                        rolling_duration,
                        self.policy.prompt_sustained_seconds,
                        self.policy.prompt_rolling_seconds,
                        multispeaker,
                    ),
                    metadata={
                        "detector": "observable_speech_pattern_rules",
                        "sustainedSpeech": sustained,
                        "repeatedSpeech": repeated,
                        "multiSpeakerContribution": multispeaker,
                        "latestSpeechDurationSeconds": round(segment.duration, 3),
                        "rollingSpeechDurationSeconds": round(rolling_duration, 3),
                        "analysisWindowSeconds": self.policy.prompt_window_seconds,
                        "reviewSignalOnly": True,
                    },
                ))
        return events

    def _append_speaker_audio(self, frame: np.ndarray) -> None:
        maximum = int(self.MAX_SPEAKER_AUDIO_SECONDS * self.policy.sample_rate)
        remaining = maximum - self.speech_sample_count
        if remaining <= 0:
            return
        selected = frame[:remaining].copy()
        self.speech_samples.append(selected)
        self.speech_sample_count += selected.size

    def _assign_cluster(self, embedding: np.ndarray, observed_at: float) -> tuple[int, float]:
        norm = float(np.linalg.norm(embedding))
        embedding = embedding / max(norm, 1e-9)
        similarities = [float(np.dot(embedding, item.centroid)) for item in self.clusters]
        best = int(np.argmax(similarities)) if similarities else -1
        best_similarity = similarities[best] if best >= 0 else 0.0
        if best < 0 or best_similarity < self.policy.speaker_similarity_threshold:
            best = len(self.clusters)
            self.clusters.append(SpeakerCluster(embedding.copy()))
        else:
            cluster = self.clusters[best]
            centroid = ((cluster.centroid * cluster.observations) + embedding) / (cluster.observations + 1)
            cluster.centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-9)
            cluster.observations += 1
        self.cluster_observations.append((observed_at, best))
        return best, best_similarity

    def _cluster_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for _, cluster in self.cluster_observations:
            counts[cluster] = counts.get(cluster, 0) + 1
        return counts

    def _prune(self, now: float) -> None:
        segment_cutoff = now - self.policy.prompt_window_seconds
        while self.segments and self.segments[0].end < segment_cutoff:
            self.segments.popleft()
        speaker_cutoff = now - self.policy.speaker_window_seconds
        while self.cluster_observations and self.cluster_observations[0][0] < speaker_cutoff:
            self.cluster_observations.popleft()

    def _window_start(self, now: float) -> float:
        return self.segments[0].start if self.segments else now

    def _cooldown_ready(self, event_type: str, now: float, cooldown: int) -> bool:
        previous = self.last_emitted.get(event_type)
        if previous is not None and now - previous < cooldown:
            return False
        self.last_emitted[event_type] = now
        return True

    @staticmethod
    def _event(
        event_type: str,
        started_at: float,
        ended_at: float,
        confidence: float,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "type": event_type,
            "startedAt": int(started_at),
            "endedAt": int(max(started_at, ended_at)),
            "durationMs": max(1, int(round(max(0.0, ended_at - started_at) * 1000))),
            "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
            "metadata": metadata,
        }


def audio_model_status(settings: Settings | None = None, *, load: bool = False) -> dict[str, Any]:
    settings = settings or get_settings()
    vad_path = settings.audio_model_root / settings.audio_silero_model
    speaker_root = settings.audio_model_root / settings.audio_speaker_model
    speaker_checkpoint = speaker_root / "embedding_model.ckpt"
    speaker = _model_description(
        speaker_checkpoint,
        settings.audio_speaker_model_sha256,
        "SpeechBrain spkrec-ecapa-voxceleb",
        settings.audio_speaker_model_version,
        "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb",
        "Apache-2.0",
    )
    required_speaker_files = {
        "classifier.ckpt": "fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535",
        "hyperparams.yaml": "6f78854fa04ba59e761437b76a2575d3aba5e5016de3e9b69f0c9a5077fb1a41",
        "mean_var_norm_emb.ckpt": "cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33",
        "label_encoder.txt": "e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9",
    }
    speaker["requiredFiles"] = [
        {
            "path": str(speaker_root / filename),
            "sha256": _sha256(speaker_root / filename) if (speaker_root / filename).is_file() else "",
            "expectedSha256": checksum,
            "ready": bool(
                (speaker_root / filename).is_file()
                and _sha256(speaker_root / filename) == checksum
            ),
        }
        for filename, checksum in required_speaker_files.items()
    ]
    speaker["ready"] = bool(speaker["ready"] and all(item["ready"] for item in speaker["requiredFiles"]))
    result: dict[str, Any] = {
        "enabled": bool(settings.audio_analysis_enabled),
        "available": False,
        "sampleRate": 16000,
        "vad": _model_description(
            vad_path,
            settings.audio_silero_model_sha256,
            "Silero VAD",
            settings.audio_silero_model_version,
            "https://github.com/snakers4/silero-vad",
            "MIT",
        ),
        "speaker": speaker,
    }
    result["available"] = bool(result["vad"]["ready"] and result["speaker"]["ready"])
    if load and result["enabled"] and result["available"]:
        try:
            SileroVad(settings)
            SpeechBrainSpeakerEncoder(settings)
            result["modelsLoaded"] = True
        except Exception as exc:
            result["modelsLoaded"] = False
            result["available"] = False
            result["error"] = str(exc)[:500]
    return result


def _model_description(
    path: Path,
    expected_checksum: str,
    name: str,
    version: str,
    source: str,
    license_name: str,
) -> dict[str, Any]:
    checksum = _sha256(path) if path.is_file() else ""
    expected = expected_checksum.strip().lower()
    return {
        "name": name,
        "version": version,
        "source": source,
        "license": license_name,
        "path": str(path),
        "exists": path.is_file(),
        "checksum": checksum,
        "checksumConfigured": len(expected) == 64,
        "checksumMatches": bool(expected and checksum and hmac.compare_digest(checksum, expected)),
        "ready": bool(path.is_file() and len(expected) == 64 and expected == checksum),
    }


def _require_model(path: Path, expected_checksum: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} model is missing: {path}")
    expected = expected_checksum.strip().lower()
    if len(expected) != 64:
        raise RuntimeError(f"{label} SHA-256 checksum is not configured")
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dbfs(samples: np.ndarray) -> float:
    rms = math.sqrt(float(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0
    return 20.0 * math.log10(max(rms, 1e-9))


def _energy_confidence(value: float, threshold: float) -> float:
    return min(1.0, max(0.0, 0.5 + ((value - threshold) / 30.0)))


def _prompt_confidence(
    sustained: float,
    rolling: float,
    sustained_threshold: float,
    rolling_threshold: float,
    multispeaker: bool,
) -> float:
    score = max(
        sustained / max(sustained_threshold, 0.1),
        rolling / max(rolling_threshold, 0.1),
    ) * 0.6
    if multispeaker:
        score += 0.2
    return min(0.95, max(0.5, score))


def _bounded(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return min(maximum, max(minimum, parsed))


def audio_service_status(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    models = audio_model_status(settings)
    heartbeat: dict[str, Any] = {}
    try:
        from app.services.state_store import StateStore

        raw = StateStore(settings).redis.get("proctorcore:audio:health")
        heartbeat = json.loads(raw) if raw else {}
    except Exception:
        path = settings.local_storage_root / "audio-analysis-health.json"
        if path.is_file():
            try:
                heartbeat = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                heartbeat = {}
    updated_at = int(heartbeat.get("updatedAt") or 0)
    service_fresh = updated_at > 0 and time.time() - updated_at <= int(settings.audio_health_stale_seconds)
    runtime_models = heartbeat.get("models") if isinstance(heartbeat.get("models"), dict) else models
    models_loaded = bool(runtime_models.get("modelsLoaded"))
    ready = bool(
        not settings.audio_analysis_enabled
        or (models.get("available") and models_loaded and service_fresh and heartbeat.get("ready"))
    )
    return {
        "enabled": bool(settings.audio_analysis_enabled),
        "ready": ready,
        "pipelineAvailable": bool(models.get("available") and models_loaded and service_fresh),
        "sampleRate": 16000,
        "vadModelLoaded": models_loaded,
        "speakerModelLoaded": models_loaded,
        "activeSessions": int(heartbeat.get("activeSessions") or 0),
        "lastHeartbeatAt": updated_at or None,
        "models": runtime_models,
        "error": heartbeat.get("error") or models.get("error"),
    }
