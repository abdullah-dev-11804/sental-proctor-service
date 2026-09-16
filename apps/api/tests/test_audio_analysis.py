import hashlib
import sys
from dataclasses import replace
from types import ModuleType

import numpy as np

from app.core.config import Settings
from app.services.audio_analysis import AudioEventEngine, AudioPolicy, SpeechBrainSpeakerEncoder


FRAME_SECONDS = 512 / 16000


class SequenceVad:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self, _frame):
        return next(self.values, 0.0)


def policy(**changes) -> AudioPolicy:
    base = AudioPolicy(
        enabled=True,
        sample_rate=16000,
        noise_enabled=True,
        noise_threshold_dbfs=-35.0,
        noise_min_seconds=1.0,
        noise_cooldown_seconds=30,
        speech_enabled=True,
        vad_threshold=0.6,
        speech_min_seconds=0.5,
        speech_cooldown_seconds=5,
        second_speaker_enabled=True,
        speaker_similarity_threshold=0.72,
        speaker_min_segments=2,
        speaker_window_seconds=60,
        second_speaker_cooldown_seconds=60,
        prompting_enabled=True,
        prompt_sustained_seconds=3.0,
        prompt_rolling_seconds=3.0,
        prompt_window_seconds=60,
        prompt_multispeaker=True,
        prompt_cooldown_seconds=60,
    )
    return replace(base, **changes)


def frames(count: int, amplitude: float = 0.0, sign: float = 1.0) -> np.ndarray:
    return np.full(count * 512, amplitude * sign, dtype=np.float32)


def finish_with_silence(engine: AudioEventEngine, started_at: float, seconds: float = 0.5):
    count = int(seconds / FRAME_SECONDS) + 1
    return engine.process(frames(count), started_at)


def test_silence_has_no_audio_violation() -> None:
    engine = AudioEventEngine(policy(), SequenceVad([0.0] * 100))
    events = engine.process(frames(50), 1000.0) + engine.flush(1002.0)
    assert events == []


def test_short_isolated_sound_is_not_sustained_noise() -> None:
    engine = AudioEventEngine(policy(noise_min_seconds=1.0), SequenceVad([0.0] * 100))
    events = engine.process(frames(10, 0.2), 1000.0)
    events += engine.process(frames(20), 1000.0 + (10 * FRAME_SECONDS))
    assert not [event for event in events if event["type"] == "background_noise"]


def test_sustained_loud_non_speech_creates_one_background_noise_event() -> None:
    engine = AudioEventEngine(policy(noise_min_seconds=1.0), SequenceVad([0.0] * 200))
    events = engine.process(frames(50, 0.2), 1000.0)
    events += engine.process(frames(20), 1000.0 + (50 * FRAME_SECONDS))
    noise = [event for event in events if event["type"] == "background_noise"]
    assert len(noise) == 1
    assert noise[0]["metadata"]["meanDbfs"] > -20


def test_brief_human_speech_creates_speech_event_without_prompting() -> None:
    speech_frames = 24
    vad = SequenceVad(([0.95] * speech_frames) + ([0.0] * 20))
    engine = AudioEventEngine(policy(), vad)
    events = engine.process(frames(speech_frames, 0.08), 1000.0)
    events += finish_with_silence(engine, 1000.0 + speech_frames * FRAME_SECONDS)
    assert [event["type"] for event in events] == ["speech_detected"]


def test_continuous_speech_can_create_prompting_review_signal() -> None:
    speech_frames = 110
    vad = SequenceVad(([0.95] * speech_frames) + ([0.0] * 20))
    engine = AudioEventEngine(policy(prompt_sustained_seconds=3.0), vad)
    events = engine.process(frames(speech_frames, 0.08), 1000.0)
    events += finish_with_silence(engine, 1000.0 + speech_frames * FRAME_SECONDS)
    assert {event["type"] for event in events} == {"speech_detected", "possible_prompting"}
    prompt = next(event for event in events if event["type"] == "possible_prompting")
    assert prompt["metadata"]["reviewSignalOnly"] is True


def test_one_speaker_with_volume_changes_does_not_become_second_voice() -> None:
    def same_speaker(_samples):
        return np.array([1.0, 0.0], dtype=np.float32)

    values = []
    samples = []
    for amplitude in (0.03, 0.08, 0.15):
        values.extend([0.95] * 34 + [0.0] * 14)
        samples.extend([frames(34, amplitude), frames(14)])
    engine = AudioEventEngine(policy(speaker_min_segments=2), SequenceVad(values), same_speaker)
    events = []
    timestamp = 1000.0
    for sample in samples:
        events += engine.process(sample, timestamp)
        timestamp += sample.size / 16000
    assert "second_voice_detected" not in [event["type"] for event in events]


def test_two_stable_speakers_create_second_voice_after_repeated_evidence() -> None:
    def encoder(samples):
        return np.array([1.0, 0.0], dtype=np.float32) if float(np.mean(samples)) >= 0 else np.array([0.0, 1.0])

    values = []
    samples = []
    for sign in (1.0, 1.0, -1.0, -1.0):
        values.extend([0.95] * 34 + [0.0] * 14)
        samples.extend([frames(34, 0.08, sign), frames(14)])
    engine = AudioEventEngine(policy(speaker_min_segments=2), SequenceVad(values), encoder)
    events = []
    timestamp = 1000.0
    for sample in samples:
        events += engine.process(sample, timestamp)
        timestamp += sample.size / 16000
    second = [event for event in events if event["type"] == "second_voice_detected"]
    assert len(second) == 1
    assert second[0]["metadata"]["stableSpeakerClusters"] == 2


def test_disabled_audio_analysis_emits_nothing() -> None:
    engine = AudioEventEngine(policy(enabled=False), SequenceVad([0.95] * 200))
    assert engine.process(frames(100, 0.2), 1000.0) == []


def test_one_continuous_speech_event_is_consolidated() -> None:
    speech_frames = 160
    vad = SequenceVad(([0.95] * speech_frames) + ([0.0] * 20))
    engine = AudioEventEngine(policy(prompting_enabled=False), vad)
    events = engine.process(frames(speech_frames, 0.08), 1000.0)
    events += finish_with_silence(engine, 1000.0 + speech_frames * FRAME_SECONDS)
    assert len([event for event in events if event["type"] == "speech_detected"]) == 1


def test_speechbrain_runtime_files_use_writable_storage_not_model_directory(tmp_path, monkeypatch) -> None:
    model_root = tmp_path / "models" / "audio"
    speaker_root = model_root / "speechbrain-spkrec-ecapa-voxceleb"
    speaker_root.mkdir(parents=True)
    checkpoint = speaker_root / "embedding_model.ckpt"
    checkpoint.write_bytes(b"pinned-speaker-model")
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    storage_root = tmp_path / "storage"
    captured = {}

    class FakeEncoderClassifier:
        @classmethod
        def from_hparams(cls, **kwargs):
            captured.update(kwargs)
            return object()

    torch_module = ModuleType("torch")
    speechbrain_module = ModuleType("speechbrain")
    inference_module = ModuleType("speechbrain.inference")
    speaker_module = ModuleType("speechbrain.inference.speaker")
    speaker_module.EncoderClassifier = FakeEncoderClassifier
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "speechbrain", speechbrain_module)
    monkeypatch.setitem(sys.modules, "speechbrain.inference", inference_module)
    monkeypatch.setitem(sys.modules, "speechbrain.inference.speaker", speaker_module)

    encoder = SpeechBrainSpeakerEncoder(Settings(
        audio_model_root=model_root,
        audio_speaker_model_sha256=checksum,
        local_storage_root=storage_root,
    ))

    assert captured["source"] == str(speaker_root)
    assert captured["savedir"] == str(encoder.cache_root)
    assert encoder.cache_root.is_dir()
    assert storage_root in encoder.cache_root.parents
    assert speaker_root not in encoder.cache_root.parents
