import json
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.services.liveness_service import LivenessService


class MemoryStore:
    def __init__(self):
        self.items = {}

    def save(self, challenge):
        self.items[challenge["challengeId"]] = challenge

    def consume(self, challenge_id):
        if challenge_id not in self.items:
            raise ValueError("challenge_expired_or_replayed")
        return self.items.pop(challenge_id)

    def get(self, challenge_id):
        if challenge_id not in self.items:
            raise ValueError("challenge_expired_or_replayed")
        return self.items[challenge_id]


class TrackingStore(MemoryStore):
    def __init__(self):
        super().__init__()
        self.failures = 0
        self.retry_window_seconds = None

    def failure_count(self, _company_id, _user_id, _context_id):
        return self.failures

    def register_failure(self, _company_id, _user_id, _context_id, _window_seconds=None):
        self.failures += 1
        self.retry_window_seconds = _window_seconds
        return self.failures

    def clear_failures(self, _company_id, _user_id, _context_id):
        self.failures = 0


class AdaptiveStore(TrackingStore):
    def __init__(self):
        super().__init__()
        self.pose = {}

    def save_pose_progress(self, challenge_id, progress):
        self.pose[challenge_id] = progress

    def get_pose_progress(self, challenge_id):
        return self.pose.get(challenge_id, {"expectedStep": 0, "holdFrames": 0, "steps": []})

    def consume_pose_progress(self, challenge_id):
        return self.pose.pop(challenge_id, None)


class FakeMatcher:
    def _quality(self, content):
        item = json.loads(content.decode())
        return SimpleNamespace(
            face_count=item.get("faces", 1),
            brightness=item.get("brightness", 120.0),
            blur=item.get("blur", 120.0),
            confidence=item.get("confidence", 0.96),
            face_width=item.get("face_width", 320),
            width=1000,
            face_center_x=item.get("center_x", 0.5),
            face_center_y=item.get("center_y", 0.5),
            headpose_yaw_degrees=item.get("yaw", 0.0),
            antispoof_scores={
                "live": item.get("live", 0.92),
                "print": item.get("print", 0.04),
                "replay": item.get("replay", 0.04),
            },
        )

    def analyse_liveness_frame(
        self,
        content,
        include_headpose=True,
        include_antispoof=True,
        include_chromaticity=True,
    ):
        quality = self._quality(content)
        if not include_headpose:
            quality.headpose_yaw_degrees = None
        if not include_antispoof:
            quality.antispoof_scores = None
        item = json.loads(content.decode())
        chromaticity = item.get("rgb", [1 / 3, 1 / 3, 1 / 3]) if include_chromaticity else None
        return quality, chromaticity

    def analyse_capture_quality_frame(self, content):
        quality = self._quality(content)
        quality.headpose_yaw_degrees = None
        return quality

    def analyse_headpose_frame(self, content):
        return self._quality(content)


def settings(**overrides):
    values = {
        "identity_temporal_passive_pad_enabled": True,
        "identity_require_passive_antispoof": True,
        "identity_active_liveness_enabled": True,
        "identity_headpose_challenge_enabled": True,
        "identity_illumination_challenge_enabled": True,
        "identity_passive_min_valid_frames": 5,
        "identity_liveness_challenge_timeout_ms": 9000,
        "identity_liveness_challenge_ttl_seconds": 45,
        "identity_headpose_turn_degrees": 14.0,
        "identity_headpose_center_degrees": 9.0,
        "identity_headpose_min_hold_frames": 2,
        "identity_headpose_min_progress_degrees": 7.0,
        "identity_liveness_min_brightness": 30.0,
        "identity_liveness_min_blur": 15.0,
        "identity_liveness_min_face_confidence": 0.55,
        "identity_headpose_min_face_confidence": 0.30,
        "identity_liveness_min_face_width_ratio": 0.12,
        "identity_liveness_max_face_width_ratio": 0.75,
        "identity_liveness_center_tolerance_x": 0.30,
        "identity_liveness_center_tolerance_y": 0.35,
        "identity_illumination_phase_ms": 850,
        "identity_illumination_min_frames_per_phase": 2,
        "identity_illumination_min_response": 0.008,
        "identity_illumination_pass_correlation": 0.35,
        "identity_illumination_fail_correlation": -0.05,
    }
    values.update(overrides)
    return Settings(**values)


def issued(service):
    return service.issue(3, 245, "quiz:10", "transaction-123", False, True)


def test_challenge_lifetime_allows_slow_cpu_inference():
    service = LivenessService(FakeMatcher(), settings(), MemoryStore())
    challenge = issued(service)
    assert challenge["expiresAtMs"] - challenge["issuedAtMs"] >= 300_000


def test_illumination_is_not_required_without_explicit_request():
    service = LivenessService(FakeMatcher(), settings(), MemoryStore())
    challenge = service.issue(3, 245, "quiz:10", "transaction-123", False)
    assert challenge["components"]["illumination"] is False


def test_illumination_request_respects_server_feature_switch():
    service = LivenessService(
        FakeMatcher(),
        settings(identity_illumination_challenge_enabled=False),
        MemoryStore(),
    )
    challenge = service.issue(3, 245, "quiz:10", "transaction-123", False, True)
    assert challenge["components"]["illumination"] is False


def test_movement_request_can_disable_headpose_without_disabling_passive_pad():
    service = LivenessService(FakeMatcher(), settings(), MemoryStore())
    challenge = service.issue(
        3,
        245,
        "quiz:10",
        "transaction-123",
        False,
        False,
        False,
    )
    assert challenge["components"]["headPose"] is False
    assert challenge["components"]["passivePad"] is True
    assert challenge["passiveCaptureMs"] == 4500


def test_liveness_signal_models_receive_only_their_evidence_stream():
    service = LivenessService(FakeMatcher(), settings(), MemoryStore())
    frames = [
        {
            "bytes": json.dumps({"live": 0.91, "yaw": 12.0}).encode(),
            "elapsedMs": 100,
            "purpose": "passive",
        },
        {
            "bytes": json.dumps({"live": 0.01, "replay": 0.98, "yaw": 18.0}).encode(),
            "elapsedMs": 400,
            "purpose": "illumination",
        },
        {
            "bytes": json.dumps({"live": 0.01, "yaw": -18.0}).encode(),
            "elapsedMs": 700,
            "purpose": "headpose",
        },
    ]

    observations, invalid = service._observations(frames, False, None)

    assert invalid == []
    assert observations[0]["quality"].antispoof_scores is not None
    assert observations[0]["chromaticity"] is None
    assert observations[0]["quality"].headpose_yaw_degrees is None
    assert observations[1]["quality"].antispoof_scores is None
    assert observations[1]["chromaticity"] is not None
    assert observations[1]["quality"].headpose_yaw_degrees is None
    assert observations[2]["quality"].antispoof_scores is None
    assert observations[2]["chromaticity"] is None
    assert observations[2]["quality"].headpose_yaw_degrees == -18.0


def evidence_for(challenge, passive=None, movement="correct", illumination="correct", faces=1):
    frames = []
    elapsed = 100
    while elapsed <= challenge["durationMs"]:
        movement_step = next(
            (step for step in challenge["movementSteps"] if step["startMs"] <= elapsed < step["endMs"]),
            challenge["movementSteps"][-1],
        )
        action = movement_step["action"]
        progress = (elapsed - movement_step["startMs"]) / max(1, movement_step["endMs"] - movement_step["startMs"])
        yaw = 0.0
        if action == "left":
            yaw = -20.0 * min(1.0, progress * 2.0)
        elif action == "right":
            yaw = 20.0 * min(1.0, progress * 2.0)
        if movement == "inverse":
            yaw = -yaw
        elif movement == "none":
            yaw = 0.0
        elif movement == "early" and movement_step["step"] == 0:
            yaw = 20.0

        rgb = [1 / 3, 1 / 3, 1 / 3]
        light_step = next(
            (step for step in challenge["illuminationSteps"] if step["startMs"] <= elapsed < step["endMs"]),
            None,
        )
        if light_step and light_step["colour"] != "neutral" and illumination != "weak":
            expected = LivenessService.COLOURS[light_step["colour"]]["rgb"]
            total = sum(expected)
            chroma = [value / total for value in expected]
            strength = 0.10
            rgb = [(1 / 3) + (strength * (value - (1 / 3))) for value in chroma]
            if illumination == "inverse":
                rgb = [(1 / 3) - (strength * (value - (1 / 3))) for value in chroma]

        scores = passive(elapsed) if passive else {"live": 0.92, "print": 0.04, "replay": 0.04}
        payload = {"yaw": yaw, "rgb": rgb, "faces": faces, **scores}
        frames.append({
            "bytes": json.dumps(payload).encode(),
            "capturedAtMs": challenge["issuedAtMs"] + elapsed,
            "elapsedMs": elapsed,
        })
        elapsed += 200
    return frames


def validate(service, challenge, frames):
    return service.validate(
        challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
        "transaction-123", frames, False,
    )


def test_genuine_temporal_multisignal_sequence_passes():
    service = LivenessService(FakeMatcher(), settings(), MemoryStore())
    challenge = issued(service)
    result = validate(service, challenge, evidence_for(challenge))
    assert result["overall"] == "pass"


def test_one_bad_passive_frame_does_not_fail_good_sequence():
    service = LivenessService(FakeMatcher(), settings(), MemoryStore())
    challenge = issued(service)
    bad_at = challenge["durationMs"] // 2
    result = validate(service, challenge, evidence_for(
        challenge,
        passive=lambda elapsed: {"live": 0.05, "print": 0.92, "replay": 0.03}
        if abs(elapsed - bad_at) < 120 else {"live": 0.93, "print": 0.04, "replay": 0.03},
    ))
    assert result["passivePad"]["result"] == "pass"


@pytest.mark.parametrize(("attack", "expected"), [("print", "print_attack"), ("replay", "replay_attack")])
def test_consistent_attack_fails(attack, expected):
    service = LivenessService(FakeMatcher(), settings(), MemoryStore())
    challenge = issued(service)
    scores = {"live": 0.04, "print": 0.03, "replay": 0.03}
    scores[attack] = 0.93
    result = validate(service, challenge, evidence_for(
        challenge,
        passive=lambda _elapsed: scores,
        movement="none",
        illumination="inverse",
    ))
    assert result["overall"] == "fail"
    assert result["passivePad"]["reason"] == expected


def test_passive_attack_signal_alone_becomes_inconclusive_when_active_signals_pass():
    service = LivenessService(FakeMatcher(), settings(), MemoryStore())
    challenge = issued(service)
    scores = {"live": 0.04, "print": 0.93, "replay": 0.03}
    result = validate(service, challenge, evidence_for(challenge, passive=lambda _elapsed: scores))
    assert result["overall"] == "inconclusive"
    assert result["reason"] == "liveness_signal_conflict"


def test_ambiguous_passive_pad_is_inconclusive():
    service = LivenessService(FakeMatcher(), settings(), MemoryStore())
    challenge = issued(service)
    scores = {"live": 0.56, "print": 0.24, "replay": 0.20}
    result = validate(service, challenge, evidence_for(challenge, passive=lambda _elapsed: scores))
    assert result["overall"] == "inconclusive"
    assert result["passivePad"]["reason"] == "passive_ambiguous"


@pytest.mark.parametrize("first", ["left", "right"])
def test_randomized_turn_then_center_succeeds(monkeypatch, first):
    sequence = (first, "center")
    monkeypatch.setattr("secrets.choice", lambda _items: sequence)
    service = LivenessService(FakeMatcher(), settings(), MemoryStore())
    challenge = issued(service)
    assert validate(service, challenge, evidence_for(challenge))["headPose"]["result"] == "pass"


@pytest.mark.parametrize(("movement", "reason"), [
    ("inverse", "headpose_sequence_failed"),
    ("none", "headpose_sequence_failed"),
    ("early", "movement_before_challenge"),
])
def test_invalid_movement_fails(movement, reason):
    service = LivenessService(FakeMatcher(), settings(), MemoryStore())
    challenge = issued(service)
    result = validate(service, challenge, evidence_for(challenge, movement=movement))
    assert result["overall"] == "fail"
    assert result["headPose"]["reason"] == reason


def test_expired_or_replayed_challenge_is_rejected():
    store = MemoryStore()
    service = LivenessService(FakeMatcher(), settings(), store)
    challenge = issued(service)
    frames = evidence_for(challenge)
    validate(service, challenge, frames)
    with pytest.raises(ValueError, match="challenge_expired_or_replayed"):
        validate(service, challenge, frames)


def test_expired_challenge_is_rejected():
    store = MemoryStore()
    service = LivenessService(FakeMatcher(), settings(), store)
    challenge = issued(service)
    store.items[challenge["challengeId"]]["expiresAtMs"] = 0
    with pytest.raises(ValueError, match="challenge_expired_or_replayed"):
        validate(service, challenge, evidence_for(challenge))


@pytest.mark.parametrize(("field", "value"), [
    ("nonce", "wrong-nonce"),
    ("company", 4),
    ("user", 246),
    ("context", "quiz:11"),
    ("transaction", "transaction-999"),
])
def test_challenge_is_bound_to_tenant_user_context_and_transaction(field, value):
    service = LivenessService(FakeMatcher(), settings(), MemoryStore())
    challenge = issued(service)
    arguments = {
        "nonce": challenge["nonce"],
        "company": 3,
        "user": 245,
        "context": "quiz:10",
        "transaction": "transaction-123",
    }
    arguments[field] = value
    with pytest.raises(ValueError, match="challenge_binding_mismatch"):
        service.validate(
            challenge["challengeId"], arguments["nonce"], arguments["company"],
            arguments["user"], arguments["context"], arguments["transaction"],
            evidence_for(challenge), False,
        )


def test_challenge_timeout_is_rejected():
    service = LivenessService(FakeMatcher(), settings(), MemoryStore())
    challenge = issued(service)
    frames = evidence_for(challenge)
    frames[-1]["elapsedMs"] = challenge["durationMs"] + 1000
    with pytest.raises(ValueError, match="invalid_challenge_timestamps"):
        validate(service, challenge, frames)


def test_multiple_faces_during_challenge_fails():
    service = LivenessService(FakeMatcher(), settings(), MemoryStore())
    challenge = issued(service)
    result = validate(service, challenge, evidence_for(challenge, faces=2))
    assert result["overall"] == "fail"
    assert result["reason"] == "multiple_faces"


def test_face_disappears_is_inconclusive():
    service = LivenessService(FakeMatcher(), settings(), MemoryStore())
    challenge = issued(service)
    frames = evidence_for(challenge)
    for frame in frames:
        item = json.loads(frame["bytes"].decode())
        item["faces"] = 0
        frame["bytes"] = json.dumps(item).encode()
    result = validate(service, challenge, frames)
    assert result["overall"] == "inconclusive"


@pytest.mark.parametrize(("mode", "expected"), [
    ("correct", "pass"),
    ("inverse", "fail"),
    ("weak", "inconclusive"),
])
def test_illumination_temporal_response(mode, expected):
    service = LivenessService(FakeMatcher(), settings(), MemoryStore())
    challenge = issued(service)
    result = validate(service, challenge, evidence_for(challenge, illumination=mode))
    assert result["illumination"]["result"] == expected


def test_disabled_settings_preserve_compatible_pass_through():
    configured = settings(
        identity_temporal_passive_pad_enabled=False,
        identity_require_passive_antispoof=False,
        identity_active_liveness_enabled=False,
        identity_require_active_liveness=False,
        identity_headpose_challenge_enabled=False,
        identity_illumination_challenge_enabled=False,
    )
    service = LivenessService(FakeMatcher(), configured, MemoryStore())
    challenge = issued(service)
    assert challenge["required"] is False
    result = validate(service, challenge, evidence_for(challenge))
    assert result["overall"] == "pass"


def test_legacy_active_and_required_headpose_flags_enable_random_head_challenge():
    configured = settings(
        identity_active_liveness_enabled=False,
        identity_require_active_liveness=True,
        identity_headpose_challenge_enabled=False,
        identity_require_headpose_liveness=True,
        identity_illumination_challenge_enabled=False,
    )
    service = LivenessService(FakeMatcher(), configured, MemoryStore())
    challenge = issued(service)
    assert challenge["components"]["headPose"] is True


def test_quality_probe_provides_framing_feedback_without_consuming_challenge():
    store = MemoryStore()
    service = LivenessService(FakeMatcher(), settings(), store)
    challenge = issued(service)
    image = json.dumps({"center_x": 0.9}).encode()
    result = service.check_frame(
        challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
        "transaction-123", image, False,
    )
    assert result["ready"] is False
    assert result["reason"] == "face_not_centered"
    assert result["diagnostics"]["faceCenterX"] == 0.9
    assert challenge["challengeId"] in store.items


def test_enrollment_quality_probe_requires_reference_grade_confidence():
    store = MemoryStore()
    service = LivenessService(FakeMatcher(), settings(), store)
    challenge = service.issue(3, 245, "quiz:10", "transaction-123", True)

    result = service.check_frame(
        challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
        "transaction-123", json.dumps({"confidence": 0.70}).encode(), True,
    )

    assert result["ready"] is False
    assert result["reason"] == "low_face_confidence"
    assert challenge["challengeId"] in store.items


def test_retry_limit_counts_failures_and_success_clears_them():
    store = TrackingStore()
    service = LivenessService(FakeMatcher(), settings(identity_liveness_retry_limit=2), store)
    challenge = issued(service)
    validate(service, challenge, evidence_for(challenge, movement="none"))
    assert store.failures == 1

    challenge = issued(service)
    validate(service, challenge, evidence_for(challenge))
    assert store.failures == 0


def test_inconclusive_capture_does_not_lock_out_genuine_user():
    store = TrackingStore()
    service = LivenessService(FakeMatcher(), settings(identity_liveness_retry_limit=1), store)
    challenge = issued(service)

    result = validate(service, challenge, evidence_for(challenge, illumination="weak"))

    assert result["overall"] == "inconclusive"
    assert store.failures == 0
    assert issued(service)["required"] is True


def test_retry_limit_rejects_new_challenge_after_configured_failures():
    store = TrackingStore()
    service = LivenessService(FakeMatcher(), settings(identity_liveness_retry_limit=1), store)
    challenge = issued(service)
    validate(service, challenge, evidence_for(challenge, movement="none"))
    with pytest.raises(ValueError, match="liveness_retry_limit_reached"):
        issued(service)


def test_challenge_retry_policy_overrides_server_defaults():
    store = TrackingStore()
    store.failures = 2
    service = LivenessService(FakeMatcher(), settings(identity_liveness_retry_limit=1), store)

    challenge = service.issue(
        3, 245, "quiz:10", "transaction-123", False, True, True, 3, 1200,
    )
    result = validate(service, challenge, evidence_for(challenge, movement="none"))

    assert challenge["maxAttempts"] == 3
    assert result["overall"] == "fail"
    assert store.retry_window_seconds == 1200


def test_adaptive_headpose_advances_only_after_each_pose_is_confirmed():
    store = AdaptiveStore()
    service = LivenessService(FakeMatcher(), settings(), store)
    challenge = issued(service)
    assert challenge["adaptiveHeadPose"] is True

    yaws = {"center": 0.0, "left": -20.0, "right": 20.0}
    for index, step in enumerate(challenge["movementSteps"]):
        image = json.dumps({"yaw": yaws[step["action"]]}).encode()
        first = service.check_pose_frame(
            challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
            "transaction-123", index, image, False,
        )
        second = service.check_pose_frame(
            challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
            "transaction-123", index, image, False,
        )
        assert first["reached"] is False
        assert second["reached"] is True

    result = validate(service, challenge, evidence_for(challenge))
    assert result["headPose"]["result"] == "pass"


def test_adaptive_headpose_rejects_out_of_order_step():
    store = AdaptiveStore()
    service = LivenessService(FakeMatcher(), settings(), store)
    challenge = issued(service)
    with pytest.raises(ValueError, match="headpose_step_out_of_order"):
        service.check_pose_frame(
            challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
            "transaction-123", 1, json.dumps({"yaw": 20.0}).encode(), False,
        )


@pytest.mark.parametrize(("direction", "turned_yaw"), [("left", -4.0), ("right", 34.0)])
def test_adaptive_headpose_uses_the_users_neutral_baseline(monkeypatch, direction, turned_yaw):
    monkeypatch.setattr("secrets.choice", lambda _items: (direction, "center"))
    store = AdaptiveStore()
    service = LivenessService(FakeMatcher(), settings(), store)
    challenge = issued(service)

    sequence = (15.0, turned_yaw, 15.0)
    for index, yaw in enumerate(sequence):
        image = json.dumps({"yaw": yaw}).encode()
        assert service.check_pose_frame(
            challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
            "transaction-123", index, image, False,
        )["reached"] is False
        assert service.check_pose_frame(
            challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
            "transaction-123", index, image, False,
        )["reached"] is True

    progress = store.get_pose_progress(challenge["challengeId"])
    assert progress["expectedStep"] == 3
    assert progress["baselineYaw"] == pytest.approx(15.0)


def test_adaptive_headpose_does_not_advance_for_the_wrong_direction(monkeypatch):
    monkeypatch.setattr("secrets.choice", lambda _items: ("left", "center"))
    store = AdaptiveStore()
    service = LivenessService(FakeMatcher(), settings(), store)
    challenge = issued(service)

    center = json.dumps({"yaw": 12.0}).encode()
    for _ in range(2):
        service.check_pose_frame(
            challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
            "transaction-123", 0, center, False,
        )

    wrong_turn = json.dumps({"yaw": 36.0}).encode()
    for _ in range(3):
        result = service.check_pose_frame(
            challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
            "transaction-123", 1, wrong_turn, False,
        )
        assert result["reached"] is False
    assert store.get_pose_progress(challenge["challengeId"])["expectedStep"] == 1


def test_pose_burst_confirms_stable_center_without_waiting_for_network_round_trips():
    store = AdaptiveStore()
    service = LivenessService(FakeMatcher(), settings(), store)
    challenge = service.issue(3, 245, "quiz:10", "transaction-123", True)
    frames = [json.dumps({"yaw": yaw}).encode() for yaw in (11.5, 12.0, 11.8)]

    result = service.check_pose_frames(
        challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
        "transaction-123", 0, frames, True,
    )

    assert result["reached"] is True
    assert result["batchFramesProcessed"] == 2
    assert result["batchRejectedFrames"] == 0


def test_pose_frames_use_movement_quality_not_strict_enrollment_quality():
    store = AdaptiveStore()
    configured = settings(
        identity_enrollment_min_blur=45.0,
        identity_liveness_min_blur=15.0,
    )
    service = LivenessService(FakeMatcher(), configured, store)
    challenge = service.issue(3, 245, "quiz:10", "transaction-123", True)
    moving_frames = [json.dumps({"yaw": 8.0, "blur": 20.0}).encode()] * 2

    result = service.check_pose_frames(
        challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
        "transaction-123", 0, moving_frames, True,
    )

    assert result["reached"] is True


def test_turn_frames_use_pose_confidence_without_weakening_straight_capture(monkeypatch):
    monkeypatch.setattr("secrets.choice", lambda _items: ("left", "center"))
    store = AdaptiveStore()
    service = LivenessService(FakeMatcher(), settings(), store)
    challenge = service.issue(3, 245, "quiz:10", "transaction-123", True)

    straight = json.dumps({"yaw": 0.0, "confidence": 0.90}).encode()
    for _ in range(2):
        service.check_pose_frame(
            challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
            "transaction-123", 0, straight, True,
        )

    turned = json.dumps({"yaw": -18.0, "confidence": 0.35}).encode()
    first = service.check_pose_frame(
        challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
        "transaction-123", 1, turned, True,
    )
    second = service.check_pose_frame(
        challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
        "transaction-123", 1, turned, True,
    )

    assert first["reached"] is False
    assert second["reached"] is True


def test_enrollment_liveness_evidence_uses_liveness_not_reference_quality(monkeypatch):
    monkeypatch.setattr("secrets.choice", lambda _items: ("left", "center"))
    store = AdaptiveStore()
    service = LivenessService(FakeMatcher(), settings(), store)
    challenge = service.issue(3, 245, "quiz:10", "transaction-123", True)

    for index, yaw in enumerate((0.0, -20.0, 0.0)):
        pose = json.dumps({"yaw": yaw, "confidence": 0.90}).encode()
        for _ in range(2):
            service.check_pose_frame(
                challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
                "transaction-123", index, pose, True,
            )

    frames = evidence_for(challenge)
    for frame in frames:
        payload = json.loads(frame["bytes"].decode())
        payload["confidence"] = 0.60
        payload["face_width"] = 150
        frame["bytes"] = json.dumps(payload).encode()

    result = service.validate(
        challenge["challengeId"], challenge["nonce"], 3, 245, "quiz:10",
        "transaction-123", frames, True,
    )

    assert result["invalidFrameCount"] == 0
    assert result["overall"] == "pass"
