from __future__ import annotations

import json
import logging
import secrets
import time
from statistics import median
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings


logger = logging.getLogger(__name__)


class LivenessChallengeStore:
    """One-use, short-lived Redis storage for server-authored challenges."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._redis = None

    @property
    def redis(self):
        if self._redis is None:
            from redis import Redis

            self._redis = Redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                socket_timeout=2,
            )
        return self._redis

    def save(self, challenge: dict[str, Any]) -> None:
        ttl = max(10, int(self.settings.identity_liveness_challenge_ttl_seconds))
        key = self._key(str(challenge["challengeId"]))
        if not self.redis.set(key, json.dumps(challenge), ex=ttl, nx=True):
            raise RuntimeError("challenge_collision")

    def consume(self, challenge_id: str) -> dict[str, Any]:
        key = self._key(challenge_id)
        raw = self.redis.getdel(key)
        if not raw:
            raise ValueError("challenge_expired_or_replayed")
        return json.loads(raw)

    def get(self, challenge_id: str) -> dict[str, Any]:
        raw = self.redis.get(self._key(challenge_id))
        if not raw:
            raise ValueError("challenge_expired_or_replayed")
        return json.loads(raw)

    def failure_count(self, company_id: int, user_id: int, context_id: str) -> int:
        return int(self.redis.get(self._attempt_key(company_id, user_id, context_id)) or 0)

    def register_failure(self, company_id: int, user_id: int, context_id: str) -> int:
        key = self._attempt_key(company_id, user_id, context_id)
        pipeline = self.redis.pipeline()
        pipeline.incr(key)
        pipeline.expire(key, max(60, int(self.settings.identity_liveness_retry_window_seconds)))
        count, _expiry = pipeline.execute()
        return int(count)

    def clear_failures(self, company_id: int, user_id: int, context_id: str) -> None:
        self.redis.delete(self._attempt_key(company_id, user_id, context_id))

    def save_pose_progress(self, challenge_id: str, progress: dict[str, Any]) -> None:
        ttl = max(10, int(self.settings.identity_liveness_challenge_ttl_seconds))
        self.redis.set(self._progress_key(challenge_id), json.dumps(progress), ex=ttl)

    def get_pose_progress(self, challenge_id: str) -> dict[str, Any]:
        raw = self.redis.get(self._progress_key(challenge_id))
        return json.loads(raw) if raw else {"expectedStep": 0, "holdFrames": 0, "steps": []}

    def consume_pose_progress(self, challenge_id: str) -> dict[str, Any] | None:
        raw = self.redis.getdel(self._progress_key(challenge_id))
        return json.loads(raw) if raw else None

    def _attempt_key(self, company_id: int, user_id: int, context_id: str) -> str:
        safe_context = self._key(context_id).split(":", 2)[-1]
        return f"proctorcore:liveness-attempts:{int(company_id)}:{int(user_id)}:{safe_context}"

    def _progress_key(self, challenge_id: str) -> str:
        return f"{self._key(challenge_id)}:pose-progress"

    @staticmethod
    def _key(challenge_id: str) -> str:
        safe = "".join(ch for ch in challenge_id if ch.isalnum() or ch in "-_")[:128]
        if not safe:
            raise ValueError("invalid_challenge_id")
        return f"proctorcore:liveness:{safe}"


class LivenessService:
    """Issues and validates temporal PAD, head-pose, and RGB illumination signals."""

    MOVEMENT_SEQUENCES = (("left", "center"), ("right", "center"))
    COLOURS = {
        "neutral": {"hex": "#ffffff", "rgb": [1.0, 1.0, 1.0]},
        "red": {"hex": "#ff4d5f", "rgb": [1.0, 0.20, 0.24]},
        "green": {"hex": "#35d07f", "rgb": [0.18, 1.0, 0.42]},
        "blue": {"hex": "#4d7dff", "rgb": [0.22, 0.38, 1.0]},
    }

    def __init__(
        self,
        matcher,
        settings: Settings | None = None,
        store: LivenessChallengeStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.matcher = matcher
        self.store = store or LivenessChallengeStore(self.settings)

    def issue(
        self,
        company_id: int,
        user_id: int,
        context_id: str,
        transaction_id: str,
        enrollment: bool,
    ) -> dict[str, Any]:
        required = self._required_components(enrollment)
        if any(required.values()) and hasattr(self.store, "failure_count"):
            failures = self.store.failure_count(company_id, user_id, context_id)
            if failures >= max(1, int(self.settings.identity_liveness_retry_limit)):
                raise ValueError("liveness_retry_limit_reached")
        challenge_id = secrets.token_urlsafe(24)
        nonce = secrets.token_urlsafe(32)
        issued_at = int(time.time() * 1000)
        timeout_ms = max(3000, int(self.settings.identity_liveness_challenge_timeout_ms))
        movements = list(secrets.choice(self.MOVEMENT_SEQUENCES)) if required["headPose"] else ["center"]
        movement_steps = self._movement_steps(movements)
        illumination = self._illumination_steps() if required["illumination"] else []
        adaptive_headpose = bool(required["headPose"] and hasattr(self.store, "save_pose_progress"))
        pose_step_timeout_ms = 3000
        duration_ms = max(
            movement_steps[-1]["endMs"] if movement_steps else 0,
            illumination[-1]["endMs"] if illumination else 0,
            min(timeout_ms, int(self.settings.identity_passive_capture_window_ms)),
            min(timeout_ms, len(movement_steps) * pose_step_timeout_ms) if adaptive_headpose else 0,
        )
        duration_ms = min(duration_ms, timeout_ms)
        challenge = {
            "challengeId": challenge_id,
            "nonce": nonce,
            "companyId": int(company_id),
            "userId": int(user_id),
            "contextId": str(context_id),
            "transactionId": str(transaction_id),
            "enrollment": bool(enrollment),
            "issuedAtMs": issued_at,
            "expiresAtMs": issued_at + (max(10, int(self.settings.identity_liveness_challenge_ttl_seconds)) * 1000),
            "durationMs": duration_ms,
            "required": required,
            "movementSteps": movement_steps,
            "illuminationSteps": illumination,
            "adaptiveHeadPose": adaptive_headpose,
            "poseStepTimeoutMs": pose_step_timeout_ms,
            "minimumCaptureMs": max(
                illumination[-1]["endMs"] if illumination else 0,
                min(timeout_ms, int(self.settings.identity_passive_capture_window_ms)),
            ),
        }
        self.store.save(challenge)
        if adaptive_headpose:
            self.store.save_pose_progress(challenge_id, {
                "expectedStep": 0,
                "holdFrames": 0,
                "firstDirectedYaw": None,
                "lastCenterYaw": 0.0,
                "steps": [],
            })
        return self._public_challenge(challenge)

    def required_components(self, enrollment: bool) -> dict[str, bool]:
        return self._required_components(enrollment)

    def validate(
        self,
        challenge_id: str,
        nonce: str,
        company_id: int,
        user_id: int,
        context_id: str,
        transaction_id: str,
        evidence: list[dict[str, Any]],
        enrollment: bool,
        quality_policy: dict | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        challenge = self.store.consume(challenge_id)
        self._validate_binding(
            challenge,
            nonce,
            company_id,
            user_id,
            context_id,
            transaction_id,
            enrollment,
        )
        try:
            self._validate_evidence_timing(challenge, evidence)
        except ValueError:
            if hasattr(self.store, "register_failure"):
                self.store.register_failure(company_id, user_id, context_id)
            raise
        adaptive_progress = (
            self.store.consume_pose_progress(challenge_id)
            if challenge.get("adaptiveHeadPose") and hasattr(self.store, "consume_pose_progress")
            else None
        )
        observations, invalid = self._observations(
            evidence,
            enrollment,
            quality_policy,
            include_headpose=not bool(challenge.get("adaptiveHeadPose")),
        )
        passive = self._passive_result(
            observations,
            invalid,
            challenge["required"]["passivePad"],
            int(challenge["durationMs"]),
        )
        headpose = (
            self._adaptive_headpose_result(challenge, adaptive_progress)
            if challenge.get("adaptiveHeadPose")
            else self._headpose_result(observations, challenge)
        )
        illumination = self._illumination_result(observations, challenge)
        components = [passive, headpose, illumination]
        required_results = [item for item in components if item["required"]]
        invalid_reasons = self._reason_counts(invalid)
        total_frames = len(observations) + len(invalid)
        invalid_ratio = len(invalid) / max(1, total_frames)
        if invalid_reasons.get("multiple_faces", 0) > 0:
            overall = "fail"
            reason = "multiple_faces"
        elif invalid_reasons.get("no_face", 0) > 0:
            overall = "inconclusive"
            reason = "face_disappeared"
        elif invalid_ratio > float(self.settings.identity_liveness_max_invalid_frame_ratio):
            overall = "inconclusive"
            reason = "unstable_capture_quality"
        elif headpose["required"] and headpose["result"] == "fail":
            overall = "fail"
            reason = headpose["reason"]
        elif passive["result"] == "fail" and illumination["required"] and illumination["result"] == "fail":
            overall = "fail"
            reason = passive["reason"]
        elif len(required_results) == 1 and required_results[0]["result"] == "fail":
            # Compatibility mode for deployments that explicitly enable only one component.
            overall = "fail"
            reason = required_results[0]["reason"]
        elif any(item["result"] == "fail" for item in required_results):
            overall = "inconclusive"
            reason = "liveness_signal_conflict"
        elif any(item["result"] == "inconclusive" for item in required_results):
            overall = "inconclusive"
            reason = next(item["reason"] for item in required_results if item["result"] == "inconclusive")
        else:
            overall = "pass"
            reason = "ok"
        result = {
            "overall": overall,
            "reason": reason,
            "challengeId": challenge_id,
            "issuedAtMs": challenge["issuedAtMs"],
            "completedAtMs": int(time.time() * 1000),
            "durationMs": max((item["elapsedMs"] for item in evidence), default=0),
            "usableFrameCount": len(observations),
            "invalidFrameCount": len(invalid),
            "passivePad": passive,
            "headPose": headpose,
            "illumination": illumination,
            "quality": {
                "invalidReasons": invalid_reasons,
                "invalidFrameRatio": round(invalid_ratio, 4),
                "medianBrightness": self._quality_median(observations, "brightness"),
                "medianBlur": self._quality_median(observations, "blur"),
                "medianFaceConfidence": self._quality_median(observations, "confidence"),
            },
            "processingMs": int(round((time.perf_counter() - started) * 1000)),
        }
        if overall == "pass" and hasattr(self.store, "clear_failures"):
            self.store.clear_failures(company_id, user_id, context_id)
        elif overall == "fail" and hasattr(self.store, "register_failure"):
            self.store.register_failure(company_id, user_id, context_id)
        logger.info(
            "liveness_result challenge=%s company=%s user=%s overall=%s reason=%s usable=%s invalid=%s passive=%s:%s aggregate=%s head=%s:%s steps=%s illumination=%s:%s correlation=%s capture_ms=%s processing_ms=%s",
            challenge_id,
            company_id,
            user_id,
            overall,
            reason,
            len(observations),
            len(invalid),
            passive["result"],
            passive["reason"],
            passive.get("aggregate"),
            headpose["result"],
            headpose["reason"],
            headpose.get("steps"),
            illumination["result"],
            illumination["reason"],
            illumination.get("correlation"),
            result["durationMs"],
            result["processingMs"],
        )
        return result

    def check_frame(
        self,
        challenge_id: str,
        nonce: str,
        company_id: int,
        user_id: int,
        context_id: str,
        transaction_id: str,
        image: bytes,
        enrollment: bool,
        quality_policy: dict | None = None,
    ) -> dict[str, Any]:
        challenge = self.store.get(challenge_id)
        self._validate_binding(
            challenge, nonce, company_id, user_id, context_id, transaction_id, enrollment,
        )
        try:
            quality, _chromaticity = self.matcher.analyse_liveness_frame(image)
        except ValueError:
            return {"ready": False, "reason": "invalid_image"}
        reason = self._quality_reason(quality, enrollment, quality_policy)
        return {"ready": reason is None, "reason": reason or "ok"}

    def check_pose_frame(
        self,
        challenge_id: str,
        nonce: str,
        company_id: int,
        user_id: int,
        context_id: str,
        transaction_id: str,
        step_index: int,
        image: bytes,
        enrollment: bool,
        quality_policy: dict | None = None,
    ) -> dict[str, Any]:
        challenge = self.store.get(challenge_id)
        self._validate_binding(
            challenge, nonce, company_id, user_id, context_id, transaction_id, enrollment,
        )
        if not challenge.get("adaptiveHeadPose") or not hasattr(self.store, "get_pose_progress"):
            raise ValueError("adaptive_headpose_unavailable")
        steps = challenge.get("movementSteps") or []
        progress = self.store.get_pose_progress(challenge_id)
        expected = int(progress.get("expectedStep", 0))
        if step_index != expected or step_index < 0 or step_index >= len(steps):
            raise ValueError("headpose_step_out_of_order")
        try:
            quality = self.matcher.analyse_headpose_frame(image)
        except ValueError:
            return {"reached": False, "reason": "invalid_image", "stepIndex": step_index}
        reason = self._quality_reason(quality, enrollment, quality_policy)
        if reason:
            return {"reached": False, "reason": reason, "stepIndex": step_index}
        yaw = quality.headpose_yaw_degrees
        if yaw is None:
            return {"reached": False, "reason": "headpose_unavailable", "stepIndex": step_index}

        action = str(steps[step_index]["action"])
        sign = 1 if int(self.settings.identity_headpose_left_sign) >= 0 else -1
        directed = float(yaw) * sign if action == "left" else -float(yaw) * sign
        matches = abs(float(yaw)) <= float(self.settings.identity_headpose_center_degrees) if action == "center" else (
            directed >= float(self.settings.identity_headpose_turn_degrees)
        )
        first = progress.get("firstDirectedYaw")
        if first is None:
            baseline_yaw = float(progress.get("lastCenterYaw", 0.0))
            first = (
                baseline_yaw * sign if action == "left"
                else -baseline_yaw * sign if action == "right"
                else 0.0
            )
        progressive = action == "center" or (
            directed - float(first) >= float(self.settings.identity_headpose_min_progress_degrees)
        )
        hold_frames = int(progress.get("holdFrames", 0)) + 1 if matches and progressive else 0
        reached = hold_frames >= max(1, int(self.settings.identity_headpose_min_hold_frames))
        if reached:
            progress.setdefault("steps", []).append({
                "step": step_index,
                "action": action,
                "result": "pass",
                "observedYaw": round(float(yaw), 2),
            })
            progress["expectedStep"] = step_index + 1
            progress["holdFrames"] = 0
            progress["firstDirectedYaw"] = None
            if action == "center":
                progress["lastCenterYaw"] = float(yaw)
        else:
            progress["holdFrames"] = hold_frames
            progress["firstDirectedYaw"] = first
        self.store.save_pose_progress(challenge_id, progress)
        logger.info(
            "headpose_progress challenge=%s company=%s user=%s step=%s action=%s yaw=%.2f hold=%s reached=%s",
            challenge_id, company_id, user_id, step_index, action, float(yaw), hold_frames, reached,
        )
        return {
            "reached": reached,
            "reason": "pose_confirmed" if reached else "keep_moving",
            "stepIndex": step_index,
            "nextStepIndex": int(progress["expectedStep"]),
        }

    def _required_components(self, enrollment: bool) -> dict[str, bool]:
        active = bool(
            self.settings.identity_active_liveness_enabled
            or self.settings.identity_require_active_liveness
            or (enrollment and self.settings.identity_require_enrollment_liveness)
        )
        return {
            "passivePad": bool(
                self.settings.identity_temporal_passive_pad_enabled
                or self.settings.identity_require_passive_antispoof
            ),
            "headPose": bool(active and (
                self.settings.identity_headpose_challenge_enabled
                or self.settings.identity_require_headpose_liveness
            )),
            "illumination": bool(active and self.settings.identity_illumination_challenge_enabled),
        }

    def _movement_steps(self, actions: list[str]) -> list[dict[str, Any]]:
        cursor = 0
        steps = []
        for index, action in enumerate(["center", *actions]):
            duration = 1000 if index == 0 else (1800 if action != "center" else 1200)
            steps.append({"step": index, "action": action, "startMs": cursor, "endMs": cursor + duration})
            cursor += duration
        return steps

    def _illumination_steps(self) -> list[dict[str, Any]]:
        names = ["red", "green", "blue"]
        secrets.SystemRandom().shuffle(names)
        names.insert(0, "neutral")
        phase_ms = max(500, int(self.settings.identity_illumination_phase_ms))
        return [
            {
                "step": index,
                "colour": name,
                "hex": self.COLOURS[name]["hex"],
                "startMs": index * phase_ms,
                "endMs": (index + 1) * phase_ms,
            }
            for index, name in enumerate(names)
        ]

    @staticmethod
    def _public_challenge(challenge: dict[str, Any]) -> dict[str, Any]:
        return {
            "required": any(challenge["required"].values()),
            "challengeId": challenge["challengeId"],
            "nonce": challenge["nonce"],
            "issuedAtMs": challenge["issuedAtMs"],
            "expiresAtMs": challenge["expiresAtMs"],
            "durationMs": challenge["durationMs"],
            "components": challenge["required"],
            "movementSteps": challenge["movementSteps"],
            "illuminationSteps": challenge["illuminationSteps"],
            "adaptiveHeadPose": bool(challenge.get("adaptiveHeadPose")),
            "poseStepTimeoutMs": int(challenge.get("poseStepTimeoutMs", 3000)),
            "minimumCaptureMs": int(challenge.get("minimumCaptureMs", 0)),
        }

    @staticmethod
    def _validate_binding(
        challenge: dict[str, Any],
        nonce: str,
        company_id: int,
        user_id: int,
        context_id: str,
        transaction_id: str,
        enrollment: bool,
    ) -> None:
        now = int(time.time() * 1000)
        expected = (
            secrets.compare_digest(str(challenge.get("nonce", "")), str(nonce))
            and int(challenge.get("companyId", -1)) == int(company_id)
            and int(challenge.get("userId", -1)) == int(user_id)
            and str(challenge.get("contextId", "")) == str(context_id)
            and str(challenge.get("transactionId", "")) == str(transaction_id)
            and bool(challenge.get("enrollment")) is bool(enrollment)
        )
        if not expected:
            raise ValueError("challenge_binding_mismatch")
        if now > int(challenge.get("expiresAtMs", 0)):
            raise ValueError("challenge_expired_or_replayed")

    @staticmethod
    def _validate_evidence_timing(challenge: dict[str, Any], evidence: list[dict[str, Any]]) -> None:
        if not evidence:
            raise ValueError("liveness_evidence_missing")
        duration = int(challenge.get("durationMs", 0))
        issued = int(challenge.get("issuedAtMs", 0))
        previous = -1
        for item in evidence:
            elapsed = int(item.get("elapsedMs", -1))
            captured = int(item.get("capturedAtMs", 0))
            if elapsed <= previous or elapsed < 0 or elapsed > duration + 750:
                raise ValueError("invalid_challenge_timestamps")
            if captured and captured < issued - 1500:
                raise ValueError("evidence_predates_challenge")
            if captured and abs((captured - issued) - elapsed) > 2500:
                raise ValueError("challenge_clock_mismatch")
            previous = elapsed

    def _observations(
        self,
        evidence: list[dict[str, Any]],
        enrollment: bool,
        quality_policy: dict | None,
        include_headpose: bool = True,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        valid, invalid = [], []
        previous_elapsed = -1
        for item in evidence:
            elapsed = int(item.get("elapsedMs", -1))
            if elapsed < 0 or elapsed <= previous_elapsed:
                invalid.append({"reason": "invalid_timestamps"})
                continue
            previous_elapsed = elapsed
            try:
                quality, chromaticity = self.matcher.analyse_liveness_frame(
                    item["bytes"],
                    include_headpose=include_headpose,
                )
            except (ValueError, KeyError):
                invalid.append({"reason": "invalid_image"})
                continue
            reason = self._quality_reason(quality, enrollment, quality_policy)
            if reason:
                invalid.append({"reason": reason})
                continue
            valid.append({
                "elapsedMs": elapsed,
                "capturedAtMs": int(item.get("capturedAtMs", 0)),
                "quality": quality,
                "chromaticity": chromaticity,
            })
        return valid, invalid

    @staticmethod
    def _adaptive_headpose_result(
        challenge: dict[str, Any],
        progress: dict[str, Any] | None,
    ) -> dict[str, Any]:
        required = bool(challenge["required"]["headPose"])
        steps = challenge.get("movementSteps") or []
        completed = int((progress or {}).get("expectedStep", 0))
        if not required:
            return {"required": False, "result": "pass", "reason": "disabled", "steps": []}
        if completed >= len(steps):
            return {
                "required": True,
                "result": "pass",
                "reason": "headpose_complete",
                "steps": list((progress or {}).get("steps", [])),
            }
        return {
            "required": True,
            "result": "fail",
            "reason": "headpose_sequence_failed",
            "steps": list((progress or {}).get("steps", [])),
            "completedSteps": completed,
            "requiredSteps": len(steps),
        }

    def _quality_reason(self, quality, enrollment: bool, policy: dict | None) -> str | None:
        if quality.face_count < 1:
            return "no_face"
        if quality.face_count > 1:
            return "multiple_faces"
        prefix = "enrollment" if enrollment else ""
        brightness_key = "enrollmentMinBrightness" if enrollment else "minBrightness"
        blur_key = "enrollmentMinBlur" if enrollment else "minBlur"
        confidence_key = "enrollmentMinFaceConfidence" if enrollment else "minFaceConfidence"
        brightness = self._policy(policy, brightness_key, getattr(self.settings, f"identity_{prefix + '_' if prefix else ''}min_brightness"))
        blur = self._policy(policy, blur_key, getattr(self.settings, f"identity_{prefix + '_' if prefix else ''}min_blur"))
        confidence = self._policy(policy, confidence_key, getattr(self.settings, f"identity_{prefix + '_' if prefix else ''}min_face_confidence"))
        if quality.brightness < float(brightness):
            return "low_light"
        if quality.blur < float(blur):
            return "blurry"
        if quality.confidence is not None and quality.confidence < float(confidence):
            return "low_face_confidence"
        ratio = quality.face_width / max(1, quality.width)
        min_ratio = float(self.settings.identity_enrollment_min_face_width_ratio if enrollment else self.settings.identity_min_face_width_ratio)
        max_ratio = float(self.settings.identity_enrollment_max_face_width_ratio if enrollment else self.settings.identity_max_face_width_ratio)
        if ratio < min_ratio:
            return "face_too_far"
        if ratio > max_ratio:
            return "face_too_close"
        tolerance_x = float(
            self.settings.identity_enrollment_center_tolerance_x
            if enrollment else self.settings.identity_center_tolerance_x
        )
        tolerance_y = float(
            self.settings.identity_enrollment_center_tolerance_y
            if enrollment else self.settings.identity_center_tolerance_y
        )
        if quality.face_center_x is None or quality.face_center_y is None:
            return "face_not_framed"
        if abs(float(quality.face_center_x) - 0.5) > tolerance_x or abs(float(quality.face_center_y) - 0.5) > tolerance_y:
            return "face_not_centered"
        return None

    @staticmethod
    def _policy(policy: dict | None, key: str, fallback):
        return policy.get(key, fallback) if isinstance(policy, dict) else fallback

    def _passive_result(
        self,
        observations: list[dict[str, Any]],
        invalid: list[dict[str, Any]],
        required: bool,
        challenge_duration_ms: int,
    ) -> dict[str, Any]:
        if not required:
            return {"required": False, "result": "pass", "reason": "disabled", "validFrames": 0, "invalidFrames": len(invalid)}
        scores = [item["quality"].antispoof_scores for item in observations if item["quality"].antispoof_scores]
        minimum = max(2, int(self.settings.identity_passive_min_valid_frames))
        if len(scores) < minimum:
            return {"required": True, "result": "inconclusive", "reason": "passive_insufficient_frames", "validFrames": len(scores), "invalidFrames": len(invalid)}
        elapsed = [item["elapsedMs"] for item in observations if item["quality"].antispoof_scores]
        required_span = min(
            int(self.settings.identity_passive_capture_window_ms),
            challenge_duration_ms,
        ) * 0.65
        actual_span = max(elapsed) - min(elapsed)
        if actual_span < required_span:
            return {
                "required": True,
                "result": "inconclusive",
                "reason": "passive_window_too_short",
                "validFrames": len(scores),
                "invalidFrames": len(invalid),
                "captureSpanMs": actual_span,
            }
        medians = {name: float(median([float(score.get(name, 0.0)) for score in scores])) for name in ("live", "print", "replay")}
        live_threshold = float(self.settings.identity_antispoof_threshold)
        spoof_threshold = float(self.settings.identity_antispoof_spoof_threshold)
        if medians["live"] >= live_threshold:
            result, reason = "pass", "passive_live"
        elif medians["live"] <= spoof_threshold and max(medians["print"], medians["replay"]) >= (1.0 - spoof_threshold) / 2.0:
            result = "fail"
            reason = "print_attack" if medians["print"] >= medians["replay"] else "replay_attack"
        else:
            result, reason = "inconclusive", "passive_ambiguous"
        return {
            "required": True,
            "result": result,
            "reason": reason,
            "validFrames": len(scores),
            "invalidFrames": len(invalid),
            "captureSpanMs": actual_span,
            "aggregate": {name: round(value, 4) for name, value in medians.items()},
            "samples": [
                {"elapsedMs": item["elapsedMs"], **{name: round(float(score.get(name, 0.0)), 4) for name in ("live", "print", "replay")}}
                for item, score in zip(
                    [item for item in observations if item["quality"].antispoof_scores],
                    scores,
                )
            ],
        }

    def _headpose_result(self, observations: list[dict[str, Any]], challenge: dict[str, Any]) -> dict[str, Any]:
        required = bool(challenge["required"]["headPose"])
        if not required:
            return {"required": False, "result": "pass", "reason": "disabled", "steps": []}
        yaws = [(item["elapsedMs"], item["quality"].headpose_yaw_degrees) for item in observations]
        yaws = [(elapsed, float(yaw)) for elapsed, yaw in yaws if yaw is not None]
        if not yaws:
            return {"required": True, "result": "inconclusive", "reason": "headpose_unavailable", "steps": []}
        turn = float(self.settings.identity_headpose_turn_degrees)
        center = float(self.settings.identity_headpose_center_degrees)
        hold = max(1, int(self.settings.identity_headpose_min_hold_frames))
        sign = 1 if int(self.settings.identity_headpose_left_sign) >= 0 else -1
        steps = []
        initial = challenge["movementSteps"][0]
        initial_values = [yaw for elapsed, yaw in yaws if initial["startMs"] <= elapsed < initial["endMs"]]
        if len(initial_values) < hold:
            return {"required": True, "result": "inconclusive", "reason": "headpose_insufficient_frames", "steps": []}
        if sum(abs(yaw) <= center for yaw in initial_values) < hold:
            return {"required": True, "result": "fail", "reason": "movement_before_challenge", "steps": []}
        for step in challenge["movementSteps"]:
            values = [(elapsed, yaw) for elapsed, yaw in yaws if step["startMs"] <= elapsed < step["endMs"]]
            action = step["action"]
            if len(values) < hold:
                steps.append({
                    "step": step["step"],
                    "action": action,
                    "startMs": step["startMs"],
                    "endMs": step["endMs"],
                    "result": "inconclusive",
                    "reachedAtMs": None,
                })
                continue
            directed = [yaw * sign if action == "left" else -yaw * sign if action == "right" else -abs(yaw) for _elapsed, yaw in values]
            threshold = turn if action != "center" else -center
            matching = [(elapsed, value) for (elapsed, _yaw), value in zip(values, directed) if value >= threshold]
            reached = len(matching) >= hold
            progressive = True
            if action != "center":
                sequence = directed
                progressive = (max(sequence) - sequence[0]) >= float(self.settings.identity_headpose_min_progress_degrees)
            steps.append({
                "step": step["step"],
                "action": action,
                "startMs": step["startMs"],
                "endMs": step["endMs"],
                "result": "pass" if reached and progressive else "fail",
                "reachedAtMs": matching[0][0] if matching else None,
                "matchingFrames": len(matching),
                "observedMinYaw": round(min(yaw for _elapsed, yaw in values), 2),
                "observedMaxYaw": round(max(yaw for _elapsed, yaw in values), 2),
            })
        if any(step["result"] == "fail" for step in steps):
            return {"required": True, "result": "fail", "reason": "headpose_sequence_failed", "steps": steps}
        if any(step["result"] == "inconclusive" for step in steps):
            return {"required": True, "result": "inconclusive", "reason": "headpose_insufficient_frames", "steps": steps}
        return {"required": True, "result": "pass", "reason": "headpose_complete", "steps": steps}

    def _illumination_result(self, observations: list[dict[str, Any]], challenge: dict[str, Any]) -> dict[str, Any]:
        required = bool(challenge["required"]["illumination"])
        if not required:
            return {"required": False, "result": "pass", "reason": "disabled", "correlation": None}
        minimum = max(1, int(self.settings.identity_illumination_min_frames_per_phase))
        phase_values = []
        phase_results = []
        for phase in challenge["illuminationSteps"]:
            values = [item["chromaticity"] for item in observations if item["chromaticity"] is not None and phase["startMs"] <= item["elapsedMs"] < phase["endMs"]]
            if len(values) < minimum:
                return {"required": True, "result": "inconclusive", "reason": "illumination_insufficient_frames", "correlation": None}
            phase_values.append(np.median(np.asarray(values, dtype=np.float32), axis=0))
            phase_results.append({
                "step": phase["step"],
                "colour": phase["colour"],
                "startMs": phase["startMs"],
                "endMs": phase["endMs"],
                "validFrames": len(values),
            })
        baseline = phase_values[0]
        observed = np.concatenate([value - baseline for value in phase_values[1:]])
        neutral = np.asarray(self.COLOURS["neutral"]["rgb"], dtype=np.float32)
        neutral /= np.sum(neutral)
        expected_parts = []
        for phase in challenge["illuminationSteps"][1:]:
            colour = np.asarray(self.COLOURS[phase["colour"]]["rgb"], dtype=np.float32)
            colour /= np.sum(colour)
            expected_parts.append(colour - neutral)
        expected = np.concatenate(expected_parts)
        amplitude = float(np.max(np.abs(observed))) if observed.size else 0.0
        if amplitude < float(self.settings.identity_illumination_min_response):
            return {"required": True, "result": "inconclusive", "reason": "illumination_weak_response", "correlation": None, "response": round(amplitude, 5), "steps": phase_results}
        correlation = float(np.corrcoef(observed, expected)[0, 1]) if np.std(observed) > 1e-8 else 0.0
        if correlation >= float(self.settings.identity_illumination_pass_correlation):
            result, reason = "pass", "illumination_correlated"
        elif correlation <= float(self.settings.identity_illumination_fail_correlation):
            result, reason = "fail", "illumination_inconsistent"
        else:
            result, reason = "inconclusive", "illumination_ambiguous"
        return {"required": True, "result": result, "reason": reason, "correlation": round(correlation, 4), "response": round(amplitude, 5), "steps": phase_results}

    @staticmethod
    def _reason_counts(items: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            reason = str(item.get("reason", "unknown"))
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    @staticmethod
    def _quality_median(observations: list[dict[str, Any]], field: str) -> float | None:
        values = [
            float(value)
            for item in observations
            if (value := getattr(item["quality"], field, None)) is not None
        ]
        return round(float(median(values)), 4) if values else None
