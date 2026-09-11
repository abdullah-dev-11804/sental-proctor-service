import time
from dataclasses import dataclass
import cv2
import numpy as np

from app.core.config import get_settings
from app.services.advanced_face_engine import AdvancedFaceEngine


@dataclass(frozen=True)
class FaceQuality:
    brightness: float
    blur: float
    face_count: int
    width: int
    height: int
    face_center_x: float | None = None
    face_center_y: float | None = None
    face_width: int = 0
    detection: str = "none"
    confidence: float | None = None
    yaw: float | None = None
    antispoof_score: float | None = None
    antispoof_passed: bool | None = None
    antispoof_scores: dict[str, float] | None = None
    headpose_yaw_degrees: float | None = None


@dataclass(frozen=True)
class FaceMatchResult:
    status: str
    decision: str
    allowed: bool
    score: float
    reason: str
    live_quality: FaceQuality
    reference_quality: FaceQuality
    engine: str


class FaceMatcher:
    """SCRFD detection and AdaFace 1:1 verification engine."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.engine = self.settings.identity_engine.strip().lower()
        self.advanced_engine = None
        if self.engine not in ("scrfd_adaface", "production_face"):
            raise RuntimeError("IDENTITY_ENGINE must be scrfd_adaface.")
        self.engine = "scrfd_adaface"
        self._load_advanced_engine()

    def _load_advanced_engine(self) -> None:
        self.advanced_engine = AdvancedFaceEngine(self.settings)

    def verify(self, live_bytes: bytes, reference_bytes: bytes, pass_threshold: float | None = None) -> FaceMatchResult:
        live_image = self._decode_image(live_bytes)
        reference_image = self._decode_image(reference_bytes)

        live_face, live_quality = self._extract_primary_face(live_image)
        reference_face, reference_quality = self._extract_primary_face(reference_image)

        retry_reason = self._retry_reason(live_quality)
        if retry_reason:
            return self._result("needs_retry", "retry", False, 0.0, retry_reason, live_quality, reference_quality)

        if live_quality.face_count > 1:
            return self._result("failed", "deny", False, 0.0, "multiple_faces", live_quality, reference_quality)
        if reference_quality.face_count < 1:
            return self._result("needs_review", "review", False, 0.0, "reference_no_face", live_quality, reference_quality)

        score = self._similarity(live_face, reference_face)
        threshold = pass_threshold if pass_threshold is not None else self.settings.identity_pass_threshold
        if score >= threshold:
            return self._result("passed", "allow", True, score, "ok", live_quality, reference_quality)
        if score >= self.settings.identity_review_threshold:
            return self._result("needs_review", "review", False, score, "low_confidence", live_quality, reference_quality)
        return self._result("failed", "deny", False, score, "mismatch", live_quality, reference_quality)

    def verify_challenge(
        self,
        center_bytes: bytes,
        left_bytes: bytes,
        right_bytes: bytes,
        reference_bytes: bytes,
        pass_threshold: float | None = None,
    ) -> dict:
        """Backwards-compatible one-frame challenge wrapper."""
        return self.verify_sequence_challenge(
            [center_bytes],
            [left_bytes],
            [right_bytes],
            reference_bytes,
            pass_threshold,
        )

    def verify_center_sequence(
        self,
        center_frames: list[bytes],
        reference_bytes: bytes,
        pass_threshold: float | None = None,
        quality_policy: dict | None = None,
    ) -> dict:
        """Verifies identity from a burst of straight-face frames only."""
        template = self._template_from_reference_bytes(reference_bytes)
        result = self.verify_against_template(center_frames, template, pass_threshold, quality_policy)
        result["quality"].update({
            "mode": "face_match_only",
        })
        return result

    def select_enrollment_reference(
        self,
        center_frames: list[bytes],
        left_frames: list[bytes] | None = None,
        right_frames: list[bytes] | None = None,
        quality_policy: dict | None = None,
    ) -> dict:
        """Builds a reusable enrollment template from multiple good frames."""
        samples = self._collect_embedding_samples(
            center_frames,
            retry_reason_fn=lambda quality: self._enrollment_retry_reason(quality, quality_policy),
            prefer_frontal=True,
        )
        min_frames = int(self._policy_value(quality_policy, "minEnrollmentFrames", self.settings.identity_min_enrollment_frames))
        if len(samples) < min_frames:
            reason = self._dominant_rejection_reason(samples, "not_enough_good_frames")
            return self._enrollment_failure(
                reason,
                samples,
                reference_quality=samples[0]["quality"] if samples else None,
                quality_policy=quality_policy,
            )

        consistency = self._embedding_consistency([sample["embedding"] for sample in samples])
        min_consistency = float(self._policy_value(
            quality_policy,
            "minTemplateConsistency",
            self.settings.identity_min_template_consistency,
        ))
        if consistency < min_consistency:
            return self._enrollment_failure(
                "unstable_reference_capture",
                samples,
                template_consistency=consistency,
                quality_policy=quality_policy,
            )

        template = self._build_template(samples, consistency)
        best_sample = max(samples, key=lambda sample: sample["score"])
        return {
            "ok": True,
            "result": "enrolled",
            "accessAllowed": True,
            "similarityScore": 1.0,
            "bestReferenceBytes": best_sample["bytes"],
            "template": template,
            "referenceFaceCount": template["quality"]["validFrameCount"],
            "liveFaceCount": template["quality"]["validFrameCount"],
            "livenessPassed": True,
            "movementScore": 0.0,
            "quality": {
                "reference": best_sample["quality"].__dict__,
                "mode": "enroll_template",
                "template": template["quality"],
            },
            "reason": "ok",
            "engine": self.engine + "-enrollment",
        }

    def verify_sequence_challenge(
        self,
        center_frames: list[bytes],
        left_frames: list[bytes],
        right_frames: list[bytes],
        reference_bytes: bytes,
        pass_threshold: float | None = None,
        quality_policy: dict | None = None,
    ) -> dict:
        """Verifies identity against the stored template from multiple frames."""
        template = self._template_from_reference_bytes(reference_bytes)
        response = self.verify_against_template(center_frames, template, pass_threshold, quality_policy)
        if self.settings.identity_require_active_liveness and left_frames and right_frames:
            center_frame, center_face, center_quality = self._best_frame(center_frames, prefer_frontal=True)
            match = self.verify(center_frame, reference_bytes, pass_threshold)
            if match.status == "passed":
                liveness = self._active_liveness(center_face, center_quality, left_frames, right_frames)
                response["livenessPassed"] = bool(liveness["passed"])
                response["movementScore"] = float(liveness["movementScore"])
                response["result"] = "matched" if liveness["passed"] else str(liveness["reason"])
                response["identityStatus"] = "passed" if liveness["passed"] else "failed"
                response["accessDecision"] = "allow" if liveness["passed"] else "retry"
                response["accessAllowed"] = bool(liveness["passed"])
                response["quality"]["liveness"] = liveness["quality"]
                response["leftYaw"] = liveness["leftYaw"]
                response["rightYaw"] = liveness["rightYaw"]
        return response

    def verify_against_template(
        self,
        center_frames: list[bytes],
        template: dict,
        pass_threshold: float | None = None,
        quality_policy: dict | None = None,
    ) -> dict:
        if not isinstance(template, dict):
            template = {}

        samples = self._collect_embedding_samples(
            center_frames,
            retry_reason_fn=lambda quality: self._verification_retry_reason(quality, quality_policy),
            prefer_frontal=True,
        )
        min_live_frames = int(self._policy_value(quality_policy, "minLiveFrames", self.settings.identity_min_live_frames))
        if len(samples) < min_live_frames:
            return self._template_verification_failure(
                self._dominant_rejection_reason(samples, "not_enough_good_live_frames"),
                samples,
                template,
                pass_threshold,
                quality_policy,
            )

        live_embeddings = [sample["embedding"] for sample in samples]
        reference_embeddings = self._template_embeddings(template)
        if not reference_embeddings:
            return self._template_verification_failure(
                "template_missing_embeddings",
                samples,
                template,
                pass_threshold,
                quality_policy,
            )

        scores: list[float] = []
        mean_embedding = self._template_mean_embedding(template)
        if mean_embedding is not None:
            scores.extend(self._cosine_scores(live_embeddings, [mean_embedding]))
        scores.extend(self._cosine_scores(live_embeddings, reference_embeddings))

        final_score = self._median_top_scores(scores, 3)
        threshold = pass_threshold if pass_threshold is not None else self.settings.identity_pass_threshold
        review_threshold = float(self.settings.identity_review_threshold)
        if final_score >= threshold:
            status = "passed"
            decision = "allow"
            allowed = True
            reason = "ok"
        elif final_score >= review_threshold:
            status = "needs_review"
            decision = "review"
            allowed = False
            reason = "low_confidence"
        else:
            status = "failed"
            decision = "deny"
            allowed = False
            reason = "mismatch"

        result = {
            "ok": True,
            "result": "matched" if status == "passed" else reason,
            "identityStatus": status,
            "accessDecision": decision,
            "accessAllowed": allowed,
            "similarityScore": float(round(final_score, 4)),
            "threshold": float(threshold),
            "reviewThreshold": review_threshold,
            "livenessPassed": True,
            "movementScore": 0.0,
            "referenceFaceCount": len(reference_embeddings),
            "liveFaceCount": len(samples),
            "quality": {
                "live": [sample["quality"].__dict__ for sample in samples],
                "reference": template.get("quality", {}),
                "template": {
                    "version": template.get("version"),
                    "templateConsistency": template.get("quality", {}).get("templateConsistency"),
                    "validFrameCount": template.get("quality", {}).get("validFrameCount"),
                },
            },
            "engine": self.engine + "-challenge",
            "reason": reason,
        }
        if status != "passed":
            result["debug"] = {
                "scores": [float(round(score, 4)) for score in scores],
                "adafaceInputShape": self._runtime_shape("adaface_input_shape"),
                "adafaceOutputShape": self._runtime_shape("adaface_output_shape"),
                "colorOrder": self.settings.identity_adaface_color_order,
            }
        return result

    def _collect_embedding_samples(
        self,
        frames: list[bytes],
        retry_reason_fn,
        prefer_frontal: bool,
    ) -> list[dict]:
        samples: list[dict] = []
        self._last_rejection_reasons = []
        self._last_rejected_qualities = []
        for frame in frames[: max(1, int(self.settings.identity_max_enrollment_frames))]:
            try:
                image = self._decode_image(frame)
                embedding, quality = self._extract_primary_face(image)
            except (ValueError, cv2.error):
                self._last_rejection_reasons.append("invalid_image")
                self._last_rejected_qualities.append({"reason": "invalid_image"})
                continue
            if quality.face_count < 1:
                self._last_rejection_reasons.append("no_face")
                self._last_rejected_qualities.append({"reason": "no_face", **quality.__dict__})
                continue
            retry_reason = retry_reason_fn(quality)
            if retry_reason:
                self._last_rejection_reasons.append(retry_reason)
                self._last_rejected_qualities.append({"reason": retry_reason, **quality.__dict__})
                continue
            sample = {
                "bytes": frame,
                "embedding": self._normalize_embedding(embedding),
                "quality": quality,
                "score": self._quality_score(quality, prefer_frontal),
            }
            if sample["embedding"] is not None:
                samples.append(sample)
        return samples

    def _build_template(self, samples: list[dict], consistency: float) -> dict:
        embeddings = [sample["embedding"] for sample in samples if sample.get("embedding") is not None]
        mean_embedding = self._normalize_embedding(np.mean(np.stack(embeddings, axis=0), axis=0)) if embeddings else None
        best_sample = max(samples, key=lambda sample: sample["score"])
        quality_summary = {
            "validFrameCount": len(samples),
            "templateConsistency": float(round(consistency, 4)),
            "bestFrameScore": float(round(best_sample["score"], 4)),
            "bestFrameFaceCount": int(best_sample["quality"].face_count),
            "bestFrameBrightness": float(best_sample["quality"].brightness),
            "bestFrameBlur": float(best_sample["quality"].blur),
            "engine": self.engine,
        }
        return {
            "version": 1,
            "engine": self.engine,
            "companyId": None,
            "userId": None,
            "createdAt": int(time.time()),
            "passThreshold": float(self.settings.identity_pass_threshold),
            "reviewThreshold": float(self.settings.identity_review_threshold),
            "embeddings": [embedding.flatten().tolist() for embedding in embeddings],
            "meanEmbedding": mean_embedding.flatten().tolist() if mean_embedding is not None else [],
            "bestReferenceImageKey": None,
            "quality": quality_summary,
        }

    def _template_from_reference_bytes(self, reference_bytes: bytes) -> dict:
        embedding, quality = self._extract_primary_face(self._decode_image(reference_bytes))
        if quality.face_count < 1:
            raise ValueError("reference_no_face")
        normalized = self._normalize_embedding(embedding)
        if normalized is None:
            raise ValueError("reference_no_face")
        return {
            "version": 1,
            "engine": self.engine,
            "companyId": None,
            "userId": None,
            "createdAt": 0,
            "passThreshold": float(self.settings.identity_pass_threshold),
            "reviewThreshold": float(self.settings.identity_review_threshold),
            "embeddings": [normalized.flatten().tolist()],
            "meanEmbedding": normalized.flatten().tolist(),
            "bestReferenceImageKey": None,
            "quality": {
                "validFrameCount": 1,
                "templateConsistency": 1.0,
                "bestFrameScore": self._quality_score(quality, True),
                "bestFrameFaceCount": int(quality.face_count),
                "bestFrameBrightness": float(quality.brightness),
                "bestFrameBlur": float(quality.blur),
                "engine": self.engine,
            },
        }

    def _template_embeddings(self, template: dict) -> list[np.ndarray]:
        if not isinstance(template, dict):
            return []
        embeddings = []
        raw_embeddings = template.get("embeddings", []) or []
        if not isinstance(raw_embeddings, list):
            return []
        for item in raw_embeddings:
            try:
                normalized = self._normalize_embedding(np.asarray(item, dtype=np.float32))
            except (TypeError, ValueError):
                continue
            if normalized is not None:
                embeddings.append(normalized)
        return embeddings

    def _template_mean_embedding(self, template: dict) -> np.ndarray | None:
        if not isinstance(template, dict):
            return None
        raw_embedding = template.get("meanEmbedding")
        if raw_embedding is None:
            return None
        try:
            return self._normalize_embedding(np.asarray(raw_embedding, dtype=np.float32))
        except (TypeError, ValueError):
            return None

    def _normalize_embedding(self, embedding: np.ndarray | list[float] | None) -> np.ndarray | None:
        if embedding is None:
            return None
        try:
            vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        except (TypeError, ValueError):
            return None
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6:
            return None
        return vector / norm

    def _cosine_scores(self, left_embeddings: list[np.ndarray], right_embeddings: list[np.ndarray]) -> list[float]:
        scores: list[float] = []
        for left in left_embeddings:
            left_vector = left.reshape(-1)
            for right in right_embeddings:
                right_vector = right.reshape(-1)
                denom = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
                if denom <= 1e-6:
                    continue
                scores.append(float(max(0.0, min(1.0, np.dot(left_vector, right_vector) / denom))))
        return scores

    def _median_top_scores(self, scores: list[float], top_n: int = 3) -> float:
        if not scores:
            return 0.0
        ordered = sorted(scores, reverse=True)[:max(1, top_n)]
        return float(np.median(np.asarray(ordered, dtype=np.float32)))

    def _embedding_consistency(self, embeddings: list[np.ndarray]) -> float:
        if len(embeddings) < 2:
            return 1.0
        scores = []
        for index, left in enumerate(embeddings):
            for right in embeddings[index + 1:]:
                scores.extend(self._cosine_scores([left], [right]))
        return float(np.median(np.asarray(scores, dtype=np.float32))) if scores else 0.0

    def _enrollment_failure(
        self,
        reason: str,
        samples: list[dict],
        reference_quality: FaceQuality | None = None,
        template_consistency: float | None = None,
        quality_policy: dict | None = None,
    ) -> dict:
        quality = {
            "reference": (reference_quality.__dict__ if reference_quality is not None else {}),
            "mode": "enroll_template",
            "validFrames": [sample["quality"].__dict__ for sample in samples],
            "rejectedFrames": list(getattr(self, "_last_rejected_qualities", [])),
            "rejectionSummary": self._rejection_summary(),
            "requirements": self._quality_requirements(quality_policy, enrollment=True),
            "template": {
                "templateConsistency": template_consistency,
                "validFrameCount": len(samples),
            },
        }
        return {
            "ok": True,
            "result": reason,
            "identityStatus": "needs_retry",
            "accessDecision": "retry",
            "accessAllowed": False,
            "similarityScore": None,
            "referenceBytes": None,
            "bestReferenceBytes": None,
            "template": None,
            "referenceFaceCount": len(samples),
            "liveFaceCount": len(samples),
            "livenessPassed": False,
            "movementScore": 0.0,
            "quality": quality,
            "reason": reason,
            "engine": self.engine + "-enrollment",
        }

    def _template_verification_failure(
        self,
        reason: str,
        samples: list[dict],
        template: dict,
        pass_threshold: float | None,
        quality_policy: dict | None = None,
    ) -> dict:
        threshold = pass_threshold if pass_threshold is not None else self.settings.identity_pass_threshold
        review_threshold = float(self.settings.identity_review_threshold)
        template_quality = template.get("quality", {}) if isinstance(template, dict) else {}
        template_embeddings = template.get("embeddings", []) if isinstance(template, dict) else []
        if not isinstance(template_embeddings, list):
            template_embeddings = []
        retry_reasons = {
            "not_enough_good_live_frames",
            "not_enough_good_frames",
            "unstable_reference_capture",
            "template_missing_embeddings",
            "invalid_image",
            "no_face",
            "low_light",
            "blurry",
            "multiple_faces",
            "low_face_confidence",
            "face_not_framed",
            "face_not_centered",
            "face_too_far",
            "face_too_close",
        }
        status = "needs_retry" if reason in retry_reasons else "failed"
        return {
            "ok": True,
            "result": reason,
            "identityStatus": status,
            "accessDecision": "retry" if status == "needs_retry" else "deny",
            "accessAllowed": False,
            "similarityScore": 0.0,
            "threshold": float(threshold),
            "reviewThreshold": review_threshold,
            "livenessPassed": True,
            "movementScore": 0.0,
            "referenceFaceCount": len(template_embeddings),
            "liveFaceCount": len(samples),
            "quality": {
                "live": [sample["quality"].__dict__ for sample in samples],
                "rejectedFrames": list(getattr(self, "_last_rejected_qualities", [])),
                "rejectionSummary": self._rejection_summary(),
                "requirements": self._quality_requirements(quality_policy, enrollment=False),
                "reference": template_quality,
                "template": template_quality,
            },
            "engine": self.engine + "-challenge",
            "reason": reason,
        }

    def _runtime_shape(self, key: str) -> list:
        if not self.advanced_engine:
            return []
        value = self.advanced_engine.runtime.get(key, [])
        if isinstance(value, (list, tuple)):
            result = []
            for item in value:
                if isinstance(item, (int, float)) and float(item).is_integer():
                    result.append(int(item))
                else:
                    result.append(item)
            return result
        return [value]

    def describe_runtime(self) -> dict:
        runtime = {
            "identity_engine_config": self.settings.identity_engine,
            "identity_engine_loaded": self.engine,
            "pass_threshold": float(self.settings.identity_pass_threshold),
            "review_threshold": float(self.settings.identity_review_threshold),
            "verification_quality": {
                "min_live_frames": int(self.settings.identity_min_live_frames),
                "min_brightness": float(self.settings.identity_min_brightness),
                "min_blur": float(self.settings.identity_min_blur),
                "min_face_confidence": float(self.settings.identity_min_face_confidence),
            },
            "enrollment_quality": {
                "min_enrollment_frames": int(self.settings.identity_min_enrollment_frames),
                "max_enrollment_frames": int(self.settings.identity_max_enrollment_frames),
                "min_template_consistency": float(self.settings.identity_min_template_consistency),
                "min_brightness": float(self.settings.identity_enrollment_min_brightness),
                "min_blur": float(self.settings.identity_enrollment_min_blur),
                "min_face_confidence": float(self.settings.identity_enrollment_min_face_confidence),
                "min_face_width_ratio": float(self.settings.identity_enrollment_min_face_width_ratio),
                "max_face_width_ratio": float(self.settings.identity_enrollment_max_face_width_ratio),
                "center_tolerance_x": float(self.settings.identity_enrollment_center_tolerance_x),
                "center_tolerance_y": float(self.settings.identity_enrollment_center_tolerance_y),
            },
            "liveness": {
                "temporal_passive_pad": bool(self.settings.identity_temporal_passive_pad_enabled),
                "passive_required": bool(self.settings.identity_require_passive_antispoof),
                "passive_min_valid_frames": int(self.settings.identity_passive_min_valid_frames),
                "active_enabled": bool(self.settings.identity_active_liveness_enabled),
                "headpose_challenge": bool(self.settings.identity_headpose_challenge_enabled),
                "illumination_challenge": bool(self.settings.identity_illumination_challenge_enabled),
                "challenge_timeout_ms": int(self.settings.identity_liveness_challenge_timeout_ms),
            },
        }
        if self.advanced_engine is not None:
            runtime.update(self.advanced_engine.describe_runtime())
        else:
            runtime.update({
                "scrfd_model_exists": False,
                "adaface_model_exists": False,
                "adaface_input_shape": [],
                "adaface_output_shape": [],
                "adaface_color_order": str(self.settings.identity_adaface_color_order).strip().lower(),
            })
        return runtime

    def _best_frame(self, frames: list[bytes], prefer_frontal: bool) -> tuple[bytes, np.ndarray, FaceQuality]:
        best: tuple[float, bytes, np.ndarray, FaceQuality] | None = None
        for frame in frames:
            image = self._decode_image(frame)
            face, quality = self._extract_primary_face(image)
            if quality.face_count < 1:
                continue
            score = self._quality_score(quality, prefer_frontal)
            if best is None or score > best[0]:
                best = (score, frame, face, quality)
        if best is None:
            image = self._decode_image(frames[0])
            face, quality = self._extract_primary_face(image)
            return frames[0], face, quality
        return best[1], best[2], best[3]

    def _valid_frame_samples(self, frames: list[bytes]) -> list[tuple[np.ndarray, FaceQuality]]:
        samples: list[tuple[np.ndarray, FaceQuality]] = []
        for frame in frames:
            try:
                image = self._decode_image(frame)
                face, quality = self._extract_primary_face(image)
            except (ValueError, cv2.error):
                continue
            if quality.face_count >= 1:
                samples.append((face, quality))
        return samples

    def _active_liveness(
        self,
        center_face: np.ndarray,
        center_quality: FaceQuality,
        left_frames: list[bytes],
        right_frames: list[bytes],
    ) -> dict:
        if not self.settings.identity_require_active_liveness:
            return {
                "passed": True,
                "reason": "ok",
                "movementScore": 1.0,
                "leftYaw": None,
                "rightYaw": None,
                "quality": {"mode": "disabled"},
            }

        min_samples = max(1, int(self.settings.identity_min_liveness_samples))
        left_samples = self._valid_frame_samples(left_frames)
        right_samples = self._valid_frame_samples(right_frames)
        if len(left_samples) < min_samples or len(right_samples) < min_samples:
            return self._liveness_result(False, "side_face_missing", 0.0, None, None, {
                "leftFrames": [quality.__dict__ for _face, quality in left_samples],
                "rightFrames": [quality.__dict__ for _face, quality in right_samples],
            })
        if any(quality.face_count != 1 for _face, quality in left_samples + right_samples):
            return self._liveness_result(False, "multiple_faces", 0.0, None, None, {
                "leftFrames": [quality.__dict__ for _face, quality in left_samples],
                "rightFrames": [quality.__dict__ for _face, quality in right_samples],
            })

        center_yaw = self._estimate_yaw(center_quality) or 0.0
        left_yaws = [yaw for _face, quality in left_samples if (yaw := self._estimate_yaw(quality)) is not None]
        right_yaws = [yaw for _face, quality in right_samples if (yaw := self._estimate_yaw(quality)) is not None]
        if not left_yaws or not right_yaws:
            return self._liveness_result(False, "head_turn_not_detected", 0.0, None, None, {
                "centerYaw": center_yaw,
                "leftFrames": [quality.__dict__ for _face, quality in left_samples],
                "rightFrames": [quality.__dict__ for _face, quality in right_samples],
            })

        left_yaw = float(np.median(left_yaws))
        right_yaw = float(np.median(right_yaws))
        yaw_delta = abs(left_yaw - right_yaw)
        center_delta = max(abs(left_yaw - center_yaw), abs(right_yaw - center_yaw))

        left_embedding_delta = max(1.0 - self._similarity(center_face, face) for face, _quality in left_samples)
        right_embedding_delta = max(1.0 - self._similarity(center_face, face) for face, _quality in right_samples)
        embedding_delta = max(left_embedding_delta, right_embedding_delta)

        min_yaw = float(self.settings.identity_min_liveness_yaw_delta)
        min_embedding = float(self.settings.identity_min_liveness_embedding_delta)
        yaw_score = min(1.0, yaw_delta / max(0.001, min_yaw))
        embedding_score = min(1.0, embedding_delta / max(0.001, min_embedding))
        movement_score = float(round((yaw_score * 0.70) + (embedding_score * 0.30), 4))
        passed = yaw_delta >= min_yaw and center_delta >= (min_yaw * 0.45) and embedding_delta >= min_embedding
        reason = "ok" if passed else "head_turn_not_detected"

        return self._liveness_result(passed, reason, movement_score, left_yaw, right_yaw, {
            "centerYaw": center_yaw,
            "yawDelta": float(round(yaw_delta, 4)),
            "centerDelta": float(round(center_delta, 4)),
            "embeddingDelta": float(round(embedding_delta, 4)),
            "leftEmbeddingDelta": float(round(left_embedding_delta, 4)),
            "rightEmbeddingDelta": float(round(right_embedding_delta, 4)),
            "leftFrames": [quality.__dict__ for _face, quality in left_samples],
            "rightFrames": [quality.__dict__ for _face, quality in right_samples],
        })

    def _liveness_result(
        self,
        passed: bool,
        reason: str,
        movement_score: float,
        left_yaw: float | None,
        right_yaw: float | None,
        quality: dict,
    ) -> dict:
        return {
            "passed": passed,
            "reason": reason,
            "movementScore": movement_score,
            "leftYaw": left_yaw,
            "rightYaw": right_yaw,
            "quality": quality,
        }

    def _quality_score(self, quality: FaceQuality, prefer_frontal: bool) -> float:
        brightness_score = max(0.0, 1.0 - abs(quality.brightness - 115.0) / 115.0)
        blur_score = min(1.0, quality.blur / 180.0)
        face_score = 1.0 if quality.face_count == 1 else 0.15
        detection_score = 0.2 if prefer_frontal and quality.detection == "scrfd" else 0.0
        confidence_score = quality.confidence if quality.confidence is not None else 1.0
        face_ratio = quality.face_width / max(1, quality.width)
        size_score = 1.0 if self.settings.identity_min_face_width_ratio <= face_ratio <= self.settings.identity_max_face_width_ratio else 0.3
        return (
            (brightness_score * 0.20)
            + (blur_score * 0.20)
            + (face_score * 0.30)
            + (confidence_score * 0.15)
            + (size_score * 0.10)
            + detection_score
        )

    def _strongest_yaw(self, yaws: list[float | None]) -> float | None:
        values = [yaw for yaw in yaws if yaw is not None]
        if not values:
            return None
        return max(values, key=lambda yaw: abs(yaw))

    def _decode_image(self, content: bytes) -> np.ndarray:
        array = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("invalid_image")
        return image

    def _extract_primary_face(self, image: np.ndarray) -> tuple[np.ndarray, FaceQuality]:
        return self._extract_primary_face_advanced(image)

    def _extract_primary_face_advanced(self, image: np.ndarray) -> tuple[np.ndarray, FaceQuality]:
        if self.advanced_engine is None:
            raise RuntimeError("Advanced face engine is not loaded.")
        embedding, advanced_quality = self.advanced_engine.extract(image)
        return embedding, self._face_quality(advanced_quality)

    def analyse_frame(self, content: bytes, include_embedding: bool = False) -> tuple[np.ndarray | None, FaceQuality]:
        """Analyses one monitoring frame without recognition unless requested."""
        if self.advanced_engine is None:
            raise RuntimeError("Advanced face engine is not loaded.")
        image = self._decode_image(content)
        if include_embedding:
            embedding, quality = self.advanced_engine.extract(image)
            return embedding, self._face_quality(quality)
        return None, self._face_quality(self.advanced_engine.analyse(image))

    def analyse_liveness_frame(self, content: bytes) -> tuple[FaceQuality, list[float] | None]:
        """Runs only detection/PAD/head-pose/illumination measurements, never AdaFace."""
        if self.advanced_engine is None:
            raise RuntimeError("Advanced face engine is not loaded.")
        image = self._decode_image(content)
        quality, chromaticity = self.advanced_engine.analyse_liveness(image)
        return self._face_quality(quality), chromaticity

    @staticmethod
    def _face_quality(advanced_quality) -> FaceQuality:
        return FaceQuality(
            brightness=advanced_quality.brightness,
            blur=advanced_quality.blur,
            face_count=advanced_quality.face_count,
            width=advanced_quality.width,
            height=advanced_quality.height,
            face_center_x=advanced_quality.face_center_x,
            face_center_y=advanced_quality.face_center_y,
            face_width=advanced_quality.face_width,
            detection=advanced_quality.detection,
            confidence=advanced_quality.confidence,
            yaw=advanced_quality.yaw,
            antispoof_score=advanced_quality.antispoof_score,
            antispoof_passed=advanced_quality.antispoof_passed,
            antispoof_scores=advanced_quality.antispoof_scores,
            headpose_yaw_degrees=advanced_quality.headpose_yaw_degrees,
        )

    def _estimate_yaw(self, quality: FaceQuality) -> float | None:
        if quality.yaw is not None:
            return quality.yaw
        if quality.face_center_x is None:
            return None
        return float(max(-1.0, min(1.0, (quality.face_center_x - 0.5) * 2.0)))

    def _retry_reason(self, live_quality: FaceQuality) -> str | None:
        return self._quality_retry_reason(
            live_quality,
            min_brightness=float(self.settings.identity_min_brightness),
            min_blur=float(self.settings.identity_min_blur),
            min_face_confidence=float(self.settings.identity_min_face_confidence),
        )

    def _quality_retry_reason(
        self,
        quality: FaceQuality,
        min_brightness: float,
        min_blur: float,
        min_face_confidence: float,
        enforce_passive_pad: bool = True,
    ) -> str | None:
        if quality.face_count < 1:
            return "no_face"
        if quality.face_count > 1:
            return "multiple_faces"
        if quality.brightness < min_brightness:
            return "low_light"
        if quality.blur < min_blur:
            return "blurry"
        if quality.confidence is not None and quality.confidence < min_face_confidence:
            return "low_face_confidence"
        if self.settings.identity_require_passive_antispoof and enforce_passive_pad:
            if quality.antispoof_passed is None:
                return "antispoof_unavailable"
            if not quality.antispoof_passed:
                return "spoof_detected"
        return None

    def _enrollment_retry_reason(self, quality: FaceQuality, quality_policy: dict | None = None) -> str | None:
        retry_reason = self._quality_retry_reason(
            quality,
            min_brightness=float(self._policy_value(
                quality_policy,
                "enrollmentMinBrightness",
                self.settings.identity_enrollment_min_brightness,
            )),
            min_blur=float(self._policy_value(
                quality_policy,
                "enrollmentMinBlur",
                self.settings.identity_enrollment_min_blur,
            )),
            min_face_confidence=float(self._policy_value(
                quality_policy,
                "enrollmentMinFaceConfidence",
                self.settings.identity_enrollment_min_face_confidence,
            )),
            enforce_passive_pad=not bool(self._policy_value(quality_policy, "skipPassivePad", False)),
        )
        if retry_reason:
            return retry_reason
        if quality.face_count != 1:
            return "multiple_faces"
        if quality.face_center_x is None or quality.face_center_y is None:
            return "face_not_framed"
        face_ratio = quality.face_width / max(1, quality.width)
        min_width_ratio = float(self._policy_value(
            quality_policy,
            "enrollmentMinFaceWidthRatio",
            self.settings.identity_enrollment_min_face_width_ratio,
        ))
        max_width_ratio = float(self._policy_value(
            quality_policy,
            "enrollmentMaxFaceWidthRatio",
            self.settings.identity_enrollment_max_face_width_ratio,
        ))
        if face_ratio < min_width_ratio:
            return "face_too_far"
        if face_ratio > max_width_ratio:
            return "face_too_close"
        tolerance_x = float(self._policy_value(
            quality_policy,
            "enrollmentCenterToleranceX",
            self.settings.identity_enrollment_center_tolerance_x,
        ))
        tolerance_y = float(self._policy_value(
            quality_policy,
            "enrollmentCenterToleranceY",
            self.settings.identity_enrollment_center_tolerance_y,
        ))
        min_x = 0.5 - tolerance_x
        max_x = 0.5 + tolerance_x
        min_y = 0.5 - tolerance_y
        max_y = 0.5 + tolerance_y
        if not (min_x <= quality.face_center_x <= max_x and min_y <= quality.face_center_y <= max_y):
            return "face_not_centered"
        return None

    def _verification_retry_reason(self, quality: FaceQuality, quality_policy: dict | None = None) -> str | None:
        retry_reason = self._quality_retry_reason(
            quality,
            min_brightness=float(self._policy_value(quality_policy, "minBrightness", self.settings.identity_min_brightness)),
            min_blur=float(self._policy_value(quality_policy, "minBlur", self.settings.identity_min_blur)),
            min_face_confidence=float(self._policy_value(
                quality_policy,
                "minFaceConfidence",
                self.settings.identity_min_face_confidence,
            )),
            enforce_passive_pad=not bool(self._policy_value(quality_policy, "skipPassivePad", False)),
        )
        if retry_reason:
            return retry_reason
        if quality.face_count < 1:
            return "no_face"
        if quality.face_count > 1:
            return "multiple_faces"
        return None

    def _policy_value(self, quality_policy: dict | None, key: str, fallback):
        if not isinstance(quality_policy, dict) or key not in quality_policy:
            return fallback
        try:
            return type(fallback)(quality_policy[key])
        except (TypeError, ValueError):
            return fallback

    def _dominant_rejection_reason(self, samples: list[dict], fallback: str) -> str:
        reasons = list(getattr(self, "_last_rejection_reasons", []))
        if not reasons and samples:
            return fallback
        priority = [
            "no_face",
            "multiple_faces",
            "low_light",
            "blurry",
            "low_face_confidence",
            "face_too_far",
            "face_too_close",
            "face_not_centered",
            "face_not_framed",
            "invalid_image",
        ]
        for reason in priority:
            if reason in reasons:
                return reason
        return reasons[0] if reasons else fallback

    def _rejection_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for reason in getattr(self, "_last_rejection_reasons", []):
            summary[reason] = summary.get(reason, 0) + 1
        return summary

    def _quality_requirements(self, quality_policy: dict | None, enrollment: bool) -> dict[str, float]:
        if enrollment:
            return {
                "minBrightness": float(self._policy_value(
                    quality_policy, "enrollmentMinBrightness", self.settings.identity_enrollment_min_brightness,
                )),
                "minBlur": float(self._policy_value(
                    quality_policy, "enrollmentMinBlur", self.settings.identity_enrollment_min_blur,
                )),
                "minFaceConfidence": float(self._policy_value(
                    quality_policy,
                    "enrollmentMinFaceConfidence",
                    self.settings.identity_enrollment_min_face_confidence,
                )),
            }
        return {
            "minBrightness": float(self._policy_value(
                quality_policy, "minBrightness", self.settings.identity_min_brightness,
            )),
            "minBlur": float(self._policy_value(
                quality_policy, "minBlur", self.settings.identity_min_blur,
            )),
            "minFaceConfidence": float(self._policy_value(
                quality_policy, "minFaceConfidence", self.settings.identity_min_face_confidence,
            )),
        }

    def _similarity(self, live_face: np.ndarray, reference_face: np.ndarray) -> float:
        left_vector = live_face.astype(np.float32).reshape(-1)
        right_vector = reference_face.astype(np.float32).reshape(-1)
        denom = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
        if denom <= 1e-6:
            return 0.0
        return float(max(0.0, min(1.0, np.dot(left_vector, right_vector) / denom)))

    def _result(
        self,
        status: str,
        decision: str,
        allowed: bool,
        score: float,
        reason: str,
        live_quality: FaceQuality,
        reference_quality: FaceQuality,
    ) -> FaceMatchResult:
        return FaceMatchResult(
            status=status,
            decision=decision,
            allowed=allowed,
            score=float(round(score, 4)),
            reason=reason,
            live_quality=live_quality,
            reference_quality=reference_quality,
            engine=self.engine,
        )

    def _challenge_response(
        self,
        match: FaceMatchResult,
        liveness_passed: bool,
        reason: str,
        movement_score: float,
        left_yaw: float | None,
        right_yaw: float | None,
    ) -> dict:
        passed = match.status == "passed" and liveness_passed
        return {
            "ok": True,
            "result": "matched" if passed else reason,
            "identityStatus": "passed" if passed else "failed",
            "accessDecision": "allow" if passed else "retry",
            "accessAllowed": passed,
            "similarityScore": match.score,
            "livenessPassed": liveness_passed,
            "movementScore": float(round(movement_score, 4)),
            "referenceFaceCount": match.reference_quality.face_count,
            "liveFaceCount": match.live_quality.face_count,
            "leftYaw": left_yaw,
            "rightYaw": right_yaw,
            "quality": {
                "live": match.live_quality.__dict__,
                "reference": match.reference_quality.__dict__,
            },
            "engine": match.engine + "-challenge",
        }
