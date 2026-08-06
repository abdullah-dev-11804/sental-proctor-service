from dataclasses import dataclass

import cv2
import numpy as np

from app.core.config import get_settings


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
    """Baseline face matcher for the first identity-verification slice.

    This intentionally uses local OpenCV primitives that work without model
    downloads. It is good enough for staging the API and access flow. Production
    should upgrade the internals to OpenCV YuNet + SFace ONNX models while
    keeping this service interface stable.
    """

    FACE_SIZE = (160, 160)

    def __init__(self) -> None:
        self.settings = get_settings()
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.profile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        if self.detector.empty():
            raise RuntimeError("OpenCV Haar face detector could not be loaded.")
        if self.profile_detector.empty():
            raise RuntimeError("OpenCV Haar profile-face detector could not be loaded.")

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
    ) -> dict:
        """Verifies identity from a burst of straight-face frames only."""
        center_bytes, _center_face, center_quality = self._best_frame(center_frames, prefer_frontal=True)
        match = self.verify(center_bytes, reference_bytes, pass_threshold)
        passed = match.status == "passed"
        result = self._challenge_response(
            match,
            passed,
            "ok" if passed else "identity_" + match.reason,
            0.0,
            None,
            None,
        )
        result["quality"].update({
            "center": center_quality.__dict__,
            "mode": "face_match_only",
        })
        return result

    def select_enrollment_reference(self, center_frames: list[bytes]) -> dict:
        """Selects and validates the best first-exam enrollment reference."""
        reference_bytes, _face, quality = self._best_frame(center_frames, prefer_frontal=True)
        retry_reason = self._enrollment_retry_reason(quality)
        if retry_reason:
            return {
                "ok": True,
                "result": retry_reason,
                "accessAllowed": False,
                "similarityScore": None,
                "referenceBytes": None,
                "referenceFaceCount": quality.face_count,
                "liveFaceCount": quality.face_count,
                "quality": {"reference": quality.__dict__, "mode": "enroll_reference"},
                "reason": retry_reason,
                "engine": "opencv-haar-quality-baseline-enrollment",
            }
        return {
            "ok": True,
            "result": "enrolled",
            "accessAllowed": True,
            "similarityScore": 1.0,
            "referenceBytes": reference_bytes,
            "referenceFaceCount": quality.face_count,
            "liveFaceCount": quality.face_count,
            "quality": {"reference": quality.__dict__, "mode": "enroll_reference"},
            "reason": "ok",
            "engine": "opencv-haar-quality-baseline-enrollment",
        }

    def verify_sequence_challenge(
        self,
        center_frames: list[bytes],
        left_frames: list[bytes],
        right_frames: list[bytes],
        reference_bytes: bytes,
        pass_threshold: float | None = None,
    ) -> dict:
        """Verifies identity plus left/right active-liveness from frame bursts."""
        center_bytes, center_face, center_quality = self._best_frame(center_frames, prefer_frontal=True)
        match = self.verify(center_bytes, reference_bytes, pass_threshold)
        if match.status != "passed":
            return self._challenge_response(match, False, "identity_" + match.reason, 0.0, None, None)

        retry_reason = self._retry_reason(center_quality)
        if retry_reason:
            return self._challenge_response(match, False, retry_reason, 0.0, None, None)

        left_samples = self._valid_frame_samples(left_frames)
        right_samples = self._valid_frame_samples(right_frames)
        if len(left_samples) < 2 or len(right_samples) < 2:
            return self._challenge_response(match, False, "side_face_missing", 0.0, None, None)
        if any(sample[1].face_count > 1 for sample in left_samples + right_samples):
            return self._challenge_response(match, False, "multiple_faces", 0.0, None, None)

        left_deltas = [1.0 - self._similarity(center_face, face) for face, _quality in left_samples]
        right_deltas = [1.0 - self._similarity(center_face, face) for face, _quality in right_samples]
        left_yaws = [self._estimate_yaw(quality) for _face, quality in left_samples]
        right_yaws = [self._estimate_yaw(quality) for _face, quality in right_samples]
        left_yaw = self._strongest_yaw(left_yaws)
        right_yaw = self._strongest_yaw(right_yaws)

        center_left_delta = max(left_deltas)
        center_right_delta = max(right_deltas)
        side_delta = max(
            1.0 - self._similarity(left_face, right_face)
            for left_face, _left_quality in left_samples
            for right_face, _right_quality in right_samples
        )
        movement_score = float(max(0.0, min(1.0, (center_left_delta + center_right_delta + side_delta) / 3.0)))

        profile_seen = any(abs(yaw or 0.0) >= 0.9 for yaw in left_yaws + right_yaws)
        directional_change = (
            left_yaw is not None
            and right_yaw is not None
            and abs(left_yaw - right_yaw) >= 0.35
        )
        enough_motion = center_left_delta >= 0.045 and center_right_delta >= 0.045 and side_delta >= 0.045
        liveness_passed = bool(profile_seen or directional_change or enough_motion)
        reason = "ok" if liveness_passed else "head_turn_not_detected"
        response = self._challenge_response(match, liveness_passed, reason, movement_score, left_yaw, right_yaw)
        response["quality"].update({
            "center": center_quality.__dict__,
            "leftFrames": [quality.__dict__ for _face, quality in left_samples],
            "rightFrames": [quality.__dict__ for _face, quality in right_samples],
            "centerLeftDelta": float(round(center_left_delta, 4)),
            "centerRightDelta": float(round(center_right_delta, 4)),
            "sideDelta": float(round(side_delta, 4)),
        })
        return response

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
            image = self._decode_image(frame)
            face, quality = self._extract_primary_face(image)
            if quality.face_count >= 1:
                samples.append((face, quality))
        return samples

    def _quality_score(self, quality: FaceQuality, prefer_frontal: bool) -> float:
        brightness_score = max(0.0, 1.0 - abs(quality.brightness - 115.0) / 115.0)
        blur_score = min(1.0, quality.blur / 180.0)
        face_score = 1.0 if quality.face_count == 1 else 0.15
        detection_score = 0.2 if prefer_frontal and quality.detection == "frontal" else 0.0
        return (brightness_score * 0.25) + (blur_score * 0.25) + (face_score * 0.4) + detection_score

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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frontal_faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        profile_faces = self.profile_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(50, 50),
        )
        flipped = cv2.flip(gray, 1)
        flipped_profile_faces = self.profile_detector.detectMultiScale(
            flipped,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(50, 50),
        )
        mirrored_profile_faces = [
            (gray.shape[1] - x - width, y, width, height)
            for (x, y, width, height) in flipped_profile_faces
        ]
        if len(frontal_faces) > 0:
            candidates = [("frontal", box) for box in frontal_faces]
        else:
            candidates = (
                [("profile_right", box) for box in profile_faces]
                + [("profile_left", box) for box in mirrored_profile_faces]
            )

        quality = FaceQuality(
            brightness=float(np.mean(gray)),
            blur=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            face_count=int(len(candidates)),
            width=int(image.shape[1]),
            height=int(image.shape[0]),
        )

        if len(candidates) == 0:
            return np.zeros(self.FACE_SIZE, dtype=np.uint8), quality

        detection, box = max(candidates, key=lambda candidate: int(candidate[1][2]) * int(candidate[1][3]))
        x, y, width, height = box
        quality = FaceQuality(
            brightness=quality.brightness,
            blur=quality.blur,
            face_count=quality.face_count,
            width=quality.width,
            height=quality.height,
            face_center_x=float((x + (width / 2)) / max(1, gray.shape[1])),
            face_center_y=float((y + (height / 2)) / max(1, gray.shape[0])),
            face_width=int(width),
            detection=detection,
        )
        margin_x = int(width * 0.18)
        margin_y = int(height * 0.22)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(gray.shape[1], x + width + margin_x)
        y2 = min(gray.shape[0], y + height + margin_y)
        crop = gray[y1:y2, x1:x2]
        normalised = cv2.resize(crop, self.FACE_SIZE, interpolation=cv2.INTER_AREA)
        normalised = cv2.equalizeHist(normalised)
        return normalised, quality

    def _estimate_yaw(self, quality: FaceQuality) -> float | None:
        if quality.detection == "profile_left":
            return -1.0
        if quality.detection == "profile_right":
            return 1.0
        if quality.face_center_x is None:
            return None
        return float(max(-1.0, min(1.0, (quality.face_center_x - 0.5) * 2.0)))

    def _retry_reason(self, live_quality: FaceQuality) -> str | None:
        if live_quality.face_count < 1:
            return "no_face"
        if live_quality.brightness < self.settings.identity_min_brightness:
            return "low_light"
        if live_quality.blur < self.settings.identity_min_blur:
            return "blurry"
        return None

    def _enrollment_retry_reason(self, quality: FaceQuality) -> str | None:
        retry_reason = self._retry_reason(quality)
        if retry_reason:
            return retry_reason
        if quality.face_count != 1:
            return "multiple_faces"
        if quality.detection != "frontal":
            return "face_not_frontal"
        if quality.face_center_x is None or quality.face_center_y is None:
            return "face_not_framed"
        face_ratio = quality.face_width / max(1, quality.width)
        if face_ratio < 0.16:
            return "face_too_far"
        if face_ratio > 0.62:
            return "face_too_close"
        if not (0.30 <= quality.face_center_x <= 0.70 and 0.22 <= quality.face_center_y <= 0.78):
            return "face_not_centered"
        return None

    def _similarity(self, live_face: np.ndarray, reference_face: np.ndarray) -> float:
        hist_score = self._histogram_similarity(live_face, reference_face)
        pixel_score = self._cosine_similarity(live_face, reference_face)
        edge_score = self._edge_similarity(live_face, reference_face)
        score = (hist_score * 0.35) + (pixel_score * 0.35) + (edge_score * 0.30)
        return float(max(0.0, min(1.0, score)))

    def _histogram_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        left_hist = cv2.calcHist([left], [0], None, [64], [0, 256])
        right_hist = cv2.calcHist([right], [0], None, [64], [0, 256])
        cv2.normalize(left_hist, left_hist)
        cv2.normalize(right_hist, right_hist)
        raw = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL)
        return float((raw + 1.0) / 2.0)

    def _cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        left_vector = left.astype(np.float32).reshape(-1)
        right_vector = right.astype(np.float32).reshape(-1)
        left_vector -= float(np.mean(left_vector))
        right_vector -= float(np.mean(right_vector))
        denom = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
        if denom <= 1e-6:
            return 0.0
        raw = float(np.dot(left_vector, right_vector) / denom)
        return (raw + 1.0) / 2.0

    def _edge_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        left_edges = cv2.Canny(left, 80, 160)
        right_edges = cv2.Canny(right, 80, 160)
        return self._cosine_similarity(left_edges, right_edges)

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
            engine="opencv-haar-quality-baseline",
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
