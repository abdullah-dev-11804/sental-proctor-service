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
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("OpenCV Haar face detector could not be loaded.")

    def verify(self, live_bytes: bytes, reference_bytes: bytes) -> FaceMatchResult:
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
        if score >= self.settings.identity_pass_threshold:
            return self._result("passed", "allow", True, score, "ok", live_quality, reference_quality)
        if score >= self.settings.identity_review_threshold:
            return self._result("needs_review", "review", False, score, "low_confidence", live_quality, reference_quality)
        return self._result("failed", "deny", False, score, "mismatch", live_quality, reference_quality)

    def _decode_image(self, content: bytes) -> np.ndarray:
        array = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("invalid_image")
        return image

    def _extract_primary_face(self, image: np.ndarray) -> tuple[np.ndarray, FaceQuality]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        quality = FaceQuality(
            brightness=float(np.mean(gray)),
            blur=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            face_count=int(len(faces)),
            width=int(image.shape[1]),
            height=int(image.shape[0]),
        )

        if len(faces) == 0:
            return np.zeros(self.FACE_SIZE, dtype=np.uint8), quality

        x, y, width, height = max(faces, key=lambda box: int(box[2]) * int(box[3]))
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

    def _retry_reason(self, live_quality: FaceQuality) -> str | None:
        if live_quality.face_count < 1:
            return "no_face"
        if live_quality.brightness < self.settings.identity_min_brightness:
            return "low_light"
        if live_quality.blur < self.settings.identity_min_blur:
            return "blurry"
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
