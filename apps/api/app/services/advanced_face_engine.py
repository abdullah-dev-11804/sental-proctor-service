from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class AdvancedFaceQuality:
    brightness: float
    blur: float
    face_count: int
    width: int
    height: int
    face_center_x: float | None = None
    face_center_y: float | None = None
    face_width: int = 0
    detection: str = "scrfd"
    confidence: float | None = None
    yaw: float | None = None
    antispoof_score: float | None = None
    antispoof_passed: bool | None = None
    headpose_yaw_degrees: float | None = None


class AdvancedFaceEngine:
    """SCRFD detector + AdaFace embedding engine.

    This engine is intentionally strict: if selected, required detector and
    recognizer models must exist locally. Optional anti-spoof/head-pose models
    can be enabled with settings once their ONNX files are installed.
    """

    ARCFACE_112_DST = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.detector = self._load_scrfd()
        self.recognizer = self._load_onnx(self._model_path(settings.identity_adaface_model), "AdaFace")
        self.antispoof = None
        self.headpose = None

        antispoof_model = str(settings.identity_antispoof_model).strip()
        if antispoof_model:
            self.antispoof = self._load_onnx(self._model_path(antispoof_model), "anti-spoof")
        elif settings.identity_require_passive_antispoof:
            raise RuntimeError("Passive anti-spoofing is required but IDENTITY_ANTISPOOF_MODEL is empty.")

        headpose_model = str(settings.identity_headpose_model).strip()
        if headpose_model:
            self.headpose = self._load_onnx(self._model_path(headpose_model), "6DRepNet head-pose")
        elif settings.identity_require_headpose_liveness:
            raise RuntimeError("Head-pose liveness is required but IDENTITY_HEADPOSE_MODEL is empty.")

    def _model_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.settings.identity_model_root / path

    def _load_scrfd(self) -> Any:
        detector_path = self._model_path(self.settings.identity_scrfd_model)
        if not detector_path.is_file():
            raise RuntimeError(f"SCRFD model is missing: {detector_path}")
        try:
            from insightface.model_zoo import get_model
        except ImportError as exc:
            raise RuntimeError("The scrfd_adaface engine requires the insightface package.") from exc

        detector = get_model(str(detector_path), providers=["CPUExecutionProvider"])
        detector.prepare(
            ctx_id=-1,
            input_size=(int(self.settings.identity_scrfd_input_size), int(self.settings.identity_scrfd_input_size)),
            det_thresh=float(self.settings.identity_min_face_confidence),
        )
        return detector

    def _load_onnx(self, model_path: Path, label: str) -> Any:
        if not model_path.is_file():
            raise RuntimeError(f"{label} model is missing: {model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("The scrfd_adaface engine requires the onnxruntime package.") from exc
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def extract(self, image: np.ndarray) -> tuple[np.ndarray, AdvancedFaceQuality]:
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        bboxes, keypoints = self.detector.detect(image, max_num=0, metric="default")
        face_count = 0 if bboxes is None else int(len(bboxes))
        base_quality = AdvancedFaceQuality(
            brightness=brightness,
            blur=blur,
            face_count=face_count,
            width=int(width),
            height=int(height),
        )
        if bboxes is None or len(bboxes) == 0:
            return np.zeros((1, 512), dtype=np.float32), base_quality

        index = int(np.argmax((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])))
        bbox = bboxes[index].astype(np.float32)
        kps = keypoints[index].astype(np.float32) if keypoints is not None else None

        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        confidence = float(bbox[4]) if len(bbox) >= 5 else None
        face_width = max(0.0, x2 - x1)
        yaw = self._landmark_yaw(kps)

        antispoof_score = self._antispoof_score(image, bbox)
        antispoof_passed = None
        if antispoof_score is not None:
            antispoof_passed = antispoof_score >= float(self.settings.identity_antispoof_threshold)

        headpose_yaw = self._headpose_yaw(image, bbox)
        if headpose_yaw is not None:
            yaw = float(max(-1.0, min(1.0, headpose_yaw / 45.0)))

        quality = AdvancedFaceQuality(
            brightness=brightness,
            blur=blur,
            face_count=face_count,
            width=int(width),
            height=int(height),
            face_center_x=float(((x1 + x2) / 2.0) / max(1, width)),
            face_center_y=float(((y1 + y2) / 2.0) / max(1, height)),
            face_width=int(face_width),
            detection="scrfd",
            confidence=confidence,
            yaw=yaw,
            antispoof_score=antispoof_score,
            antispoof_passed=antispoof_passed,
            headpose_yaw_degrees=headpose_yaw,
        )
        aligned = self._align_face(image, kps, bbox)
        embedding = self._adaface_embedding(aligned)
        return embedding, quality

    def _align_face(self, image: np.ndarray, keypoints: np.ndarray | None, bbox: np.ndarray) -> np.ndarray:
        size = int(self.settings.identity_adaface_input_size)
        if keypoints is not None and keypoints.shape[0] >= 5:
            dst = self.ARCFACE_112_DST.copy()
            if size != 112:
                dst *= size / 112.0
            transform, _inliers = cv2.estimateAffinePartial2D(keypoints[:5], dst, method=cv2.LMEDS)
            if transform is not None:
                return cv2.warpAffine(image, transform, (size, size), borderValue=0.0)

        x1, y1, x2, y2 = [int(round(value)) for value in bbox[:4]]
        margin = int(max(x2 - x1, y2 - y1) * 0.18)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.shape[1], x2 + margin)
        y2 = min(image.shape[0], y2 + margin)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((size, size, 3), dtype=np.uint8)
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    def _adaface_embedding(self, aligned_bgr: np.ndarray) -> np.ndarray:
        tensor = aligned_bgr.astype(np.float32)
        if str(self.settings.identity_adaface_color_order).strip().lower() == "rgb":
            tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
        tensor = (tensor / 255.0 - 0.5) / 0.5
        tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :].astype(np.float32)
        input_name = self.recognizer.get_inputs()[0].name
        output = self.recognizer.run(None, {input_name: tensor})[0]
        embedding = np.asarray(output[0], dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(embedding))
        if norm > 1e-6:
            embedding = embedding / norm
        return embedding

    def _antispoof_score(self, image: np.ndarray, bbox: np.ndarray) -> float | None:
        if self.antispoof is None:
            return None
        crop = self._crop_bbox(image, bbox, margin_ratio=0.35)
        input_tensor = self._generic_image_tensor(
            crop,
            int(self.settings.identity_antispoof_input_size),
            mean=0.5,
            std=0.5,
            bgr=True,
        )
        input_name = self.antispoof.get_inputs()[0].name
        output = np.asarray(self.antispoof.run(None, {input_name: input_tensor})[0]).reshape(-1)
        if output.size == 1:
            return float(1.0 / (1.0 + np.exp(-float(output[0]))))
        probabilities = self._softmax(output)
        index = int(self.settings.identity_antispoof_live_class_index)
        if index < 0 or index >= probabilities.size:
            index = int(np.argmax(probabilities))
        return float(probabilities[index])

    def _headpose_yaw(self, image: np.ndarray, bbox: np.ndarray) -> float | None:
        if self.headpose is None:
            return None
        crop = self._crop_bbox(image, bbox, margin_ratio=0.25)
        tensor = self._generic_image_tensor(crop, 224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], bgr=False)
        input_name = self.headpose.get_inputs()[0].name
        output = np.asarray(self.headpose.run(None, {input_name: tensor})[0]).reshape(-1)
        if output.size >= 3:
            return float(output[1])
        return None

    def _crop_bbox(self, image: np.ndarray, bbox: np.ndarray, margin_ratio: float) -> np.ndarray:
        x1, y1, x2, y2 = [int(round(value)) for value in bbox[:4]]
        margin = int(max(x2 - x1, y2 - y1) * margin_ratio)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.shape[1], x2 + margin)
        y2 = min(image.shape[0], y2 + margin)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((80, 80, 3), dtype=np.uint8)
        return crop

    def _generic_image_tensor(
        self,
        image: np.ndarray,
        size: int,
        mean: float | list[float],
        std: float | list[float],
        bgr: bool,
    ) -> np.ndarray:
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if not bgr:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        mean_array = np.asarray(mean if isinstance(mean, list) else [mean, mean, mean], dtype=np.float32)
        std_array = np.asarray(std if isinstance(std, list) else [std, std, std], dtype=np.float32)
        resized = (resized - mean_array) / std_array
        return np.transpose(resized, (2, 0, 1))[None, :, :, :].astype(np.float32)

    def _landmark_yaw(self, keypoints: np.ndarray | None) -> float | None:
        if keypoints is None or keypoints.shape[0] < 3:
            return None
        left_eye = keypoints[0]
        right_eye = keypoints[1]
        nose = keypoints[2]
        eye_distance = float(np.linalg.norm(right_eye - left_eye))
        if eye_distance <= 1e-6:
            return None
        eye_center = (left_eye + right_eye) / 2.0
        raw = float((nose[0] - eye_center[0]) / eye_distance)
        return float(max(-1.0, min(1.0, raw * 2.4)))

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        shifted = values.astype(np.float32) - float(np.max(values))
        exponentials = np.exp(shifted)
        return exponentials / max(1e-6, float(np.sum(exponentials)))
