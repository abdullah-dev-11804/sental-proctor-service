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


class _ScrfdOnnxDetector:
    def __init__(self, model_path: Path, input_size: int, nms_threshold: float = 0.4) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("SCRFD ONNX detection requires the onnxruntime package.") from exc

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_size = (int(input_size), int(input_size))
        self.nms_threshold = float(nms_threshold)
        self.strides = [8, 16, 32]
        self.feature_count = 3
        self.use_keypoints = len(self.output_names) >= 9
        self.num_anchors = 2
        self.center_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def detect(self, image: np.ndarray, threshold: float, max_num: int = 0, metric: str = "default") -> tuple[np.ndarray, np.ndarray | None]:
        input_width, input_height = self.input_size
        image_ratio = float(image.shape[0]) / max(1, image.shape[1])
        model_ratio = float(input_height) / max(1, input_width)
        if image_ratio > model_ratio:
            resized_height = input_height
            resized_width = int(resized_height / image_ratio)
        else:
            resized_width = input_width
            resized_height = int(resized_width * image_ratio)

        det_scale = float(resized_height) / max(1, image.shape[0])
        resized = cv2.resize(image, (resized_width, resized_height))
        canvas = np.zeros((input_height, input_width, 3), dtype=np.uint8)
        canvas[:resized_height, :resized_width, :] = resized

        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 128.0,
            size=self.input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        scores_list = []
        boxes_list = []
        keypoints_list = []
        for index, stride in enumerate(self.strides):
            scores = outputs[index][0].reshape(-1)
            box_predictions = outputs[index + self.feature_count][0] * stride
            keypoint_predictions = None
            if self.use_keypoints:
                keypoint_predictions = outputs[index + (self.feature_count * 2)][0] * stride

            height = input_height // stride
            width = input_width // stride
            centers = self._anchor_centers(height, width, stride)
            selected = np.where(scores >= threshold)[0]
            if selected.size == 0:
                continue

            decoded_boxes = self._distance_to_boxes(centers, box_predictions)
            scores_list.append(scores[selected])
            boxes_list.append(decoded_boxes[selected])
            if keypoint_predictions is not None:
                decoded_keypoints = self._distance_to_keypoints(centers, keypoint_predictions).reshape((-1, 5, 2))
                keypoints_list.append(decoded_keypoints[selected])

        if not boxes_list:
            return np.zeros((0, 5), dtype=np.float32), None

        scores = np.concatenate(scores_list).reshape(-1)
        boxes = np.vstack(boxes_list) / det_scale
        order = scores.argsort()[::-1]
        detections = np.hstack((boxes, scores[:, None])).astype(np.float32, copy=False)[order]
        keep = self._nms(detections)
        detections = detections[keep]

        keypoints = None
        if keypoints_list:
            keypoints = np.vstack(keypoints_list)[order][keep] / det_scale

        if max_num > 0 and detections.shape[0] > max_num:
            selected = self._select_best_faces(detections, image.shape, max_num, metric)
            detections = detections[selected]
            if keypoints is not None:
                keypoints = keypoints[selected]
        return detections, keypoints

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        if key not in self.center_cache:
            centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape((-1, 2))
            centers = np.stack([centers] * self.num_anchors, axis=1).reshape((-1, 2))
            self.center_cache[key] = centers
        return self.center_cache[key]

    def _distance_to_boxes(self, points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                points[:, 0] - distance[:, 0],
                points[:, 1] - distance[:, 1],
                points[:, 0] + distance[:, 2],
                points[:, 1] + distance[:, 3],
            ],
            axis=-1,
        )

    def _distance_to_keypoints(self, points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        values = []
        for index in range(0, distance.shape[1], 2):
            values.append(points[:, 0] + distance[:, index])
            values.append(points[:, 1] + distance[:, index + 1])
        return np.stack(values, axis=-1)

    def _nms(self, detections: np.ndarray) -> list[int]:
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 4]
        areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            current = int(order[0])
            keep.append(current)
            xx1 = np.maximum(x1[current], x1[order[1:]])
            yy1 = np.maximum(y1[current], y1[order[1:]])
            xx2 = np.minimum(x2[current], x2[order[1:]])
            yy2 = np.minimum(y2[current], y2[order[1:]])
            width = np.maximum(0.0, xx2 - xx1 + 1.0)
            height = np.maximum(0.0, yy2 - yy1 + 1.0)
            overlap = (width * height) / np.maximum(1e-6, areas[current] + areas[order[1:]] - (width * height))
            order = order[np.where(overlap <= self.nms_threshold)[0] + 1]
        return keep

    def _select_best_faces(self, detections: np.ndarray, image_shape: tuple[int, ...], max_num: int, metric: str) -> np.ndarray:
        areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
        image_center = image_shape[0] // 2, image_shape[1] // 2
        offsets = np.vstack([
            (detections[:, 0] + detections[:, 2]) / 2.0 - image_center[1],
            (detections[:, 1] + detections[:, 3]) / 2.0 - image_center[0],
        ])
        values = areas if metric == "max" else areas - (np.sum(np.power(offsets, 2.0), axis=0) * 2.0)
        return np.argsort(values)[::-1][:max_num]


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

        self.runtime = self._build_runtime()

    def _model_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.settings.identity_model_root / path

    def _load_scrfd(self) -> Any:
        detector_path = self._model_path(self.settings.identity_scrfd_model)
        if not detector_path.is_file():
            raise RuntimeError(f"SCRFD model is missing: {detector_path}")
        return _ScrfdOnnxDetector(detector_path, int(self.settings.identity_scrfd_input_size))

    def _load_onnx(self, model_path: Path, label: str) -> Any:
        if not model_path.is_file():
            raise RuntimeError(f"{label} model is missing: {model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("The scrfd_adaface engine requires the onnxruntime package.") from exc
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def _build_runtime(self) -> dict[str, Any]:
        return {
            "identity_engine_config": str(self.settings.identity_engine),
            "identity_engine_loaded": "scrfd_adaface",
            "scrfd_model_exists": self._model_path(self.settings.identity_scrfd_model).is_file(),
            "adaface_model_exists": self._model_path(self.settings.identity_adaface_model).is_file(),
            "adaface_input_shape": self._shape_list(self.recognizer.get_inputs()[0].shape),
            "adaface_output_shape": self._shape_list(self.recognizer.get_outputs()[0].shape),
            "adaface_color_order": str(self.settings.identity_adaface_color_order).strip().lower(),
            "pass_threshold": float(self.settings.identity_pass_threshold),
            "review_threshold": float(self.settings.identity_review_threshold),
            "antispoof_model_exists": self._model_path(self.settings.identity_antispoof_model).is_file() if str(self.settings.identity_antispoof_model).strip() else False,
            "antispoof_input_shape": self._session_shape(self.antispoof, "input"),
            "antispoof_output_shape": self._session_shape(self.antispoof, "output"),
            "antispoof_live_class_index": int(self.settings.identity_antispoof_live_class_index),
            "antispoof_crop_scale": float(self.settings.identity_antispoof_crop_scale),
            "headpose_model_exists": self._model_path(self.settings.identity_headpose_model).is_file() if str(self.settings.identity_headpose_model).strip() else False,
            "headpose_input_shape": self._session_shape(self.headpose, "input"),
            "headpose_output_shape": self._session_shape(self.headpose, "output"),
        }

    def describe_runtime(self) -> dict[str, Any]:
        return dict(self.runtime)

    def _shape_list(self, shape: Any) -> list[Any]:
        if isinstance(shape, (list, tuple)):
            result = []
            for value in shape:
                if isinstance(value, (int, float)) and float(value).is_integer():
                    result.append(int(value))
                else:
                    result.append(value)
            return result
        return [shape]

    def _session_shape(self, session: Any, kind: str) -> list[Any]:
        if session is None:
            return []
        try:
            infos = session.get_inputs() if kind == "input" else session.get_outputs()
        except Exception:
            return []
        if not infos:
            return []
        return self._shape_list(infos[0].shape)

    def analyse(self, image: np.ndarray) -> AdvancedFaceQuality:
        """Runs detection, anti-spoofing, and head pose without AdaFace."""
        _bbox, _keypoints, quality = self._detect_and_measure(image)
        return quality

    def extract(self, image: np.ndarray) -> tuple[np.ndarray, AdvancedFaceQuality]:
        """Returns an AdaFace embedding and the complete quality result."""
        bbox, keypoints, quality = self._detect_and_measure(image)
        if bbox is None:
            output_shape = self.recognizer.get_outputs()[0].shape
            dimensions = output_shape[-1] if output_shape and isinstance(output_shape[-1], int) else 512
            return np.zeros((1, int(dimensions)), dtype=np.float32), quality

        aligned = self._align_face(image, keypoints, bbox)
        embedding = self._adaface_embedding(aligned)
        return embedding, quality

    def _detect_and_measure(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, AdvancedFaceQuality]:
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        bboxes, keypoints = self.detector.detect(
            image,
            threshold=float(self.settings.identity_min_face_confidence),
            max_num=0,
            metric="default",
        )
        face_count = 0 if bboxes is None else int(len(bboxes))
        base_quality = AdvancedFaceQuality(
            brightness=brightness,
            blur=blur,
            face_count=face_count,
            width=int(width),
            height=int(height),
        )
        if bboxes is None or len(bboxes) == 0:
            return None, None, base_quality

        index = int(np.argmax((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])))
        bbox = bboxes[index].astype(np.float32)
        kps = keypoints[index].astype(np.float32) if keypoints is not None else None

        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        confidence = float(bbox[4]) if len(bbox) >= 5 else None
        face_width = max(0.0, x2 - x1)
        yaw = self._landmark_yaw(kps)

        # Enrollment quality must describe the candidate's face, not the room.
        # Keep whole-frame values only for the no-face case above.
        face_crop = self._crop_bbox(image, bbox, margin_ratio=0.12)
        if face_crop.size > 0:
            face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(face_gray))
            blur = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())

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
        return bbox, kps, quality

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
        crop = self._crop_bbox_scaled(image, bbox, scale=float(self.settings.identity_antispoof_crop_scale))
        input_tensor = self._generic_image_tensor(
            crop,
            int(self.settings.identity_antispoof_input_size),
            mean=0.0,
            std=1.0,
            bgr=True,
        )
        input_name = self.antispoof.get_inputs()[0].name
        output = np.asarray(self.antispoof.run(None, {input_name: input_tensor})[0]).reshape(-1)
        if output.size == 1:
            value = float(output[0])
            if 0.0 <= value <= 1.0:
                return value
            return float(1.0 / (1.0 + np.exp(-value)))
        probabilities = self._class_probabilities(output)
        index = int(self.settings.identity_antispoof_live_class_index)
        if index < 0 or index >= probabilities.size:
            index = int(np.argmax(probabilities))
        return float(probabilities[index])

    def _class_probabilities(self, output: np.ndarray) -> np.ndarray:
        """Accepts either model probabilities or raw logits without double-softmax."""
        values = np.asarray(output, dtype=np.float32).reshape(-1)
        total = float(np.sum(values))
        if (np.all(np.isfinite(values)) and np.all(values >= 0.0) and np.all(values <= 1.0)
                and abs(total - 1.0) <= 1e-3):
            return values / max(total, 1e-12)
        return self._softmax(values)

    def _headpose_yaw(self, image: np.ndarray, bbox: np.ndarray) -> float | None:
        if self.headpose is None:
            return None
        crop = self._crop_bbox(image, bbox, margin_ratio=0.25)
        tensor = self._generic_image_tensor(crop, 224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], bgr=False)
        input_name = self.headpose.get_inputs()[0].name
        output = np.asarray(self.headpose.run(None, {input_name: tensor})[0], dtype=np.float32)
        if output.shape[-2:] == (3, 3):
            return self._rotation_matrix_yaw_degrees(output.reshape(-1, 3, 3)[0])
        output = output.reshape(-1)
        if output.size >= 3:
            return float(output[1])
        return None

    def _rotation_matrix_yaw_degrees(self, rotation: np.ndarray) -> float:
        sy = float(np.sqrt((rotation[0, 0] * rotation[0, 0]) + (rotation[1, 0] * rotation[1, 0])))
        yaw = float(np.arctan2(-rotation[2, 0], sy))
        return float(np.degrees(yaw))

    def _crop_bbox_scaled(self, image: np.ndarray, bbox: np.ndarray, scale: float) -> np.ndarray:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        side = max(width, height) * max(1.0, float(scale))
        side_int = max(1, int(round(side)))
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        left = int(round(center_x - (side / 2.0)))
        top = int(round(center_y - (side / 2.0)))
        right = left + side_int
        bottom = top + side_int

        crop = np.zeros((side_int, side_int, 3), dtype=image.dtype)
        source_left = max(0, left)
        source_top = max(0, top)
        source_right = min(image.shape[1], right)
        source_bottom = min(image.shape[0], bottom)
        if source_right <= source_left or source_bottom <= source_top:
            return crop

        dest_left = source_left - left
        dest_top = source_top - top
        dest_right = dest_left + (source_right - source_left)
        dest_bottom = dest_top + (source_bottom - source_top)
        crop[dest_top:dest_bottom, dest_left:dest_right] = image[source_top:source_bottom, source_left:source_right]
        return crop

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
