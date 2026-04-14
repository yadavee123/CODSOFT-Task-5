from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from .utils import scale_box


@dataclass
class DetectionResult:
    box: tuple[int, int, int, int]
    confidence: float
    method: str


class FaceDetector:
    def __init__(self) -> None:
        self.haar_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn_detector = MTCNN(keep_all=True, device=device)

    def detect_faces(
        self,
        image_bgr: np.ndarray,
        method: str = "mtcnn",
        min_confidence: float = 0.80,
    ) -> List[DetectionResult]:
        if method == "haar":
            return self._detect_with_haar(image_bgr)
        if method == "mtcnn":
            return self._detect_with_mtcnn(image_bgr, min_confidence=min_confidence)
        raise ValueError(f"Unsupported detection method: {method}")

    def _detect_with_haar(self, image_bgr: np.ndarray) -> List[DetectionResult]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.haar_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        results: List[DetectionResult] = []
        for (x, y, w, h) in faces:
            results.append(
                DetectionResult(
                    box=(int(x), int(y), int(x + w), int(y + h)),
                    confidence=1.0,
                    method="haar",
                )
            )
        return results

    def _detect_with_mtcnn(
        self,
        image_bgr: np.ndarray,
        min_confidence: float = 0.80,
    ) -> List[DetectionResult]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, probabilities = self.mtcnn_detector.detect(image_rgb)
        if boxes is None or probabilities is None:
            return []

        results: List[DetectionResult] = []
        for box, probability in zip(boxes, probabilities):
            confidence = float(probability)
            if confidence < min_confidence:
                continue
            x1, y1, x2, y2 = box
            results.append(
                DetectionResult(
                    box=(int(max(0, x1)), int(max(0, y1)), int(max(0, x2)), int(max(0, y2))),
                    confidence=confidence,
                    method="mtcnn",
                )
            )
        return results

    def detect_resized(
        self,
        image_bgr: np.ndarray,
        scale: float,
        method: str = "mtcnn",
        min_confidence: float = 0.80,
    ) -> List[DetectionResult]:
        detections = self.detect_faces(image_bgr, method=method, min_confidence=min_confidence)
        if scale == 1.0:
            return detections

        scaled_results: List[DetectionResult] = []
        for detection in detections:
            scaled_results.append(
                DetectionResult(
                    box=scale_box(detection.box, scale),
                    confidence=detection.confidence,
                    method=detection.method,
                )
            )
        return scaled_results
