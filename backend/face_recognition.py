from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from fastapi import HTTPException

from .utils import ensure_directory


@dataclass
class RecognitionMatch:
    name: str
    score: float
    matched: bool


class FaceRecognitionService:
    def __init__(
        self,
        embeddings_path: Path,
        attendance_log_path: Path,
        similarity_threshold: float = 0.72,
    ) -> None:
        self.embeddings_path = embeddings_path
        self.attendance_log_path = attendance_log_path
        self.similarity_threshold = similarity_threshold

        ensure_directory(self.embeddings_path.parent)
        ensure_directory(self.attendance_log_path.parent)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.align_mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=self.device)
        self.embedding_model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.database = self._load_database()
        self._ensure_attendance_header()

    def register_face(self, name: str, image_bgr: np.ndarray) -> dict[str, Any]:
        cleaned_name = name.strip()
        if not cleaned_name:
            raise HTTPException(status_code=400, detail="A name is required to register a face.")

        embedding = self.extract_embedding(image_bgr)
        entry = {
            "name": cleaned_name,
            "embedding": embedding.tolist(),
            "registered_at": datetime.utcnow().isoformat(),
        }
        self.database.append(entry)
        self._save_database()
        return entry

    def recognize_face(self, face_bgr: np.ndarray) -> RecognitionMatch:
        if not self.database:
            return RecognitionMatch(name="Unknown", score=0.0, matched=False)

        embedding = self.extract_embedding(face_bgr)
        normalized_embedding = self._normalize(embedding)

        best_name = "Unknown"
        best_score = -1.0
        for item in self.database:
            stored = self._normalize(np.asarray(item["embedding"], dtype=np.float32))
            score = float(np.dot(normalized_embedding, stored))
            if score > best_score:
                best_name = item["name"]
                best_score = score

        matched = best_score >= self.similarity_threshold
        return RecognitionMatch(
            name=best_name if matched else "Unknown",
            score=max(0.0, best_score),
            matched=matched,
        )

    def extract_embedding(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        aligned_face = self.align_mtcnn(image_rgb)
        if aligned_face is None:
            raise HTTPException(status_code=400, detail="No clear face found for recognition.")

        aligned_face = aligned_face.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.embedding_model(aligned_face).cpu().numpy()[0]
        return embedding.astype(np.float32)

    def recognize_from_detections(
        self,
        frame_bgr: np.ndarray,
        detections: list[Any],
    ) -> list[dict[str, Any]]:
        frame_height, frame_width = frame_bgr.shape[:2]
        results: list[dict[str, Any]] = []
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            x1 = max(0, min(frame_width, x1))
            x2 = max(0, min(frame_width, x2))
            y1 = max(0, min(frame_height, y1))
            y2 = max(0, min(frame_height, y2))
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                results.append({"name": "Unknown", "score": 0.0, "matched": False})
                continue
            try:
                match = self.recognize_face(crop)
            except HTTPException:
                match = RecognitionMatch(name="Unknown", score=0.0, matched=False)
            results.append({"name": match.name, "score": match.score, "matched": match.matched})
        return results

    def log_attendance(self, name: str, source_name: str, score: float) -> None:
        if name == "Unknown":
            return
        with self.attendance_log_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                name,
                source_name,
                f"{score:.3f}",
            ])

    def list_registered_faces(self) -> list[dict[str, str]]:
        summary: dict[str, str] = {}
        for item in self.database:
            summary[item["name"]] = item["registered_at"]
        return [
            {"name": name, "registered_at": registered_at}
            for name, registered_at in sorted(summary.items())
        ]

    def _ensure_attendance_header(self) -> None:
        if self.attendance_log_path.exists():
            return
        with self.attendance_log_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["timestamp", "name", "source", "score"])

    def _load_database(self) -> List[Dict[str, Any]]:
        if not self.embeddings_path.exists():
            return []
        with self.embeddings_path.open("rb") as file_handle:
            return pickle.load(file_handle)

    def _save_database(self) -> None:
        with self.embeddings_path.open("wb") as file_handle:
            pickle.dump(self.database, file_handle)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm
