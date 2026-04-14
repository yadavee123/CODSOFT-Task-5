from __future__ import annotations

import base64
import os
import shutil
import uuid
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_allowed_file(filename: str, allowed_extensions: Iterable[str]) -> bool:
    return Path(filename).suffix.lower() in set(allowed_extensions)


def unique_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    stem = Path(filename).stem.replace(" ", "_") or "file"
    return f"{stem}_{uuid.uuid4().hex[:10]}{suffix}"


async def save_upload_file(upload_file: UploadFile, destination_dir: Path) -> Path:
    if not upload_file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a name.")

    ensure_directory(destination_dir)
    output_path = destination_dir / unique_filename(upload_file.filename)

    with output_path.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return output_path


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode the uploaded image.")
    return image


def decode_base64_image(data_url: str) -> np.ndarray:
    if "," not in data_url:
        raise HTTPException(status_code=400, detail="Invalid webcam frame payload.")

    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode webcam image.")
    return image


def resize_for_processing(frame: np.ndarray, target_width: int = 960) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame.copy(), 1.0

    scale = target_width / float(width)
    resized = cv2.resize(frame, (target_width, int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def scale_box(box: list[int] | tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    if scale == 1.0:
        x1, y1, x2, y2 = box
        return int(x1), int(y1), int(x2), int(y2)

    inverse = 1.0 / scale
    x1, y1, x2, y2 = box
    return (
        int(x1 * inverse),
        int(y1 * inverse),
        int(x2 * inverse),
        int(y2 * inverse),
    )


def box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - height - baseline - 8)
    cv2.rectangle(frame, (x, top), (x + width + 10, top + height + baseline + 8), color, -1)
    cv2.putText(frame, text, (x + 5, top + height + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def safe_crop(image: np.ndarray, box: tuple[int, int, int, int], padding: float = 0.15) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    return image[y1:y2, x1:x2]


def write_image(image: np.ndarray, output_path: Path) -> Path:
    ensure_directory(output_path.parent)
    if not cv2.imwrite(str(output_path), image):
        raise HTTPException(status_code=500, detail="Unable to save the processed image.")
    return output_path


def file_url(path: Path) -> str:
    normalized = path.as_posix()
    if "/outputs/" in normalized:
        return f"/media/outputs/{path.name}"
    if "/uploads/" in normalized:
        return f"/media/uploads/{path.name}"
    return f"/media/{path.name}"


def remove_file_quietly(path: Path) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
