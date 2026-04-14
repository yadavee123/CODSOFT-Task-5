from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .face_detection import FaceDetector
from .face_recognition import FaceRecognitionService
from .utils import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    decode_base64_image,
    draw_label,
    ensure_directory,
    file_url,
    is_allowed_file,
    load_image,
    resize_for_processing,
    save_upload_file,
    unique_filename,
    write_image,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = ensure_directory(BASE_DIR / "uploads")
OUTPUT_DIR = ensure_directory(BASE_DIR / "outputs")
MODEL_DIR = ensure_directory(BASE_DIR / "backend" / "models")
EMBEDDINGS_PATH = MODEL_DIR / "face_embeddings.pkl"
ATTENDANCE_LOG_PATH = MODEL_DIR / "attendance_log.csv"

app = FastAPI(title="Face AI Studio", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/media/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/media/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

face_detector = FaceDetector()
recognition_service = FaceRecognitionService(
    embeddings_path=EMBEDDINGS_PATH,
    attendance_log_path=ATTENDANCE_LOG_PATH,
)
executor = ThreadPoolExecutor(max_workers=2)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/faces")
def list_registered_faces() -> dict[str, list[dict[str, str]]]:
    return {"faces": recognition_service.list_registered_faces()}


@app.post("/api/register")
async def register_face(
    name: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, object]:
    if not is_allowed_file(file.filename or "", IMAGE_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Only JPG and PNG files are supported for registration.")

    saved_path = await save_upload_file(file, UPLOAD_DIR)
    image = load_image(saved_path)
    entry = recognition_service.register_face(name, image)
    return {
        "message": f"Registered face for {entry['name']}.",
        "name": entry["name"],
        "registered_at": entry["registered_at"],
        "source_url": file_url(saved_path),
    }


@app.post("/api/process/image")
async def process_image(
    file: UploadFile = File(...),
    method: Literal["haar", "mtcnn"] = Form("mtcnn"),
    recognition_enabled: bool = Form(True),
    blur_mode: bool = Form(False),
) -> dict[str, object]:
    if not is_allowed_file(file.filename or "", IMAGE_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported.")

    saved_path = await save_upload_file(file, UPLOAD_DIR)
    image = load_image(saved_path)
    processed_image, summary = annotate_frame(
        frame=image,
        source_name=file.filename or saved_path.name,
        method=method,
        recognition_enabled=recognition_enabled,
        blur_mode=blur_mode,
    )

    output_path = OUTPUT_DIR / unique_filename(f"processed_{saved_path.name}")
    write_image(processed_image, output_path)
    return {
        "message": "Image processed successfully.",
        "detections": summary["detections"],
        "recognized_faces": summary["recognized_faces"],
        "unknown_faces": summary["unknown_faces"],
        "method": method,
        "output_url": file_url(output_path),
        "download_url": file_url(output_path),
        "input_url": file_url(saved_path),
    }


@app.post("/api/process/video")
async def process_video(
    file: UploadFile = File(...),
    method: Literal["haar", "mtcnn"] = Form("mtcnn"),
    recognition_enabled: bool = Form(True),
    blur_mode: bool = Form(False),
) -> dict[str, object]:
    if not is_allowed_file(file.filename or "", VIDEO_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Only MP4 and MOV videos are supported.")

    saved_path = await save_upload_file(file, UPLOAD_DIR)
    output_path = OUTPUT_DIR / unique_filename(f"processed_{saved_path.stem}.mp4")

    loop = __import__("asyncio").get_running_loop()
    summary = await loop.run_in_executor(
        executor,
        process_video_file,
        saved_path,
        output_path,
        method,
        recognition_enabled,
        blur_mode,
    )

    return {
        "message": "Video processed successfully.",
        "method": method,
        "output_url": file_url(output_path),
        "download_url": file_url(output_path),
        **summary,
    }


@app.post("/api/process/webcam-frame")
async def process_webcam_frame(
    image_data: str = Form(...),
    method: Literal["haar", "mtcnn"] = Form("mtcnn"),
    recognition_enabled: bool = Form(True),
    blur_mode: bool = Form(False),
) -> dict[str, object]:
    frame = decode_base64_image(image_data)
    processed_frame, summary = annotate_frame(
        frame=frame,
        source_name="webcam",
        method=method,
        recognition_enabled=recognition_enabled,
        blur_mode=blur_mode,
    )
    output_path = OUTPUT_DIR / unique_filename("webcam_frame.jpg")
    write_image(processed_frame, output_path)
    return {
        "message": "Webcam frame processed successfully.",
        "output_url": file_url(output_path),
        **summary,
    }


def annotate_frame(
    frame: cv2.typing.MatLike,
    source_name: str,
    method: str,
    recognition_enabled: bool,
    blur_mode: bool,
) -> tuple[cv2.typing.MatLike, dict[str, object]]:
    annotated = frame.copy()
    resized, scale = resize_for_processing(frame)
    detections = face_detector.detect_resized(resized, scale=scale, method=method)

    recognized_faces = 0
    unknown_faces = 0
    labels = []

    recognition_results = []
    if recognition_enabled and detections:
        recognition_results = recognition_service.recognize_from_detections(annotated, detections)
    else:
        recognition_results = [
            {"name": "Detection Only", "score": 0.0, "matched": False} for _ in detections
        ]

    for detection, recognition in zip(detections, recognition_results):
        x1, y1, x2, y2 = detection.box
        color = (32, 201, 151) if recognition.get("matched") else (0, 124, 255)

        if blur_mode:
            face_roi = annotated[y1:y2, x1:x2]
            if face_roi.size > 0:
                blurred = cv2.GaussianBlur(face_roi, (41, 41), 0)
                annotated[y1:y2, x1:x2] = blurred

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        if recognition_enabled:
            if recognition.get("matched"):
                recognized_faces += 1
                recognition_service.log_attendance(
                    recognition["name"],
                    source_name=source_name,
                    score=float(recognition.get("score", 0.0)),
                )
            else:
                unknown_faces += 1
            label = f"{recognition['name']} ({float(recognition.get('score', 0.0)):.2f})"
        else:
            label = f"Face {detection.confidence:.2f}"

        labels.append(label)
        draw_label(annotated, label, x1, y1, color)

    summary = {
        "detections": len(detections),
        "recognized_faces": recognized_faces,
        "unknown_faces": unknown_faces,
        "labels": labels,
    }
    return annotated, summary


def process_video_file(
    input_path: Path,
    output_path: Path,
    method: str,
    recognition_enabled: bool,
    blur_mode: bool,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="Unable to open the uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if width <= 0 or height <= 0:
        capture.release()
        raise HTTPException(status_code=400, detail="Invalid video stream dimensions.")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise HTTPException(status_code=500, detail="Unable to create the output video.")

    total_frames = 0
    total_detections = 0
    total_recognized = 0
    total_unknown = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            annotated_frame, summary = annotate_frame(
                frame=frame,
                source_name=input_path.name,
                method=method,
                recognition_enabled=recognition_enabled,
                blur_mode=blur_mode,
            )
            writer.write(annotated_frame)
            total_frames += 1
            total_detections += int(summary["detections"])
            total_recognized += int(summary["recognized_faces"])
            total_unknown += int(summary["unknown_faces"])
    finally:
        capture.release()
        writer.release()

    return {
        "frames_processed": total_frames,
        "detections": total_detections,
        "recognized_faces": total_recognized,
        "unknown_faces": total_unknown,
    }
