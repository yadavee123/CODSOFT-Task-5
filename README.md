# Face AI Studio

Face AI Studio is a polished full-stack AI web application for local face analysis. It supports image uploads, video uploads, and webcam frame capture with:

- Face detection using both Haar Cascade and MTCNN
- Face recognition using FaceNet embeddings via `facenet-pytorch`
- Local embedding storage in pickle format
- Privacy blur mode as a standout custom feature
- Attendance logging for recognized faces
- Downloadable processed outputs in the browser

## Project Structure

```text
face-ai-app/
|
|-- backend/
|   |-- __init__.py
|   |-- app.py
|   |-- face_detection.py
|   |-- face_recognition.py
|   |-- utils.py
|   `-- models/
|
|-- frontend/
|   |-- index.html
|   |-- style.css
|   `-- script.js
|
|-- uploads/
|-- outputs/
|-- requirements.txt
`-- README.md
```

## Features

### Core Features
- Upload image files in JPG or PNG format
- Detect faces and draw bounding boxes
- Upload MP4 or MOV videos and process them frame by frame
- Register known people with names and store their embeddings locally
- Recognize registered people inside processed images, videos, and webcam snapshots
- Download processed outputs directly from the UI

### Detection Methods
- `Haar Cascade`: a lightweight classical baseline
- `MTCNN`: a deep learning detector for better robustness

### Recognition Pipeline
- FaceNet-style embeddings using `InceptionResnetV1` from `facenet-pytorch`
- Local storage in `backend/models/face_embeddings.pkl`
- Cosine similarity matching with a configurable threshold

### Unique Feature
- `Privacy Blur Mode`: blur all detected faces while preserving labels and detection overlays so you can safely share results in demos, reports, or internship submissions

### Bonus Features
- REST API endpoints for registration and processing
- Webcam capture in the browser
- Attendance logging to `backend/models/attendance_log.csv`
- Threaded video processing using `ThreadPoolExecutor`

## Screenshots

Add your screenshots here after running the app locally.

- `Home dashboard screenshot`
- `Image upload result screenshot`
- `Video processing result screenshot`
- `Webcam recognition screenshot`

## Setup Instructions

### 1. Create and activate a virtual environment

```bash
cd face-ai-app
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the application

```bash
uvicorn backend.app:app --reload
```

### 4. Open the app

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Important First-Run Note

The FaceNet and MTCNN weights used by `facenet-pytorch` are downloaded automatically the first time the app starts if they are not already cached locally. Make sure your machine has internet access for the initial run.

## How To Use

### Register Faces
1. Open the app in your browser.
2. In the `Register a Face` panel, enter a name.
3. Upload a clear front-facing portrait.
4. Submit the form to store the embedding locally.

### Process an Image
1. Stay on the `Image` tab.
2. Drag and drop a JPG or PNG file.
3. Choose the detection method.
4. Optionally enable recognition and privacy blur.
5. Click `Process Image`.

### Process a Video
1. Switch to the `Video` tab.
2. Drag and drop an MP4 or MOV file.
3. Click `Process Video`.
4. Wait for frame-by-frame processing to finish.
5. Download the processed video from the results panel.

### Use the Webcam Bonus
1. Switch to the `Webcam Bonus` tab.
2. Click `Start Webcam`.
3. Capture the current frame.
4. Review the processed result in the results panel.

## API Endpoints

- `GET /api/health`: health check
- `GET /api/faces`: list registered faces
- `POST /api/register`: register a person from an uploaded image
- `POST /api/process/image`: detect and optionally recognize faces in an image
- `POST /api/process/video`: detect and optionally recognize faces in a video
- `POST /api/process/webcam-frame`: process a base64 webcam frame

## Error Handling

The app returns clear error messages for:

- Unsupported file types
- Invalid or corrupted uploads
- Missing faces in registration images
- Video decoding issues
- Webcam decoding failures

## Performance Notes

- Video frames are resized before detection to keep processing responsive
- Bounding boxes are scaled back to the original frame size for clean output
- Video processing runs in a thread pool to keep the API responsive
- Recognition runs only when enabled from the UI

## Suggested Demo Flow

1. Register two known faces.
2. Upload a group image and process it with `MTCNN`.
3. Re-run the same image with `Privacy Blur Mode` enabled.
4. Upload a short MP4 clip and download the annotated result.
5. Show the webcam tab for the bonus live workflow.

## Submission Notes

This project is intentionally modular, production-minded, and internship-ready. To make it even stronger for submission, add your own screenshots and optionally deploy it behind a reverse proxy or Docker container.
