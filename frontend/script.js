const state = {
  imageFile: null,
  videoFile: null,
  webcamStream: null,
};

const detectorMethod = document.getElementById("detectorMethod");
const enableRecognition = document.getElementById("enableRecognition");
const enableBlur = document.getElementById("enableBlur");
const registerForm = document.getElementById("registerForm");
const registerFile = document.getElementById("registerFile");
const registeredFaces = document.getElementById("registeredFaces");
const statusPill = document.getElementById("statusPill");
const resultStats = document.getElementById("resultStats");
const resultPreview = document.getElementById("resultPreview");
const downloadLink = document.getElementById("downloadLink");
const messageBox = document.getElementById("messageBox");
const webcamPreview = document.getElementById("webcamPreview");
const webcamCanvas = document.getElementById("webcamCanvas");

function setStatus(text, mode = "idle") {
  statusPill.textContent = text;
  statusPill.className = `status-pill ${mode}`;
}

function showMessage(message, isError = false) {
  messageBox.textContent = message;
  messageBox.style.color = isError ? "#ff9b9b" : "#a8bdd7";
}

function getCommonFormData() {
  const formData = new FormData();
  formData.append("method", detectorMethod.value);
  formData.append("recognition_enabled", String(enableRecognition.checked));
  formData.append("blur_mode", String(enableBlur.checked));
  return formData;
}

function updateStats(payload) {
  resultStats.innerHTML = `Faces: <strong>${payload.detections ?? 0}</strong> &nbsp; Recognized: <strong>${payload.recognized_faces ?? 0}</strong> &nbsp; Unknown: <strong>${payload.unknown_faces ?? 0}</strong>`;
}

function renderResult(url, type) {
  resultPreview.classList.remove("empty");
  if (type === "video") {
    resultPreview.innerHTML = `<video controls src="${url}"></video>`;
  } else {
    resultPreview.innerHTML = `<img src="${url}?t=${Date.now()}" alt="Processed result" />`;
  }
  downloadLink.href = url;
  downloadLink.classList.remove("hidden");
  downloadLink.textContent = type === "video" ? "Download Processed Video" : "Download Processed Image";
}

async function sendFormRequest(url, formData) {
  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Something went wrong while processing the request.");
  }
  return payload;
}

function attachDropzone(dropzoneId, inputId, key) {
  const dropzone = document.getElementById(dropzoneId);
  const input = document.getElementById(inputId);

  dropzone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    state[key] = input.files[0] || null;
    dropzone.querySelector("p:last-child").textContent = state[key]
      ? `Selected: ${state[key].name}`
      : dropzone.dataset.type === "image"
        ? "Supports JPG and PNG. Or click to browse."
        : "Supports MP4 and MOV. Processed video will be downloadable.";
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    if (!file) return;
    state[key] = file;
    input.files = event.dataTransfer.files;
    dropzone.querySelector("p:last-child").textContent = `Selected: ${file.name}`;
  });
}

async function loadRegisteredFaces() {
  try {
    const response = await fetch("/api/faces");
    const payload = await response.json();
    const faces = payload.faces || [];
    if (!faces.length) {
      registeredFaces.innerHTML = `<div class="face-chip"><span>No registered faces yet</span><span>Start with one portrait</span></div>`;
      return;
    }

    registeredFaces.innerHTML = faces
      .map(
        (face) => `
          <div class="face-chip">
            <span>${face.name}</span>
            <span>${new Date(face.registered_at).toLocaleDateString()}</span>
          </div>
        `,
      )
      .join("");
  } catch (error) {
    registeredFaces.innerHTML = `<div class="face-chip"><span>Could not load faces</span><span>Check backend</span></div>`;
  }
}

registerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = registerFile.files[0];
  const name = document.getElementById("registerName").value.trim();

  if (!file || !name) {
    showMessage("Please provide both a name and a portrait image.", true);
    return;
  }

  try {
    setStatus("Registering", "busy");
    showMessage("Creating a FaceNet embedding for the new person...");
    const formData = new FormData();
    formData.append("name", name);
    formData.append("file", file);
    const payload = await sendFormRequest("/api/register", formData);
    showMessage(payload.message);
    registerForm.reset();
    await loadRegisteredFaces();
    setStatus("Ready", "idle");
  } catch (error) {
    setStatus("Error", "error");
    showMessage(error.message, true);
  }
});

document.getElementById("processImageButton").addEventListener("click", async () => {
  if (!state.imageFile) {
    showMessage("Please choose an image first.", true);
    return;
  }

  try {
    setStatus("Processing", "busy");
    showMessage("Running face detection on the uploaded image...");
    const formData = getCommonFormData();
    formData.append("file", state.imageFile);
    const payload = await sendFormRequest("/api/process/image", formData);
    renderResult(payload.output_url, "image");
    updateStats(payload);
    showMessage(payload.message);
    setStatus("Ready", "idle");
  } catch (error) {
    setStatus("Error", "error");
    showMessage(error.message, true);
  }
});

document.getElementById("processVideoButton").addEventListener("click", async () => {
  if (!state.videoFile) {
    showMessage("Please choose a video first.", true);
    return;
  }

  try {
    setStatus("Processing", "busy");
    showMessage("Processing the full video frame by frame. This can take a moment.");
    const formData = getCommonFormData();
    formData.append("file", state.videoFile);
    const payload = await sendFormRequest("/api/process/video", formData);
    renderResult(payload.output_url, "video");
    updateStats(payload);
    showMessage(`${payload.message} Frames processed: ${payload.frames_processed}.`);
    setStatus("Ready", "idle");
  } catch (error) {
    setStatus("Error", "error");
    showMessage(error.message, true);
  }
});

document.getElementById("startWebcamButton").addEventListener("click", async () => {
  try {
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach((track) => track.stop());
      state.webcamStream = null;
      webcamPreview.srcObject = null;
      showMessage("Webcam stopped.");
      return;
    }

    state.webcamStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    webcamPreview.srcObject = state.webcamStream;
    showMessage("Webcam started. Capture a frame whenever you are ready.");
  } catch (error) {
    showMessage("Unable to access the webcam in this browser.", true);
  }
});

document.getElementById("captureWebcamButton").addEventListener("click", async () => {
  if (!state.webcamStream) {
    showMessage("Start the webcam before capturing a frame.", true);
    return;
  }

  const width = webcamPreview.videoWidth;
  const height = webcamPreview.videoHeight;
  if (!width || !height) {
    showMessage("Webcam is still warming up. Try again in a second.", true);
    return;
  }

  webcamCanvas.width = width;
  webcamCanvas.height = height;
  const context = webcamCanvas.getContext("2d");
  context.drawImage(webcamPreview, 0, 0, width, height);

  try {
    setStatus("Analyzing", "busy");
    showMessage("Inspecting the current webcam frame...");
    const formData = getCommonFormData();
    formData.append("image_data", webcamCanvas.toDataURL("image/jpeg", 0.92));
    const payload = await sendFormRequest("/api/process/webcam-frame", formData);
    renderResult(payload.output_url, "image");
    updateStats(payload);
    showMessage(payload.message);
    setStatus("Ready", "idle");
  } catch (error) {
    setStatus("Error", "error");
    showMessage(error.message, true);
  }
});

document.querySelectorAll(".tab").forEach((tabButton) => {
  tabButton.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((button) => button.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((panel) => panel.classList.remove("active"));
    tabButton.classList.add("active");
    document.getElementById(`${tabButton.dataset.tab}Tab`).classList.add("active");
  });
});

attachDropzone("imageDropzone", "imageFile", "imageFile");
attachDropzone("videoDropzone", "videoFile", "videoFile");
loadRegisteredFaces();
