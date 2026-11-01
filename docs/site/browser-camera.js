import { loadRuntime } from "./wasm-runtime.js";

const listButton = document.querySelector("#camera-enumerate");
const startButton = document.querySelector("#camera-start");
const stopButton = document.querySelector("#camera-stop");
const captureButton = document.querySelector("#camera-capture");
const device = document.querySelector("#camera-device");
const facing = document.querySelector("#camera-facing");
const video = document.querySelector("#camera-video");
const canvas = document.querySelector("#camera-canvas");
const status = document.querySelector("#camera-status");
const capabilities = document.querySelector("#camera-capabilities");
const frameReport = document.querySelector("#camera-frame-report");
const revision = document.querySelector('meta[name="etalon-revision"]').content;
let wasm;
let state = "idle";
let generation = 0;

function parse(encoded) {
  const report = JSON.parse(encoded);
  if (!report.ok) throw new Error(report.error || "Camera operation failed");
  if (report.source_revision !== revision)
    throw new Error(
      "Camera WASM revision differs from this page; reload before capture.",
    );
  return report;
}

function controls() {
  startButton.disabled = state !== "idle";
  listButton.disabled = state !== "idle";
  stopButton.disabled = state === "idle" || state === "listing";
  captureButton.disabled = state !== "live" || video.readyState < 2;
  device.disabled = state !== "idle";
  facing.disabled = state !== "idle" || Boolean(device.value);
}

function stop(
  message = "Camera stopped. No frames were uploaded or automatically saved.",
) {
  generation++;
  let failure;
  try {
    if (wasm) parse(wasm.camera_stop());
  } catch (error) {
    failure = String(error);
  }
  video.pause();
  video.srcObject = null;
  canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  state = "idle";
  status.textContent = failure || message;
  controls();
}

listButton.addEventListener("click", async () => {
  if (state !== "idle") return;
  const previousState = state;
  const request = generation;
  state = "listing";
  controls();
  status.textContent =
    "Reading camera devices exposed by this browser; no permission request…";
  try {
    wasm = await loadRuntime();
    const report = parse(await wasm.camera_capabilities());
    if (request !== generation) return;
    capabilities.textContent = JSON.stringify(report, null, 2);
    const selected = device.value;
    device.replaceChildren(new Option("Browser default", ""));
    report.cameras.forEach((camera, index) => {
      if (camera.device_id)
        device.append(
          new Option(
            camera.label || `Camera ${index + 1} (label hidden)`,
            camera.device_id,
          ),
        );
    });
    if ([...device.options].some((option) => option.value === selected))
      device.value = selected;
    status.textContent = `${report.cameras.length} exposed camera entries. Hidden labels may become visible after explicit permission. IR and thermal classification remain unknown.`;
  } catch (error) {
    if (request === generation) status.textContent = String(error);
  } finally {
    if (request === generation) state = previousState;
    controls();
  }
});

startButton.addEventListener("click", async () => {
  if (state !== "idle") return;
  const request = ++generation;
  state = "starting";
  controls();
  status.textContent =
    "Waiting for camera permission. Stop cancels this app's request; a browser prompt may remain until dismissed.";
  frameReport.textContent = "";
  let timer;
  try {
    wasm = await loadRuntime();
    if (request !== generation) return;
    timer = setTimeout(() => {
      if (request === generation)
        stop(
          "Camera acquisition timed out and was cancelled; any late stream will be stopped.",
        );
    }, 60000);
    const report = parse(await wasm.camera_start(device.value, facing.value));
    if (request !== generation) return;
    video.srcObject = wasm.camera_stream();
    const tracks = video.srcObject.getVideoTracks();
    for (const track of tracks)
      track.addEventListener(
        "ended",
        () => {
          if (request === generation)
            stop("Camera disconnected or permission ended; stream stopped.");
        },
        { once: true },
      );
    await video.play();
    if (request !== generation) return;
    capabilities.textContent = JSON.stringify(report, null, 2);
    state = "live";
    status.textContent =
      "Camera preview active. Ingest one frame explicitly to compute its summary in Rust/WASM. Infrared classification remains unknown.";
    controls();
  } catch (error) {
    if (request === generation) stop(String(error));
  } finally {
    clearTimeout(timer);
  }
});

captureButton.addEventListener("click", () => {
  if (state !== "live" || !wasm) return;
  try {
    if (video.readyState < 2 || !video.videoWidth || !video.videoHeight)
      throw new Error("Camera has not delivered a video frame yet.");
    const scale = Math.min(1, 640 / video.videoWidth, 480 / video.videoHeight);
    canvas.width = Math.max(1, Math.floor(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.floor(video.videoHeight * scale));
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Canvas pixel readback is unavailable.");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const report = parse(
      wasm.camera_ingest_rgba(
        canvas.width,
        canvas.height,
        new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength),
      ),
    );
    frameReport.textContent = JSON.stringify(report, null, 2);
    status.textContent = `Rust/WASM ingested ${report.bytes_ingested} RGBA bytes (${report.width}×${report.height}); checksum ${report.checksum_fnv1a64}. No thermal or GPU qualification is established.`;
    context.clearRect(0, 0, canvas.width, canvas.height);
  } catch (error) {
    frameReport.textContent = "";
    status.textContent = String(error);
  }
});

stopButton.addEventListener("click", () => stop());
device.addEventListener("change", controls);
video.addEventListener("loadeddata", controls);
window.addEventListener("pagehide", () => stop());
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden" && state !== "idle")
    stop("Camera stopped when this page became hidden.");
});
controls();
