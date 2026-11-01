import { loadRuntime } from "./wasm-runtime.js";

const revision = document.querySelector('meta[name="etalon-revision"]').content;
const version = document.querySelector('meta[name="etalon-version"]').content;
const capabilities = document.querySelector("#radio-capabilities");
const status = document.querySelector("#radio-status");
const receipt = document.querySelector("#radio-receipt");
const notificationView = document.querySelector("#ble-notifications");
const service = document.querySelector("#ble-service");
const characteristic = document.querySelector("#ble-characteristic");
const writeHex = document.querySelector("#ble-write-hex");
const controls = Object.fromEntries(
  [
    "connect",
    "read",
    "write",
    "start-notifications",
    "stop-notifications",
    "disconnect",
  ].map((name) => [name, document.querySelector("#ble-" + name)]),
);
let wasm;
let bluetooth;
let busy = false;
let closed = false;
let connected = false;
let subscribed = false;
let available = false;
const pending = new Set();
let polling;
let capabilityBusy = false;
let notifications = [];
let displayDropped = 0;
let generation = 0;

function bounded(value) {
  const encoded = JSON.stringify(value, null, 2);
  const bytes = new TextEncoder().encode(encoded).length;
  return bytes <= 65536
    ? encoded
    : JSON.stringify(
        {
          schema: "etalon.browser-radio-ui.v1",
          source_revision: revision,
          library_version: version,
          truncated: true,
          original_bytes: bytes,
          error: "Displayed receipt exceeds the 64 KiB limit",
        },
        null,
        2,
      );
}
function buttons() {
  const enabled = Boolean(bluetooth) && !closed && !busy;
  controls.connect.disabled = !enabled || !available || connected;
  controls.read.disabled = !enabled || !connected;
  controls.write.disabled = !enabled || !connected;
  controls["start-notifications"].disabled =
    !enabled || !connected || subscribed;
  controls["stop-notifications"].disabled = !enabled || !subscribed;
  controls.disconnect.disabled =
    !bluetooth || closed || (!connected && !busy && !subscribed);
  service.disabled = !enabled || connected;
  characteristic.disabled = !enabled || connected;
  writeHex.disabled = !enabled || !connected;
}
function stopPolling() {
  clearInterval(polling);
  polling = undefined;
}
function state(result) {
  if (typeof result.connected === "boolean") connected = result.connected;
  if (typeof result.subscribed === "boolean") subscribed = result.subscribed;
  if (!connected) subscribed = false;
  if (!subscribed) stopPolling();
  buttons();
}
function decode(encoded) {
  const result = typeof encoded === "string" ? JSON.parse(encoded) : encoded;
  if (!result || typeof result.ok !== "boolean")
    throw new Error("Invalid Bluetooth response");
  if (result.source_revision !== revision || result.library_version !== version)
    throw new Error("Bluetooth response version differs from this page");
  return result;
}
function show(operation, result) {
  receipt.textContent = bounded({
    schema: "etalon.browser-radio-ui.v1",
    source_revision: revision,
    library_version: version,
    operation,
    recorded_at: new Date().toISOString(),
    automatic_upload: false,
    result,
  });
  state(result);
  status.textContent = result.ok
    ? `${operation} completed${connected ? "; device connected" : "; device disconnected"}.`
    : `${operation}: ${String(result.error || "Operation failed").slice(0, 1024)}`;
}
async function perform(operation, action) {
  if (busy || !bluetooth || closed) return;
  const epoch = generation;
  const owner = bluetooth;
  let request;
  busy = true;
  buttons();
  status.textContent = `${operation} in progress…`;
  try {
    request = { owner, promise: Promise.resolve(action()) };
    pending.add(request);
    const result = decode(await request.promise);
    if (closed || epoch !== generation) return;
    show(operation, result);
    if (result.ok && subscribed && !polling) polling = setInterval(drain, 1000);
  } catch (error) {
    if (!closed && epoch === generation)
      show(operation, { ok: false, error: String(error).slice(0, 1024) });
  } finally {
    if (request) pending.delete(request);
    if (epoch === generation) {
      busy = false;
      buttons();
    }
  }
}
function drain() {
  if (!bluetooth || busy || closed || !subscribed) return;
  try {
    const result = decode(bluetooth.drain_notifications());
    state(result);
    if (!result.ok) {
      stopPolling();
      show("Notifications", result);
      return;
    }
    const batch = Array.isArray(result.notifications)
      ? result.notifications.slice(-32)
      : [];
    if (batch.length) {
      notifications.push(...batch);
      if (notifications.length > 32) {
        displayDropped += notifications.length - 32;
        notifications = notifications.slice(-32);
      }
    }
    notificationView.textContent = bounded({
      schema: "etalon.bluetooth-notifications-ui.v1",
      source_revision: revision,
      library_version: version,
      connected,
      subscribed,
      automatic_upload: false,
      received: notifications,
      display_dropped: displayDropped,
      device_ring_dropped: result.dropped,
      errors: result.errors,
    });
    if (!connected)
      status.textContent =
        "Bluetooth device disconnected. Choose and connect explicitly to resume.";
    else if (result.errors)
      status.textContent = `Notification stream reports ${result.errors} errors; see evidence.`;
  } catch (error) {
    stopPolling();
    show("Notifications", { ok: false, error: String(error).slice(0, 1024) });
  }
}
controls.connect.addEventListener("click", () => {
  const uuid =
    /^(?:[0-9a-f]{4}|[0-9a-f]{8}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i;
  const serviceId = service.value.trim();
  const characteristicId = characteristic.value.trim();
  if (!uuid.test(serviceId) || !uuid.test(characteristicId)) {
    show("Connect", {
      ok: false,
      error:
        "Enter a 4-digit, 8-digit or canonical UUID for both service and characteristic.",
    });
    return;
  }
  notifications = [];
  displayDropped = 0;
  notificationView.textContent = "";
  perform("Connect", () => bluetooth.connect(serviceId, characteristicId));
});
controls.read.addEventListener("click", () =>
  perform("Read", () => bluetooth.read()),
);
controls.write.addEventListener("click", () => {
  const hex = writeHex.value.replace(/\s/g, "");
  if (!/^[0-9a-f]+$/i.test(hex) || hex.length % 2 || hex.length > 1024) {
    show("Write", {
      ok: false,
      error:
        "Enter 1–512 bytes as complete hexadecimal pairs; spaces are optional.",
    });
    return;
  }
  perform("Write", () => bluetooth.write(hex));
});
controls["start-notifications"].addEventListener("click", () =>
  perform("Start notifications", () => bluetooth.start_notifications()),
);
controls["stop-notifications"].addEventListener("click", () =>
  perform("Stop notifications", () => bluetooth.stop_notifications()),
);
controls.disconnect.addEventListener("click", () => {
  if (!bluetooth || closed) return;
  generation += 1;
  busy = false;
  capabilityBusy = false;
  stopPolling();
  try {
    show("Cancel / disconnect", decode(bluetooth.disconnect()));
  } catch (error) {
    show("Cancel / disconnect", {
      ok: false,
      error: String(error).slice(0, 1024),
    });
  }
  buttons();
});
async function refreshCapabilities() {
  if (!wasm || closed || capabilityBusy) return;
  capabilityBusy = true;
  const epoch = generation;
  try {
    const result = JSON.parse(await wasm.radio_capabilities());
    if (closed || epoch !== generation) return;
    if (
      result.source_revision !== revision ||
      result.library_version !== version
    )
      throw new Error("Page and radio WASM versions differ");
    available =
      result.bluetooth?.api_available === true &&
      result.bluetooth?.available !== false;
    capabilities.textContent = bounded(result);
  } catch (error) {
    if (closed || epoch !== generation) return;
    available = false;
    capabilities.textContent = bounded({
      source_revision: revision,
      library_version: version,
      error: String(error).slice(0, 1024),
    });
  } finally {
    if (epoch === generation) {
      capabilityBusy = false;
      buttons();
    }
  }
}
document
  .querySelector("#refresh-radio")
  .addEventListener("click", refreshCapabilities);
window.addEventListener("online", refreshCapabilities);
window.addEventListener("offline", refreshCapabilities);
navigator.connection?.addEventListener("change", refreshCapabilities);
async function initialize() {
  const epoch = ++generation;
  buttons();
  try {
    wasm = await loadRuntime();
    if (closed || epoch !== generation) return;
    bluetooth = new wasm.BrowserBluetooth();
    await refreshCapabilities();
    if (closed || epoch !== generation) return;
    status.textContent = available
      ? "Ready. Enter service and characteristic UUIDs, then choose a device explicitly."
      : "Bluetooth connection is unavailable in this browser context. Network capability evidence remains available.";
  } catch (error) {
    if (closed || epoch !== generation) return;
    status.textContent = `Radio initialization failed: ${String(error).slice(0, 1024)}`;
  }
  buttons();
}
window.addEventListener("pagehide", () => {
  closed = true;
  generation += 1;
  stopPolling();
  const owner = bluetooth;
  const closing = [...pending]
    .filter((request) => request.owner === owner)
    .map((request) => request.promise);
  busy = false;
  capabilityBusy = false;
  bluetooth = undefined;
  connected = false;
  subscribed = false;
  if (!owner) return;
  try {
    owner.disconnect();
  } catch (error) {
    console.warn(
      "Etalon Bluetooth disconnect during page close failed:",
      String(error).slice(0, 512),
    );
  }
  const release = () => {
    try {
      owner.free();
    } catch (error) {
      console.warn(
        "Etalon Bluetooth owner release failed:",
        String(error).slice(0, 512),
      );
    }
  };
  if (closing.length) Promise.allSettled(closing).then(release);
  else release();
});
window.addEventListener("pageshow", (event) => {
  if (event.persisted && closed) {
    closed = false;
    initialize();
  }
});
initialize();
