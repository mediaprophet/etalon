import { loadRuntime } from "./wasm-runtime.js";
const capabilities = document.querySelector("#storage-capabilities");
const status = document.querySelector("#storage-status");
const saveButton = document.querySelector("#save-local-report");
const refreshButton = document.querySelector("#refresh-local-reports");
const persistenceButton = document.querySelector("#request-persistence");
const source = document.querySelector("#local-report-source");
const reports = document.querySelector("#local-reports");
const revision = document.querySelector('meta[name="etalon-revision"]').content;
let wasm;
let supported = false;
let persistenceAvailable = false;
let busy = false;

function parse(value) {
  const result = JSON.parse(value);
  if (!result.ok) throw new Error(result.error || "Storage operation failed");
  return result;
}

function selectedReceipt() {
  return document.getElementById(source.value)?.textContent || "";
}

function updateButtons() {
  if (document.body.dataset.etalonComponentStorage === "failed") return;
  saveButton.disabled = busy || !supported || !selectedReceipt();
  refreshButton.disabled = busy || !wasm;
  persistenceButton.disabled = busy || !wasm || !persistenceAvailable;
  source.disabled = busy;
}

async function run(action) {
  if (busy || !wasm) return;
  busy = true;
  updateButtons();
  try {
    await action();
  } catch (error) {
    status.textContent = String(error);
  } finally {
    busy = false;
    updateButtons();
  }
}

async function refreshCapabilities() {
  const result = parse(await wasm.storage_capabilities());
  if (result.source_revision !== revision) {
    throw new Error(
      "Page and storage WASM revisions differ; reload before storing receipts.",
    );
  }
  capabilities.textContent = JSON.stringify(result, null, 2);
  supported = result.opfs && result.web_locks;
  persistenceAvailable = result.persistence_request_available;
  return result;
}

function download(encoded, id) {
  const url = URL.createObjectURL(
    new Blob([encoded], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = `etalon-receipt-${id}`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function refreshReports() {
  const result = parse(await wasm.list_local_reports());
  reports.replaceChildren();
  if (!result.reports.length) {
    reports.textContent = "No receipts saved in this browser origin.";
    return;
  }
  const list = document.createElement("ul");
  for (const item of result.reports) {
    const row = document.createElement("li");
    const text = document.createElement("p");
    text.textContent = `${item.schema || "Invalid receipt"} · ${item.bytes} bytes · ${item.source_revision || "unknown revision"} · ${item.id}`;
    row.append(text);
    if (item.validation_error) {
      const error = document.createElement("p");
      error.textContent = item.validation_error;
      row.append(error);
    }
    const downloadButton = document.createElement("button");
    downloadButton.type = "button";
    downloadButton.textContent = "Download receipt";
    downloadButton.disabled = Boolean(item.validation_error);
    downloadButton.addEventListener("click", () =>
      run(async () => {
        const loaded = parse(await wasm.load_local_report(item.id));
        download(loaded.encoded, item.id);
        status.textContent = `Downloaded ${item.id}; stored metadata preserved.`;
      }),
    );
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.textContent = "Delete receipt";
    deleteButton.addEventListener("click", () =>
      run(async () => {
        parse(await wasm.delete_local_report(item.id));
        await refreshReports();
        await refreshCapabilities();
        status.textContent = `Deleted ${item.id} from this browser origin.`;
      }),
    );
    row.append(downloadButton, document.createTextNode(" "), deleteButton);
    list.append(row);
  }
  reports.append(list);
}

saveButton.addEventListener("click", () =>
  run(async () => {
    status.textContent = "Saving and verifying receipt in browser storage…";
    const saved = parse(await wasm.save_local_report(selectedReceipt()));
    await refreshReports();
    await refreshCapabilities();
    status.textContent = `Saved ${saved.id} (${saved.bytes} bytes); exact JSON read-back verified. This is storage evidence, not GPU qualification.`;
  }),
);
refreshButton.addEventListener("click", () =>
  run(async () => {
    await refreshCapabilities();
    await refreshReports();
    status.textContent = "Storage capabilities and local receipts refreshed.";
  }),
);
persistenceButton.addEventListener("click", () =>
  run(async () => {
    const result = parse(await wasm.request_storage_persistence());
    await refreshCapabilities();
    status.textContent = result.persisted
      ? "Browser reports persistent storage granted. Clearing site data still removes saved receipts."
      : "Browser did not grant persistent storage; saved receipts may be evicted.";
  }),
);
source.addEventListener("change", updateButtons);
const observer = new MutationObserver(updateButtons);
for (const id of ["qualification-report", "adapter-diagnostic-report"]) {
  const output = document.getElementById(id);
  if (output)
    observer.observe(output, {
      childList: true,
      characterData: true,
      subtree: true,
    });
}

async function initialize() {
  if (document.body.dataset.etalonComponentStorage === "failed") return;
  updateButtons();
  try {
    const module = await loadRuntime();
    wasm = module;
    await refreshCapabilities();
    if (supported) {
      await refreshReports();
      status.textContent =
        "Local receipt vault ready. Save a completed report explicitly; nothing is uploaded.";
    } else {
      status.textContent =
        "Local receipt vault unavailable in this browser context. Capability errors are shown above.";
    }
  } catch (error) {
    supported = false;
    wasm = undefined;
    status.textContent = `Storage capability check failed: ${String(error)}`;
  }
  updateButtons();
}
initialize();
