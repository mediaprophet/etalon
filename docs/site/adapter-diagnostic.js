import { loadRuntime } from "./wasm-runtime.js";
import { appEnvironment } from "./app-environment.js";

const runButton = document.querySelector("#run-adapter-diagnostic");
const downloadButton = document.querySelector("#download-adapter-diagnostic");
const output = document.querySelector("#adapter-diagnostic-report");
const status = document.querySelector("#adapter-diagnostic-status");
const suiteButton = document.querySelector("#run-tests");
const revision = document.querySelector('meta[name="etalon-revision"]').content;
let reportUrl;
let active = false;

async function boundedWait(promise, ms) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(
          () =>
            reject(new Error("Diagnostic timed out; reload before retrying")),
          ms,
        );
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

runButton.addEventListener("click", async () => {
  if (
    active ||
    suiteButton.disabled ||
    document.body.dataset.etalonState === "running"
  )
    return;
  active = true;
  runButton.disabled = true;
  suiteButton.disabled = true;
  document.body.dataset.etalonDiagnostic = "running";
  downloadButton.disabled = true;
  if (reportUrl) URL.revokeObjectURL(reportUrl);
  output.textContent = "";
  status.textContent = "Running individual Rust/WASM adapter checks…";
  let report;
  let failedToComplete = false;
  const started = new Date().toISOString();
  try {
    const wasm = await boundedWait(loadRuntime(), 15000);
    report = JSON.parse(
      await boundedWait(wasm.adapter_diagnostic_errors(), 15000),
    );
    if (report.source_revision !== revision)
      throw new Error("Page and WASM source revisions differ");
    report.started_at = started;
    report.finished_at = new Date().toISOString();
    report.environment = {
      app: appEnvironment(),
      secure_context: window.isSecureContext,
      browser: navigator.userAgent,
    };
    try {
      report.environment.storage = JSON.parse(
        await boundedWait(wasm.storage_capabilities(), 5000),
      );
    } catch (error) {
      report.environment.storage = {
        status: "unavailable",
        message: String(error).slice(0, 512),
      };
    }
    try {
      report.environment.browser_details = JSON.parse(
        await boundedWait(wasm.browser_environment_json(), 5000),
      );
    } catch (error) {
      report.errors.push({
        test_id: "environment",
        stage: "metadata",
        message: String(error).slice(0, 512),
      });
    }
    report.error_count = report.errors.length;
    status.textContent = report.errors.length
      ? `${report.errors.length} diagnostic errors. No GPU workload was run.`
      : "No adapter acquisition errors detected. GPU execution remains untested.";
  } catch (error) {
    failedToComplete = true;
    report = {
      schema: "etalon.adapter-diagnostic.v1",
      source_revision: revision,
      started_at: started,
      finished_at: new Date().toISOString(),
      gpu_execution_tested: false,
      automatic_upload: false,
      environment: {
        app: appEnvironment(),
        secure_context: window.isSecureContext,
        browser: navigator.userAgent,
      },
      errors: [
        {
          test_id: "diagnostic-runner",
          stage: "load-or-run",
          message: String(error).slice(0, 2000),
        },
      ],
    };
    status.textContent =
      "Diagnostic could not complete. Error details below; reload before retrying.";
  }
  report.error_count = report.errors.length;
  report.finished_at = new Date().toISOString();
  const text = JSON.stringify(report, null, 2);
  output.textContent = text;
  reportUrl = URL.createObjectURL(
    new Blob([text], { type: "application/json" }),
  );
  downloadButton.disabled = false;
  document.body.dataset.etalonDiagnostic = "complete";
  active = false;
  runButton.disabled = failedToComplete;
  suiteButton.disabled = failedToComplete;
});

downloadButton.addEventListener("click", () => {
  if (!reportUrl) return;
  const link = document.createElement("a");
  link.href = reportUrl;
  link.download = `etalon-${revision.slice(0, 12)}-adapter-errors.json`;
  link.click();
});
window.addEventListener("pagehide", () => {
  if (reportUrl) URL.revokeObjectURL(reportUrl);
});
