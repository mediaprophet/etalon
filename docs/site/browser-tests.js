const start = document.querySelector("#run-tests");
const download = document.querySelector("#download-report");
const progress = document.querySelector("#test-progress");
const summary = document.querySelector("#test-summary");
const environment = document.querySelector("#environment");
const tbody = document.querySelector("#test-results");
const revision = document.querySelector('meta[name="etalon-revision"]').content;
let active = false;
let blocked = false;
let reportUrl;

function parse(value) {
  return typeof value === "string" ? JSON.parse(value) : value;
}
function bounded(value) {
  return JSON.stringify(value, null, 2).slice(0, 12000);
}
function deadline(promise, milliseconds) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      const error = new Error(
        "Timed out. Reload before another run; submitted GPU work may still be pending.",
      );
      error.code = "timeout";
      reject(error);
    }, milliseconds);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}
function row(test) {
  const tr = document.createElement("tr");
  const label = document.createElement("td");
  label.textContent = test.label || test.id;
  const status = document.createElement("td");
  status.className = "status not_run";
  status.textContent = "Not run";
  const time = document.createElement("td");
  time.textContent = "—";
  const evidence = document.createElement("td");
  tr.append(label, status, time, evidence);
  tbody.append(tr);
  return { status, time, evidence };
}
function updateRow(view, result) {
  view.status.textContent =
    result.status === "not_run" ? "Not run" : result.status;
  view.status.className = `status ${result.status}`;
  view.time.textContent = Number.isFinite(result.duration_ms)
    ? `${result.duration_ms.toFixed(2)} ms`
    : "—";
  const details = document.createElement("details");
  const title = document.createElement("summary");
  title.textContent = "Details";
  const pre = document.createElement("pre");
  pre.textContent = bounded(result.details || {});
  details.append(title, pre);
  view.evidence.replaceChildren(details);
}
function totals(report) {
  const counts = { pass: 0, fail: 0, unsupported: 0, not_run: 0 };
  for (const result of report.results) counts[result.status] += 1;
  summary.textContent = `${counts.pass} passed · ${counts.fail} failed · ${counts.unsupported} unsupported · ${counts.not_run} not run`;
  return counts;
}
function preview(session) {
  const bytes = session.image_rgba();
  const caption = document.querySelector("#render-caption");
  if (bytes.length !== 64 * 64 * 4)
    throw new Error("Unexpected render readback size.");
  const context = document.querySelector("#render-result").getContext("2d");
  if (context) {
    context.putImageData(
      new ImageData(new Uint8ClampedArray(bytes), 64, 64),
      0,
      0,
    );
    caption.textContent =
      "64 × 64 GPU-rendered pixels, read back and checked before display.";
  } else {
    caption.textContent =
      "Readback verified; this browser could not display the 2D preview.";
  }
}
function exportReport(report) {
  const json = JSON.stringify(report, null, 2);
  if (new TextEncoder().encode(json).length > 262144) {
    progress.textContent += " Report exceeds the export byte limit.";
    return;
  }
  reportUrl = URL.createObjectURL(
    new Blob([json], { type: "application/json" }),
  );
  download.disabled = false;
}
async function run() {
  if (active || blocked) return;
  active = true;
  start.disabled = true;
  download.disabled = true;
  if (reportUrl) URL.revokeObjectURL(reportUrl);
  reportUrl = undefined;
  tbody.replaceChildren();
  document.body.dataset.etalonState = "running";
  const canvas = document.querySelector("#render-result");
  canvas.getContext("2d")?.clearRect(0, 0, 64, 64);
  document.querySelector("#render-caption").textContent =
    "No verified render from this run yet.";
  const report = {
    schema: "etalon.browser-suite.v1",
    source_revision: revision,
    started_at: new Date().toISOString(),
    context: {
      secure_context: window.isSecureContext,
      webgpu_exposed: Boolean(navigator.gpu),
      browser: navigator.userAgent,
      viewport: {
        width: innerWidth,
        height: innerHeight,
        scale: devicePixelRatio,
      },
    },
    results: [],
  };
  let session;
  let inFlight;
  try {
    progress.textContent = "Loading the Etalon Rust/WASM build…";
    const module = await deadline(
      import("./browser/etalon_browser_demo.js?rev=" + revision),
      30000,
    );
    await deadline(
      module.default({
        module_or_path: new URL(
          "./browser/etalon_browser_demo_bg.wasm?rev=" + revision,
          import.meta.url,
        ),
      }),
      30000,
    );
    const selectedProfile = document.querySelector("#feature-level").value;
    const creating = module.create_session_for_profile(selectedProfile);
    try {
      session = await deadline(creating, 30000);
    } catch (error) {
      creating
        .then((late) => {
          late.destroy();
          late.free();
        })
        .catch(() => {});
      throw error;
    }
    report.runtime = parse(session.info_json());
    if (report.runtime.source_revision !== revision) {
      throw new Error(
        "The WASM build revision does not match this page. Reload to avoid testing stale assets.",
      );
    }
    environment.textContent = bounded({ ...report.context, ...report.runtime });
    const catalog = parse(session.cases_json());
    if (!Array.isArray(catalog) || catalog.length < 2 || catalog.length > 16) {
      throw new Error("Invalid bounded test catalog.");
    }
    const views = catalog.map(row);
    for (let i = 0; i < catalog.length; i += 1) {
      const test = catalog[i];
      progress.textContent = `Running ${i + 1}/${catalog.length}: ${test.label || test.id}`;
      let result;
      try {
        inFlight = session.run_case(test.id);
        const value = await deadline(inFlight, 30000);
        inFlight = undefined;
        result = parse(value);
        if (
          result.id !== test.id ||
          !["pass", "fail", "unsupported"].includes(result.status)
        ) {
          throw new Error("Invalid test result.");
        }
      } catch (error) {
        result = {
          id: test.id,
          status: "fail",
          details: {
            code: error.code || "exception",
            message: String(error).slice(0, 2000),
          },
        };
        if (error.code === "timeout") blocked = true;
      }
      report.results.push(result);
      updateRow(views[i], result);
      totals(report);
      if (test.id === "render-readback" && result.status === "pass") {
        try {
          preview(session);
        } catch (error) {
          result.status = "fail";
          result.details = {
            code: "preview-contract",
            message: String(error).slice(0, 2000),
          };
          updateRow(views[i], result);
        }
      }
      if (blocked) {
        for (let j = i + 1; j < catalog.length; j += 1) {
          const pending = {
            id: catalog[j].id,
            status: "not_run",
            details: { reason: "Earlier timeout; owner destroyed." },
          };
          report.results.push(pending);
          updateRow(views[j], pending);
        }
        break;
      }
    }
    report.counts = totals(report);
    progress.textContent = blocked
      ? "Run timed out. Reload before another attempt."
      : "Run complete. Unsupported cases are not passes.";
  } catch (error) {
    if (error.code === "timeout") blocked = true;
    report.results.push({
      id: "initialization",
      status: "fail",
      details: { message: String(error).slice(0, 2000) },
    });
    report.counts = totals(report);
    progress.textContent = `Could not complete the suite: ${String(error).slice(0, 2000)}`;
    environment.textContent = bounded(report.context);
  } finally {
    if (session) {
      try {
        session.destroy();
        if (inFlight && blocked) {
          // The async Rust borrow must settle before its wrapper can be freed.
          inFlight.finally(() => session.free()).catch(() => {});
        } else {
          session.free();
        }
      } catch (error) {
        blocked = true;
        report.results.push({
          id: "cleanup",
          status: "fail",
          details: { message: String(error).slice(0, 2000) },
        });
      }
    }
    report.finished_at = new Date().toISOString();
    report.counts = totals(report);
    document.body.dataset.etalonState = "complete";
    document.body.dataset.etalonPassed = String(report.counts.pass);
    document.body.dataset.etalonFailed = String(report.counts.fail);
    document.body.dataset.etalonUnsupported = String(report.counts.unsupported);
    document.body.dataset.etalonNotRun = String(report.counts.not_run);
    document.body.dataset.etalonErrors = JSON.stringify(
      report.results
        .filter((r) => r.status === "fail")
        .map((r) => ({ id: r.id, details: r.details })),
    ).slice(0, 12000);
    exportReport(report);
    active = false;
    start.disabled = blocked;
  }
}
start.addEventListener("click", run);
download.addEventListener("click", () => {
  if (!reportUrl) return;
  const link = document.createElement("a");
  link.href = reportUrl;
  link.download = `etalon-${revision.slice(0, 12)}-browser-report.json`;
  link.hidden = true;
  document.body.append(link);
  link.click();
  link.remove();
});
window.addEventListener("pagehide", () => {
  if (reportUrl) URL.revokeObjectURL(reportUrl);
});
if (new URLSearchParams(location.search).get("autorun") === "1") run();
