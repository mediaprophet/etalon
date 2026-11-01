import { chromium } from "playwright";
import { runBackendContracts } from "./backend-contracts.js";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, value, index, all) => {
    if (index % 2 === 0) pairs.push([value, all[index + 1]]);
    return pairs;
  }, []),
);
const mode = args["--mode"];
const featureLevel = args["--feature-level"] || "core";
if (
  ![
    "default-availability",
    "software-webgpu",
    "adapter-null",
    "adapter-rejected",
  ].includes(mode)
)
  throw new Error("Unknown browser qualification mode");
if (!["core", "compatibility"].includes(featureLevel))
  throw new Error("Unknown feature level");
for (const name of ["--chrome", "--profile", "--url"])
  if (!args[name]) throw new Error(`Missing ${name}`);

const browserArgs = [
  "--disable-background-networking",
  "--disable-component-update",
  "--disable-extensions",
  "--disable-sync",
  "--disk-cache-size=1048576",
  "--media-cache-size=1048576",
];
if (mode === "software-webgpu")
  browserArgs.push(
    "--enable-unsafe-webgpu",
    "--use-webgpu-adapter=swiftshader",
    "--use-gpu-in-tests",
  );
const context = await chromium.launchPersistentContext(args["--profile"], {
  executablePath: args["--chrome"],
  headless: true,
  viewport: { width: 1280, height: 900 },
  args: browserArgs,
});
const messages = [];
const boundary = mode === "adapter-null" || mode === "adapter-rejected";
const rejectionMessage = "etalon-injected-adapter-rejection";
try {
  const page = context.pages()[0] || (await context.newPage());
  page.on("pageerror", (error) => {
    if (messages.length < 8) messages.push(String(error).slice(0, 1000));
  });
  if (mode === "software-webgpu") {
    await page.addInitScript(() => {
      window.__etalonCapabilityAcquisitions = 0;
      const acquireTexture = GPUCanvasContext.prototype.getCurrentTexture;
      GPUCanvasContext.prototype.getCurrentTexture = function (...args) {
        if (document.body.dataset.etalonCase === "browser-canvas-capabilities")
          window.__etalonCapabilityAcquisitions++;
        return acquireTexture.apply(this, args);
      };
      window.__etalonDeviceDestruction = [];
      window.__etalonDeviceLoss = [];
      const requestDevice = GPUAdapter.prototype.requestDevice;
      GPUAdapter.prototype.requestDevice = async function (...args) {
        const device = await requestDevice.apply(this, args);
        device.lost.then((info) => {
          if (window.__etalonDeviceLoss.length < 8)
            window.__etalonDeviceLoss.push({
              reason: info.reason,
              message: info.message,
              stage: document.querySelector("#test-progress")?.textContent,
            });
        });
        return device;
      };
      const original = GPUDevice.prototype.destroy;
      GPUDevice.prototype.destroy = function (...args) {
        if (window.__etalonDeviceDestruction.length < 4)
          window.__etalonDeviceDestruction.push(
            String(document.querySelector("#test-progress")?.textContent) +
              "\n" +
              String(new Error("device.destroy").stack).slice(0, 3000),
          );
        return original.apply(this, args);
      };
    });
  }
  if (boundary) {
    await page.addInitScript(
      ({ mode, rejectionMessage }) => {
        window.__etalonAdapterRequests = [];
        const requestAdapter = (...args) => {
          const options = args[0];
          window.__etalonAdapterRequests.push({
            options,
            argumentCount: args.length,
            hasFeatureLevel: Object.hasOwn(options || {}, "featureLevel"),
            hasForceFallbackAdapter: Object.hasOwn(
              options || {},
              "forceFallbackAdapter",
            ),
          });
          return mode === "adapter-null"
            ? Promise.resolve(null)
            : Promise.reject(
                new DOMException(rejectionMessage, "OperationError"),
              );
        };
        if (navigator.gpu) {
          Object.defineProperty(navigator.gpu, "requestAdapter", {
            value: requestAdapter,
          });
        } else {
          Object.defineProperty(navigator, "gpu", {
            value: { requestAdapter },
          });
        }
      },
      { mode, rejectionMessage },
    );
  }
  await page.goto(args["--url"], { waitUntil: "load", timeout: 30000 });
  const selector = page.locator("#feature-level");
  if ((await selector.count()) > 0) await selector.selectOption(featureLevel);
  else if (featureLevel !== "core") throw new Error("Feature selector missing");
  await page.locator("#run-tests").click();
  await page.waitForFunction(
    () => document.body.dataset.etalonState === "complete",
    null,
    { timeout: 60000 },
  );
  const report = JSON.parse(
    await page.locator("#qualification-report").textContent(),
  );
  const debug = report.debug_log;
  if (
    !report.debug_logging_enabled ||
    !report.debug_environment_attached ||
    debug.schema !== "etalon.debug-log.v1" ||
    debug.source_revision !== report.source_revision ||
    debug.library_version !== report.runtime.library_version ||
    debug.environment?.browser !== report.context.browser ||
    !debug.environment?.browser_details ||
    debug.environment.browser_details.navigator.user_agent.value !==
      report.context.browser ||
    debug?.status !== "enabled" ||
    debug.level !== "debug" ||
    debug.automatic_upload !== false ||
    !Array.isArray(debug.events) ||
    debug.events.length === 0 ||
    debug.events.length > debug.max_events ||
    new TextEncoder().encode(JSON.stringify(debug)).length >
      debug.max_json_bytes ||
    debug.events.some(
      (e) =>
        new TextEncoder().encode(e.message).length > debug.max_message_bytes ||
        new TextEncoder().encode(e.target).length > debug.max_target_bytes,
    ) ||
    !debug.events.some(
      (e) =>
        e.target.includes("backend::webgpu") &&
        e.message.includes("cached_gpu=true"),
    )
  )
    throw new Error(
      "Bounded Rust debug capture did not include actual backend acquisition",
    );
  if (
    boundary &&
    !debug.events.some((e) => e.level === "ERROR" || e.level === "WARN")
  )
    throw new Error(
      "Failed adapter acquisition did not produce an error/debug record",
    );
  const completion = await page.evaluate(() => ({ ...document.body.dataset }));
  const allPassed =
    completion.etalonPassed === "20" &&
    completion.etalonFailed === "0" &&
    completion.etalonUnsupported === "0" &&
    completion.etalonNotRun === "0";
  if (!allPassed && mode === "software-webgpu") {
    const destruction = await page.evaluate(() => ({
      calls: window.__etalonDeviceDestruction,
      losses: window.__etalonDeviceLoss,
    }));
    process.stderr.write(
      `Device destruction calls: ${JSON.stringify(destruction)}\n`,
    );
  }
  if (
    mode === "software-webgpu" &&
    (await page.evaluate(() => window.__etalonCapabilityAcquisitions !== 0))
  )
    throw new Error("Canvas capability query acquired a presentation texture");
  const expectedQualification = allPassed ? "passed" : "failed";
  if (completion.etalonQualification !== expectedQualification)
    throw new Error(
      "Overall qualification is missing or contradicts actual case results",
    );
  if (mode === "software-webgpu" && completion.etalonQualification !== "passed")
    throw new Error(
      `Software WebGPU qualification failed: ${JSON.stringify(completion).slice(0, 16000)}`,
    );
  if (boundary && completion.etalonQualification !== "failed")
    throw new Error(
      "Synthetic unavailable/rejected adapter was incorrectly qualified",
    );
  process.stderr.write(
    `Overall qualification: ${completion.etalonQualification}\n`,
  );
  const environment = await page.locator("#environment").innerText();
  process.stderr.write(environment.slice(0, 16000) + "\n");
  if (boundary) {
    const observed = await page.evaluate(() => ({
      requests: window.__etalonAdapterRequests,
      rows: Array.from(
        document.querySelectorAll("#test-results tr"),
        (row) => ({
          status: row.querySelector(".status").textContent,
          details: JSON.parse(row.querySelector("pre").textContent),
        }),
      ),
    }));
    const expectedStatus = mode === "adapter-null" ? "unsupported" : "fail";
    const expectedCode =
      mode === "adapter-null"
        ? "adapter-unavailable"
        : "adapter-request-rejected";
    const runtime = JSON.parse(environment);
    if (observed.requests.length !== 5)
      throw new Error(
        "Expected one Etalon request followed by four independent controls",
      );
    const control = runtime.adapter_acquisition_control;
    const expectedControlStatus = mode === "adapter-null" ? "null" : "rejected";
    if (
      control?.gpu_execution_tested !== false ||
      control.attempts?.length !== 4 ||
      control.attempts.some((a) => a.result.status !== expectedControlStatus)
    )
      throw new Error(
        "Independent Rust/WASM controls did not preserve browser outcomes",
      );
    const controls = observed.requests.slice(1);
    if (
      controls[0].argumentCount !== 0 ||
      controls[1].argumentCount !== 1 ||
      Object.keys(controls[1].options).length !== 0 ||
      JSON.stringify(controls[2].options) !== '{"featureLevel":"core"}' ||
      JSON.stringify(controls[3].options) !== '{"featureLevel":"compatibility"}'
    )
      throw new Error(
        "Independent Rust/WASM request arguments differ from their labels",
      );
    const request = observed.requests[0];
    const expectedFeatureLevel = featureLevel === "compatibility";
    if (
      request.hasFeatureLevel !== expectedFeatureLevel ||
      (expectedFeatureLevel &&
        request.options.featureLevel !== "compatibility") ||
      request.hasForceFallbackAdapter ||
      Object.hasOwn(request.options, "powerPreference")
    )
      throw new Error(
        `Rust/WASM adapter request differs from the expected upstream shape: ${JSON.stringify(request)}`,
      );
    if (
      runtime.gpu_status !== expectedStatus ||
      runtime.gpu_reason?.code !== expectedCode
    )
      throw new Error(
        "Rust/WASM adapter failure classification was not preserved",
      );
    if (
      mode === "adapter-rejected" &&
      !runtime.gpu_reason.message.includes(rejectionMessage)
    )
      throw new Error("Rust/WASM discarded the browser rejection message");
    if (
      observed.rows.length !== 20 ||
      observed.rows.slice(0, 2).some((r) => r.status !== "pass")
    )
      throw new Error("Portable Rust/WASM cases did not execute");
    for (const result of observed.rows.slice(2)) {
      if (
        result.status !== expectedStatus ||
        result.details.code !== expectedCode
      )
        throw new Error("GPU case did not retain the expected adapter result");
      if (
        mode === "adapter-rejected" &&
        !result.details.message.includes(rejectionMessage)
      )
        throw new Error("GPU case discarded the browser rejection message");
    }
    process.stderr.write(
      `Synthetic adapter boundary verified: ${mode}; no GPU execution claimed\n`,
    );
  }
  if (mode === "software-webgpu") {
    const contracts = await runBackendContracts(page);
    process.stderr.write(`Backend contracts: ${JSON.stringify(contracts)}\n`);
  }
  const document = await page.content();
  if (Buffer.byteLength(document) > 2 * 1024 * 1024)
    throw new Error("Browser DOM budget exceeded");
  process.stdout.write(document);
} catch (error) {
  throw new Error(`${error}; browser errors: ${messages.join("; ")}`);
} finally {
  await context.close();
}
