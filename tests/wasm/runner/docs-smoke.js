import { chromium } from "playwright";

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
  if (boundary) {
    await page.addInitScript(
      ({ mode, rejectionMessage }) => {
        window.__etalonAdapterRequests = [];
        const requestAdapter = (options) => {
          window.__etalonAdapterRequests.push({
            options,
            hasFeatureLevel: Object.hasOwn(options, "featureLevel"),
            hasForceFallbackAdapter: Object.hasOwn(
              options,
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
  const completion = await page.evaluate(() => ({ ...document.body.dataset }));
  const allPassed =
    completion.etalonPassed === "13" &&
    completion.etalonFailed === "0" &&
    completion.etalonUnsupported === "0" &&
    completion.etalonNotRun === "0";
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
    if (observed.requests.length !== 1)
      throw new Error("Rust/WASM did not make exactly one adapter request");
    const request = observed.requests[0];
    const expectedFeatureLevel = featureLevel === "compatibility";
    if (
      request.hasFeatureLevel !== expectedFeatureLevel ||
      (expectedFeatureLevel &&
        request.options.featureLevel !== "compatibility") ||
      request.hasForceFallbackAdapter ||
      request.options.powerPreference !== "low-power"
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
      observed.rows.length !== 13 ||
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
  const document = await page.content();
  if (Buffer.byteLength(document) > 2 * 1024 * 1024)
    throw new Error("Browser DOM budget exceeded");
  process.stdout.write(document);
} catch (error) {
  throw new Error(`${error}; browser errors: ${messages.join("; ")}`);
} finally {
  await context.close();
}
