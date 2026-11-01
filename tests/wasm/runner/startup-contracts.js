function assert(condition, message) {
  if (!condition) throw new Error(message);
}

export async function runStartupContracts(context, url) {
  const browser = context.browser();
  assert(browser, "Startup contracts require a browser-owned context");
  const target = new URL(url);
  target.search = "?autorun=0";
  const results = [];
  for (const fault of [
    "delayed-suite",
    "http503-suite",
    "aborted-camera",
    "hung-suite",
  ]) {
    const isolated = await browser.newContext({ serviceWorkers: "block" });
    const page = await isolated.newPage();
    const failures = [];
    const errors = [];
    let release;
    try {
      page.on("requestfailed", (request) =>
        failures.push({
          url: request.url(),
          error: request.failure()?.errorText,
        }),
      );
      page.on("response", (response) => {
        if (response.status() >= 400)
          failures.push({ url: response.url(), status: response.status() });
      });
      page.on("pageerror", (error) => errors.push(String(error)));
      const path =
        fault === "aborted-camera" ? "browser-camera.js" : "browser-tests.js";
      const gated = new Promise((resolve) => {
        release = resolve;
      });
      await page.route(`**/${path}`, async (route) => {
        if (fault === "delayed-suite" || fault === "hung-suite") {
          await gated;
          await route.continue();
        } else if (fault === "http503-suite")
          await route.fulfill({
            status: 503,
            contentType: "text/plain",
            body: "Injected startup transport failure",
          });
        else await route.abort("connectionreset");
      });
      await page.goto(target.href, { waitUntil: "load" });
      if (fault === "delayed-suite") {
        assert(
          await page.locator("#run-tests").isDisabled(),
          "Run became clickable before its module initialized",
        );
        assert(
          (await page.locator("body").getAttribute("data-etalon-ready")) !==
            "true",
          "Delayed module falsely marked the suite ready",
        );
        release();
      }
      await page.waitForFunction(
        () => ["ready", "failed"].includes(document.body.dataset.etalonStartup),
        null,
        { timeout: 20000 },
      );
      if (fault === "hung-suite") {
        assert(
          await page.locator("#run-tests").isDisabled(),
          "Timed-out module enabled its controls",
        );
        release();
        await page.waitForFunction(
          () =>
            JSON.parse(
              document.querySelector("#app-startup-report").textContent,
            ).components.find((entry) => entry.id === "suite").late_loaded ===
            true,
        );
      }
      const evidence = JSON.parse(
        await page.locator("#app-startup-report").textContent(),
      );
      assert(
        evidence.schema === "etalon.app-startup.v1" &&
          /^[0-9a-f]{40}$/.test(evidence.source_revision) &&
          evidence.library_version === "0.0.2",
        "Startup receipt is missing exact build identity",
      );
      const entry = evidence.components.find((item) => item.path === path);
      if (fault === "delayed-suite") {
        assert(
          entry.status === "loaded" &&
            !(await page.locator("#run-tests").isDisabled()),
          "Successful delayed loading did not enable the actual listener",
        );
        assert(
          (await page.locator("body").getAttribute("data-etalon-ready")) ===
            "true",
          "Resolved module is not actually ready",
        );
      } else {
        assert(
          entry.status === "failed" &&
            entry.error?.url.endsWith(path) &&
            entry.error.message,
          "Module failure lacks actionable path/error evidence",
        );
        assert(
          fault === "hung-suite" ||
            failures.some((failure) => failure.url.endsWith(path)),
          "The transport failure was not captured by the test collector",
        );
        const affected =
          fault === "aborted-camera" ? "#camera-start" : "#run-tests";
        assert(
          await page.locator(affected).isDisabled(),
          "Failed component retained an inert enabled control",
        );
        if (fault === "aborted-camera") {
          assert(
            !(await page.locator("#run-tests").isDisabled()),
            "Optional camera startup failure disabled a working independent suite",
          );
        } else {
          assert(
            fault === "hung-suite"
              ? entry.error.name === "ModuleLoadTimeout"
              : failures.some((failure) => failure.status === 503),
            "HTTP startup status was lost",
          );
          assert(
            (await page.locator("body").getAttribute("data-etalon-ready")) ===
              "false",
            "Failed suite remained marked ready",
          );
        }
        assert(
          errors.length === 0,
          "Caught resource failure unexpectedly surfaced as an uncaught execution error",
        );
      }
      results.push({ fault, status: "verified", module_state: entry.status });
    } finally {
      release?.();
      await isolated.close();
    }
  }
  return results;
}
