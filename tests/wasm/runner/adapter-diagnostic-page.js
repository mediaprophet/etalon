export async function runAdapterDiagnosticPage(context, url, revision) {
  const page = await context.newPage();
  const wasmRequests = [];
  page.on("request", (request) => {
    if (new URL(request.url()).pathname.endsWith("etalon_browser_demo_bg.wasm"))
      wasmRequests.push(request.url());
  });
  try {
    await page.addInitScript(() => {
      const gpu = { requestAdapter: () => Promise.resolve(null) };
      Object.defineProperty(navigator, "gpu", {
        configurable: true,
        value: gpu,
      });
    });
    await page.goto(url, { waitUntil: "load" });
    await page.locator("#run-adapter-diagnostic").click();
    await page.waitForFunction(
      () => document.body.dataset.etalonDiagnostic === "complete",
    );
    const result = JSON.parse(
      await page.locator("#adapter-diagnostic-report").textContent(),
    );
    if (
      result.source_revision !== revision ||
      result.library_version !== "0.0.2" ||
      result.error_count !== 5 ||
      result.errors.length !== 5 ||
      result.errors.some(
        (e) => e.status !== "null" || !e.test_id || !e.api_path,
      ) ||
      result.gpu_execution_tested !== false ||
      !result.environment?.browser ||
      (await page.locator("body").getAttribute("data-etalon-qualification")) !==
        "not_run" ||
      (await page.locator("#download-adapter-diagnostic").isDisabled())
    )
      throw new Error(
        "Errors-only diagnostic page did not preserve failure identities or isolated qualification state",
      );
    if (
      !wasmRequests.length ||
      wasmRequests.some((u) => new URL(u).searchParams.get("rev") !== revision)
    )
      throw new Error(
        "Diagnostic-first initialization did not load revision-keyed WASM",
      );
    return {
      mode: "errors-only-page",
      errors: result.error_count,
      version_verified: true,
      revision_keyed_wasm: true,
      full_suite_not_run: true,
    };
  } finally {
    await page.close();
  }
}
