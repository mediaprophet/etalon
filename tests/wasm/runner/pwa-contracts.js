function check(condition, message) {
  if (!condition) throw new Error(message);
}

async function storage(page, revision, method, argument) {
  return page.evaluate(
    async ({ revision, method, argument }) => {
      const { loadRuntime } = await import("./wasm-runtime.js");
      const module = await loadRuntime();
      return JSON.parse(
        await module[method](...(argument === undefined ? [] : [argument])),
      );
    },
    { revision, method, argument },
  );
}

async function ready(page, revision) {
  await page.waitForFunction(
    (revision) => {
      const body = document.body.dataset;
      const capabilities = document.querySelector(
        "#storage-capabilities",
      ).textContent;
      return (
        body.etalonOfflineReady === "true" &&
        body.etalonOfflineRevision === revision &&
        navigator.serviceWorker.controller &&
        !document.querySelector("#refresh-local-reports").disabled &&
        capabilities &&
        JSON.parse(capabilities).source_revision === revision
      );
    },
    revision,
    { timeout: 30000 },
  );
}

async function manifestEvidence(context, page) {
  const cdp = await context.newCDPSession(page);
  try {
    await cdp.send("Page.enable");
    const manifest = await cdp.send("Page.getAppManifest");
    check(
      !manifest.errors?.some((error) => error.critical),
      "PWA manifest contains a critical parsing error",
    );
    const data = JSON.parse(manifest.data);
    check(
      data.id === "./" &&
        data.scope === "./" &&
        data.display === "standalone" &&
        data.start_url === "./testing.html?app=1&autorun=0",
      "PWA manifest identity/display/start URL contract changed",
    );
    for (const size of [192, 512])
      check(
        data.icons?.some(
          (icon) =>
            icon.sizes === `${size}x${size}` && icon.type === "image/png",
        ),
        `PWA lacks a ${size}px PNG icon`,
      );
    let installability = { status: "unavailable", errors: [] };
    try {
      const result = await cdp.send("Page.getInstallabilityErrors");
      installability = {
        status: "reported",
        errors: result.installabilityErrors || [],
      };
      check(
        installability.errors.length === 0,
        `Browser reports PWA installability errors: ${JSON.stringify(installability.errors)}`,
      );
    } catch (error) {
      if (!/wasn't found|method not found|not supported/i.test(String(error)))
        throw error;
    }
    return { manifest_verified: true, installability };
  } finally {
    await cdp.detach();
  }
}

export async function runPwaContracts(context, url, revision) {
  const target = new URL(url);
  target.search = "?autorun=0&app=1";
  const page = await context.newPage();
  let peer;
  const owned = new Set();
  let baseline;
  let cleanupError;
  try {
    await page.goto(target.href, { waitUntil: "load" });
    await ready(page, revision);
    const manifest = await manifestEvidence(context, page);
    baseline = await storage(page, revision, "list_local_reports");
    check(
      baseline.ok && baseline.reports.length < 16,
      "Test origin lacks room for an owned receipt; existing reports preserved",
    );
    await context.setOffline(true);
    await page.reload({ waitUntil: "load" });
    await ready(page, revision);
    await page.locator("#run-tests").click();
    await page.waitForFunction(
      () => document.body.dataset.etalonState === "complete",
      null,
      { timeout: 45000 },
    );
    const encoded = await page.locator("#qualification-report").textContent();
    const receipt = JSON.parse(encoded);
    const app = receipt.context?.app;
    check(
      receipt.source_revision === revision &&
        receipt.counts?.pass === 21 &&
        receipt.counts?.fail === 0 &&
        receipt.qualification?.status === "passed",
      "Offline Rust/WASM suite did not execute all 21 cases successfully",
    );
    check(
      app?.source_revision === revision &&
        app.app_version === "0.0.2" &&
        app.online === false &&
        app.service_worker?.controlled === true &&
        typeof app.service_worker.controller_script_url === "string" &&
        app.offline_cache?.ready === true &&
        app.offline_cache.revision === revision,
      "Offline receipt omitted exact app/cache/controller evidence",
    );
    check(
      app.installed_display_mode === false && app.display_mode === "browser",
      "Headless browser test must not claim installed-mode qualification",
    );
    const caps = await storage(page, revision, "storage_capabilities");
    check(
      caps.ok &&
        caps.opfs &&
        caps.web_locks &&
        caps.storage_api &&
        caps.source_revision === revision &&
        caps.limits.max_reports === 16 &&
        caps.limits.max_report_bytes === 262144 &&
        caps.limits.max_total_bytes === 4194304,
      "Rust storage capabilities or bounds differ from the vault contract",
    );
    const beforeSave = await storage(page, revision, "list_local_reports");
    check(
      beforeSave.ok &&
        JSON.stringify(beforeSave.reports) === JSON.stringify(baseline.reports),
      "Running the offline suite unexpectedly saved a report without an explicit request",
    );
    const saved = await storage(page, revision, "save_local_report", encoded);
    if (saved.ok) owned.add(saved.id);
    check(
      saved.ok &&
        saved.verified === true &&
        saved.bytes === new TextEncoder().encode(encoded).length,
      `Actual offline receipt was not saved and verified: ${JSON.stringify(saved)}`,
    );
    const loaded = await storage(page, revision, "load_local_report", saved.id);
    check(
      loaded.ok && loaded.encoded === encoded,
      "OPFS round-trip changed the actual suite receipt",
    );
    for (const invalid of [
      {
        method: "save_local_report",
        value: "x".repeat(262145),
        reason: "oversize",
      },
      {
        method: "save_local_report",
        value: JSON.stringify({ schema: "corrupt", source_revision: revision }),
        reason: "schema",
      },
      {
        method: "load_local_report",
        value: "../outside.json",
        reason: "unsafe-load-id",
      },
      {
        method: "delete_local_report",
        value: "../outside.json",
        reason: "unsafe-delete-id",
      },
    ]) {
      const rejected = await storage(
        page,
        revision,
        invalid.method,
        invalid.value,
      );
      check(
        rejected.ok === false && typeof rejected.error === "string",
        `Vault accepted ${invalid.reason}`,
      );
    }
    await page.evaluate(async (id) => {
      const root = await navigator.storage.getDirectory();
      const vault = await root.getDirectoryHandle("etalon-receipts-v1");
      const file = await vault.getFileHandle(id);
      const writer = await file.createWritable();
      await writer.write('{"schema":"corrupted"}');
      await writer.close();
    }, saved.id);
    const corrupted = await storage(
      page,
      revision,
      "load_local_report",
      saved.id,
    );
    const listedCorruption = await storage(
      page,
      revision,
      "list_local_reports",
    );
    check(
      corrupted.ok === false &&
        listedCorruption.ok &&
        listedCorruption.reports.find((entry) => entry.id === saved.id)
          ?.validation_error,
      "Corrupted on-disk receipt was not rejected and surfaced by the Rust vault",
    );
    const removed = await storage(
      page,
      revision,
      "delete_local_report",
      saved.id,
    );
    check(removed.ok, "Could not remove the owned corruption fixture");
    owned.delete(saved.id);
    const fixture = JSON.stringify({
      schema: "etalon.browser-suite.v1",
      source_revision: revision,
      fixture: "capacity-only-not-a-qualification",
    });
    while (baseline.reports.length + owned.size < 15) {
      const result = await storage(
        page,
        revision,
        "save_local_report",
        fixture,
      );
      if (result.ok) owned.add(result.id);
      check(
        result.ok,
        `Could not fill owned capacity fixtures: ${JSON.stringify(result)}`,
      );
    }
    peer = await context.newPage();
    await peer.goto(target.href, { waitUntil: "load" });
    await ready(peer, revision);
    const concurrent = await Promise.allSettled([
      storage(page, revision, "save_local_report", fixture),
      storage(peer, revision, "save_local_report", fixture),
    ]);
    for (const result of concurrent) {
      if (result.status === "fulfilled" && result.value.ok)
        owned.add(result.value.id);
    }
    for (const result of concurrent) {
      check(
        result.status === "fulfilled",
        "Cross-tab vault operation rejected its transport",
      );
      if (!result.value.ok)
        check(
          /busy|full/i.test(result.value.error),
          `Unexpected cross-tab result: ${JSON.stringify(result.value)}`,
        );
    }
    let full = await storage(page, revision, "list_local_reports");
    check(
      full.ok && full.reports.length <= 16,
      "Concurrent tabs exceeded the 16-report bound",
    );
    if (full.reports.length < 16) {
      const result = await storage(
        page,
        revision,
        "save_local_report",
        fixture,
      );
      if (result.ok) owned.add(result.id);
      check(
        result.ok,
        "Serial capacity fill failed after a busy cross-tab request",
      );
      full = await storage(page, revision, "list_local_reports");
    }
    check(
      full.ok && full.reports.length === 16 && full.total_bytes <= 4194304,
      "Vault capacity does not match the documented count/byte limit",
    );
    const overCapacity = await storage(
      page,
      revision,
      "save_local_report",
      fixture,
    );
    if (overCapacity.ok) owned.add(overCapacity.id);
    check(
      overCapacity.ok === false && /full/i.test(overCapacity.error),
      "Vault accepted a seventeenth receipt",
    );
    return {
      ...manifest,
      execution:
        "real Rust/WASM in an offline browser tab; installed mode not qualified",
      offline_cases_passed: receipt.counts.pass,
      version_and_worker_verified: true,
      opfs_roundtrip: true,
      malformed_receipts_rejected: true,
      on_disk_corruption_detected: true,
      cross_tab_capacity_enforced: true,
    };
  } finally {
    await context.setOffline(false);
    if (peer) await peer.close();
    for (const id of owned) {
      try {
        const result = await storage(page, revision, "delete_local_report", id);
        if (!result.ok) throw new Error(result.error);
      } catch (error) {
        cleanupError ||= new Error(
          `Owned receipt cleanup failed for ${id}: ${error}`,
        );
      }
    }
    if (baseline?.ok) {
      try {
        const remaining = await storage(page, revision, "list_local_reports");
        check(
          remaining.ok &&
            JSON.stringify(remaining.reports) ===
              JSON.stringify(baseline.reports),
          "Vault test did not restore the pre-existing receipt inventory",
        );
      } catch (error) {
        cleanupError ||= error;
      }
    }
    await page.close();
    if (cleanupError) throw cleanupError;
  }
}
