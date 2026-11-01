(() => {
  const revision = document.querySelector(
    'meta[name="etalon-revision"]',
  ).content;
  const output = document.querySelector("#app-startup-report");
  const status = document.querySelector("#app-startup-status");
  const reload = document.querySelector("#reload-app");
  const modules = [
    {
      id: "suite",
      path: "browser-tests.js",
      controls: ["run-tests", "feature-level"],
      status: "test-progress",
    },
    {
      id: "diagnostic",
      path: "adapter-diagnostic.js",
      controls: ["run-adapter-diagnostic"],
      status: "adapter-diagnostic-status",
    },
    {
      id: "install",
      path: "app-install.js",
      controls: ["install-app"],
      status: "app-install-status",
    },
    {
      id: "storage",
      path: "app-storage.js",
      controls: [
        "save-local-report",
        "refresh-local-reports",
        "request-persistence",
      ],
      status: "storage-status",
    },
    {
      id: "radio",
      path: "browser-radio.js",
      controls: [
        "ble-connect",
        "ble-read",
        "ble-write",
        "ble-start-notifications",
        "ble-stop-notifications",
        "ble-disconnect",
        "refresh-radio",
      ],
      status: "radio-status",
    },
    {
      id: "camera",
      path: "browser-camera.js",
      controls: [
        "camera-enumerate",
        "camera-start",
        "camera-stop",
        "camera-capture",
      ],
      status: "camera-status",
    },
  ];
  const report = {
    schema: "etalon.app-startup.v1",
    source_revision: revision,
    library_version: document.querySelector('meta[name="etalon-version"]')
      .content,
    started_at: new Date().toISOString(),
    automatic_upload: false,
    scope:
      "Interface module loading; device and WASM execution are qualified separately",
    components: modules.map(({ id, path }) => ({
      id,
      path,
      status: "loading",
    })),
  };
  function show() {
    const pending = report.components.filter(
      (entry) => entry.status === "loading",
    ).length;
    const failed = report.components.filter(
      (entry) => entry.status === "failed",
    ).length;
    document.body.dataset.etalonStartup = pending
      ? "loading"
      : failed
        ? "failed"
        : "ready";
    status.textContent = failed
      ? `${failed} app component(s) failed to load${pending ? `; ${pending} still loading` : ""}. See the exact module error below. Reload explicitly to retry.`
      : pending
        ? `Loading ${pending} app component(s)…`
        : "App interface loaded. Run the tests to establish device capabilities.";
    output.textContent = JSON.stringify(report, null, 2);
  }
  reload.disabled = false;
  reload.addEventListener("click", () => location.reload());
  show();
  document.querySelector("#download-startup-report").disabled = false;
  document
    .querySelector("#download-startup-report")
    .addEventListener("click", () => {
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(report, null, 2)], {
          type: "application/json",
        }),
      );
      const link = document.createElement("a");
      link.href = url;
      link.download = `etalon-${revision.slice(0, 12)}-startup.json`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
  for (const [index, module] of modules.entries()) {
    const entry = report.components[index];
    const key =
      "etalonComponent" + module.id[0].toUpperCase() + module.id.slice(1);
    document.body.dataset[key] = "loading";
    const started = performance.now();
    function fail(error) {
      const evidence = {
        name: String(error?.name || "Error").slice(0, 128),
        message: String(error?.message || error).slice(0, 2048),
        url: new URL(module.path, document.baseURI).href.slice(0, 1024),
      };
      if (entry.status === "failed") {
        entry.late_error = evidence;
        show();
        return;
      }
      entry.status = "failed";
      entry.error = evidence;
      entry.duration_ms = Math.round(performance.now() - started);
      document.body.dataset[key] = "failed";
      for (const id of module.controls)
        document.getElementById(id).disabled = true;
      document.getElementById(module.status).textContent =
        `${module.path} failed to load: ${entry.error.message}`;
      if (module.id === "suite") document.body.dataset.etalonReady = "false";
      show();
    }
    const timer = setTimeout(
      () =>
        fail({
          name: "ModuleLoadTimeout",
          message:
            "Module loading exceeded 15 seconds. Its controls remain disabled until an explicit reload.",
        }),
      15000,
    );
    import(new URL(module.path, document.baseURI).href)
      .then(() => {
        if (entry.status === "failed") {
          entry.late_loaded = true;
          return;
        }
        entry.status = "loaded";
        document.body.dataset[key] = "loaded";
      })
      .catch(fail)
      .finally(() => {
        clearTimeout(timer);
        entry.duration_ms = Math.round(performance.now() - started);
        show();
      });
  }
})();
