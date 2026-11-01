export function appEnvironment() {
  const modes = ["standalone", "fullscreen", "minimal-ui", "browser"];
  const matched = modes.filter(
    (mode) => window.matchMedia?.(`(display-mode: ${mode})`).matches,
  );
  const iosStandalone =
    typeof navigator.standalone === "boolean" ? navigator.standalone : null;
  const standalone = matched.includes("standalone");
  const worker = navigator.serviceWorker;
  return {
    schema: "etalon.app-environment.v1",
    app_version:
      document.querySelector('meta[name="etalon-version"]')?.content ||
      "unidentified",
    source_revision:
      document.querySelector('meta[name="etalon-revision"]')?.content ||
      "unidentified",
    display_mode: matched[0] || "unknown",
    display_modes: matched,
    standalone_media: standalone,
    navigator_standalone: iosStandalone,
    installed_display_mode:
      standalone || matched.includes("minimal-ui") || iosStandalone === true,
    online: navigator.onLine,
    offline_cache: {
      ready: document.body.dataset.etalonOfflineReady === "true",
      revision: document.body.dataset.etalonOfflineRevision || null,
    },
    service_worker: {
      supported: "serviceWorker" in navigator,
      controlled: Boolean(worker?.controller),
      controller_script_url:
        worker?.controller?.scriptURL?.slice(0, 512) || null,
    },
  };
}
