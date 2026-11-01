import { appEnvironment } from "./app-environment.js";

const install = document.querySelector("#install-app");
const status = document.querySelector("#app-install-status");
const mode = document.querySelector("#app-mode-status");
const offline = document.querySelector("#app-offline-status");
const update = document.querySelector("#app-update-status");
const revision = document.querySelector('meta[name="etalon-revision"]').content;
let installPrompt;
let installing = false;

function showMode() {
  const environment = appEnvironment();
  document.body.dataset.etalonAppMode = environment.display_mode;
  mode.textContent = environment.installed_display_mode
    ? `Running in app mode (${environment.display_mode}); test receipts record this environment.`
    : `Running in ${environment.display_mode} mode. Open the installed app from your launcher to compare its receipt.`;
  install.disabled =
    environment.installed_display_mode || !installPrompt || installing;
}
showMode();
for (const name of ["standalone", "fullscreen", "minimal-ui", "browser"])
  window
    .matchMedia?.(`(display-mode: ${name})`)
    .addEventListener?.("change", showMode);
window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  installPrompt = event;
  status.textContent =
    "This browser offers installation. Use Install app when you want to add it.";
  showMode();
});
window.addEventListener("appinstalled", () => {
  installPrompt = undefined;
  status.textContent =
    "App installed. Open Etalon from your launcher and run the tests there.";
  showMode();
});
install.addEventListener("click", async () => {
  if (!installPrompt || installing) return;
  installing = true;
  const prompt = installPrompt;
  installPrompt = undefined;
  showMode();
  try {
    await prompt.prompt();
    const choice = await prompt.userChoice;
    status.textContent =
      choice.outcome === "accepted"
        ? "Installation accepted by the browser. Open Etalon from your launcher to test app mode."
        : "Installation was dismissed. You can continue testing in this browser.";
  } catch (error) {
    status.textContent = `Installation could not complete: ${String(error).slice(0, 512)}. You can use the browser menu.`;
  } finally {
    installing = false;
    showMode();
  }
});
function showUpdate(registration) {
  if (registration.waiting)
    update.textContent =
      "An app update is ready. Close all Etalon tabs and app windows, then reopen. The current run will not reload.";
}
function workerStatus(worker) {
  return new Promise((resolve, reject) => {
    const channel = new MessageChannel();
    const close = () => {
      clearTimeout(timer);
      channel.port1.close();
      channel.port2.close();
    };
    const timer = setTimeout(() => {
      close();
      reject(new Error("Timed out checking the offline app revision"));
    }, 5000);
    channel.port1.onmessage = (event) => {
      close();
      resolve(event.data);
    };
    channel.port1.onmessageerror = () => {
      close();
      reject(new Error("Invalid offline status response"));
    };
    try {
      worker.postMessage({ type: "ETALON_APP_STATUS" }, [channel.port2]);
    } catch (error) {
      close();
      reject(error);
    }
  });
}
async function registerApp() {
  if (!window.isSecureContext || !("serviceWorker" in navigator)) {
    offline.textContent =
      "Offline installation is unavailable in this browser context. Online Rust/WASM tests remain available.";
    return;
  }
  try {
    const registration = await navigator.serviceWorker.register(
      new URL("./service-worker.js?rev=" + revision, import.meta.url),
      { scope: "./", updateViaCache: "none" },
    );
    showUpdate(registration);
    registration.addEventListener("updatefound", () => {
      const worker = registration.installing;
      worker?.addEventListener("statechange", () => {
        showUpdate(registration);
        if (worker.state === "redundant")
          offline.textContent =
            "App asset caching did not complete. Keep this page online and retry on a later visit.";
      });
    });
    const ready = await navigator.serviceWorker.ready;
    if (!ready.active) throw new Error("No active offline worker");
    const receipt = await workerStatus(ready.active);
    if (typeof receipt?.revision !== "string" || receipt.offline_ready !== true)
      throw new Error("Offline worker did not confirm a cached revision");
    document.body.dataset.etalonOfflineRevision = receipt.revision.slice(0, 64);
    if (receipt.revision === revision) {
      offline.textContent =
        "This revision's app assets are cached for offline use. GPU availability is checked separately each run.";
      document.body.dataset.etalonOfflineReady = "true";
    } else {
      offline.textContent =
        "The cached app revision differs from this page. Close all Etalon tabs and app windows, then reopen to use a consistent revision.";
      document.body.dataset.etalonOfflineReady = "false";
    }
    showUpdate(registration);
  } catch (error) {
    offline.textContent = `Offline setup failed: ${String(error).slice(0, 512)}. Online tests remain available.`;
    document.body.dataset.etalonOfflineReady = "false";
  }
}
navigator.serviceWorker?.addEventListener("controllerchange", showMode);
registerApp();
