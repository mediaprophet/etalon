import { chromium } from "playwright";

const REVISION = "5f361e53deccf39d7e31d42f1f0d7c90e9377d13";
const MAX_OUTPUT = 500 * 1024;
const args = Object.fromEntries(
  process.argv.slice(2).reduce((all, value, index, raw) => {
    if (index % 2 === 0) all.push([value, raw[index + 1]]);
    return all;
  }, []),
);
for (const key of ["--chrome", "--profile", "--url", "--variant"])
  if (!args[key]) throw new Error(`Missing ${key}`);
if (!["baseline", "swangle", "swiftshader-vulkan"].includes(args["--variant"]))
  throw new Error("Invalid diagnostic variant");
const address = new URL(args["--url"]);
if (address.hostname !== "127.0.0.1" || address.protocol !== "http:")
  throw new Error("Expected owned loopback server");
const flags = [
  "--enable-unsafe-webgpu",
  "--use-webgpu-adapter=swiftshader",
  "--use-gpu-in-tests",
  "--disable-background-networking",
  "--disable-component-update",
  "--disable-extensions",
  "--disable-sync",
  "--disk-cache-size=1048576",
  "--media-cache-size=1048576",
];
if (args["--variant"] === "swangle")
  flags.push("--use-gl=angle", "--use-angle=swiftshader");
if (args["--variant"] === "swiftshader-vulkan")
  flags.push(
    "--enable-features=Vulkan",
    "--use-angle=vulkan",
    "--use-vulkan=swiftshader",
    "--disable-vulkan-surface",
  );
const launch = (suffix = "") =>
  chromium.launchPersistentContext(args["--profile"] + suffix, {
    executablePath: args["--chrome"],
    headless: true,
    args: flags,
  });
let context = await launch();

async function instrument(page) {
  await page.addInitScript(() => {
    const events = [];
    let bytes = 0,
      sequence = 0;
    const ids = new WeakMap();
    const id = (object) => {
      if (!ids.has(object)) ids.set(object, ++sequence);
      return ids.get(object);
    };
    const stage = () =>
      window.__traceStage ||
      document.querySelector("#test-progress")?.textContent;
    const log = (operation, details = {}) => {
      const entry = {
        operation,
        stage: stage(),
        time: performance.now(),
        ...details,
      };
      const size = new TextEncoder().encode(JSON.stringify(entry)).byteLength;
      if (events.length >= 512 || bytes + size > 96 * 1024) {
        window.__traceTruncated = true;
        return;
      }
      bytes += size;
      events.push(entry);
    };
    const wrap = (prototype, name, body) => {
      const original = prototype[name];
      prototype[name] = function (...values) {
        return body.call(this, original, values);
      };
    };
    window.__trace = events;
    window.__deviceDescriptors = [];
    window.__retainedDevices = [];
    wrap(GPU.prototype, "requestAdapter", function (raw, values) {
      log("requestAdapter.begin", { options: values[0] });
      return raw.apply(this, values).then(
        (adapter) => {
          log("requestAdapter.resolved", {
            adapter: adapter ? id(adapter) : null,
          });
          return adapter;
        },
        (error) => {
          log("requestAdapter.rejected", {
            error: String(error).slice(0, 1000),
          });
          throw error;
        },
      );
    });
    wrap(GPUAdapter.prototype, "requestDevice", function (raw, values) {
      const descriptor = JSON.parse(JSON.stringify(values[0] || {}));
      if (window.__deviceDescriptors.length < 4)
        window.__deviceDescriptors.push(descriptor);
      log("requestDevice.begin", { adapter: id(this), descriptor });
      return raw.apply(this, values).then((device) => {
        log("requestDevice.resolved", { device: id(device) });
        window.__retainedDevices.push(device);
        device.lost.then((info) =>
          log("device.lost", {
            device: id(device),
            reason: info.reason,
            message: info.message.slice(0, 1000),
          }),
        );
        return device;
      });
    });
    wrap(GPUDevice.prototype, "destroy", function (raw, values) {
      log("device.destroy", {
        device: id(this),
        stack: new Error().stack?.slice(0, 2000),
      });
      return raw.apply(this, values);
    });
    wrap(GPUDevice.prototype, "pushErrorScope", function (raw, values) {
      log("scope.push", { device: id(this), filter: values[0] });
      return raw.apply(this, values);
    });
    wrap(GPUDevice.prototype, "popErrorScope", function (raw, values) {
      log("scope.pop", { device: id(this) });
      return raw.apply(this, values).then(
        (error) => {
          log("scope.resolved", {
            device: id(this),
            error: error?.message?.slice(0, 1000) || null,
          });
          return error;
        },
        (error) => {
          log("scope.rejected", { error: String(error).slice(0, 1000) });
          throw error;
        },
      );
    });
    wrap(GPUCanvasContext.prototype, "configure", function (raw, values) {
      log("canvas.configure.begin", {
        context: id(this),
        device: id(values[0].device),
        format: values[0].format,
      });
      try {
        const result = raw.apply(this, values);
        log("canvas.configure.end", { context: id(this) });
        return result;
      } catch (error) {
        log("canvas.configure.threw", { error: String(error).slice(0, 1000) });
        throw error;
      }
    });
    wrap(
      GPUCanvasContext.prototype,
      "getCurrentTexture",
      function (raw, values) {
        log("canvas.getCurrentTexture.begin", { context: id(this) });
        const texture = raw.apply(this, values);
        log("canvas.getCurrentTexture.end", {
          context: id(this),
          texture: id(texture),
        });
        return texture;
      },
    );
    wrap(GPUCanvasContext.prototype, "unconfigure", function (raw, values) {
      log("canvas.unconfigure.begin", { context: id(this) });
      const result = raw.apply(this, values);
      log("canvas.unconfigure.end", { context: id(this) });
      return result;
    });
    wrap(GPUTexture.prototype, "destroy", function (raw, values) {
      log("texture.destroy.begin", { texture: id(this) });
      const result = raw.apply(this, values);
      log("texture.destroy.end", { texture: id(this) });
      return result;
    });
  });
}

async function sessionScenario(page) {
  return page.evaluate(async (revision) => {
    window.__traceStage = "canvas-alone: initialize Rust session";
    const wasm = await import(
      `./browser/etalon_browser_demo.js?rev=${revision}`
    );
    await wasm.default({
      module_or_path: `./browser/etalon_browser_demo_bg.wasm?rev=${revision}`,
    });
    const session = await wasm.create_session_for_profile("core");
    try {
      const runtime = JSON.parse(session.info_json());
      if (runtime.source_revision !== revision)
        throw new Error("WASM source mismatch");
      const cases = [];
      for (const id of ["browser-canvas-capabilities", "agent-gpu-lease"]) {
        window.__traceStage = `canvas-alone: ${id}`;
        cases.push(JSON.parse(await session.run_case(id)));
      }
      return { runtime, cases };
    } finally {
      window.__traceStage = "canvas-alone: explicit session cleanup";
      try {
        session.destroy();
      } finally {
        session.free();
      }
    }
  }, REVISION);
}

async function rawScenario(page, descriptor, mode) {
  return page.evaluate(
    async ({ descriptor, mode }) => {
      const mark = (name) => {
        window.__traceStage = `${mode}: ${name}`;
      };
      mark("request default Core adapter");
      const adapter = await navigator.gpu.requestAdapter({});
      if (!adapter) throw new Error("No Core adapter for raw control");
      const device = await adapter.requestDevice(descriptor);
      let lost = null,
        context,
        texture,
        pendingScopes;
      const result = { mode, descriptor, scopes: [], checkpoints: [] };
      device.lost.then((info) => {
        lost = { reason: info.reason, message: info.message };
      });
      const checkpoint = async (name) => {
        mark(name);
        await new Promise((resolve) => setTimeout(resolve, 50));
        result.checkpoints.push({ stage: name, lost });
      };
      const popScopes = () =>
        Promise.allSettled([
          device.popErrorScope(),
          device.popErrorScope(),
          device.popErrorScope(),
        ]);
      const storeScopes = (values) =>
        values.map((value) =>
          value.status === "fulfilled"
            ? { status: value.status, error: value.value?.message || null }
            : {
                status: value.status,
                error: String(value.reason).slice(0, 1000),
              },
        );
      const cleanup = () => {
        if (
          texture &&
          ![
            "raw-hdr-no-destroy",
            "raw-baseline",
            "raw-hdr-acquire-settle",
          ].includes(mode)
        ) {
          mark("explicit texture destroy");
          texture.destroy();
        }
        if (context) {
          mark("context unconfigure");
          context.unconfigure();
          context = null;
        }
        texture = null;
      };
      try {
        context = new OffscreenCanvas(1, 1).getContext("webgpu");
        device.pushErrorScope("validation");
        device.pushErrorScope("internal");
        device.pushErrorScope("out-of-memory");
        const baseline = mode === "raw-baseline";
        const configureOnly =
          mode === "raw-hdr-configure" || mode === "raw-hdr-configure-settle";
        const settleFirst = [
          "raw-hdr-configure-settle",
          "raw-hdr-acquire-settle",
          "raw-baseline",
        ].includes(mode);
        result.format = baseline
          ? navigator.gpu.getPreferredCanvasFormat()
          : "rgba16float";
        mark("configure");
        context.configure({ device, format: result.format });
        if (!configureOnly) {
          mark("getCurrentTexture");
          texture = context.getCurrentTexture();
        }
        if (!settleFirst) cleanup();
        mark(
          settleFirst
            ? "scope settlement BEFORE cleanup"
            : "scope settlement AFTER cleanup",
        );
        pendingScopes = popScopes();
        result.scopes = storeScopes(await pendingScopes);
        result.lost_before_cleanup = lost;
        if (settleFirst) {
          await checkpoint("settled before cleanup");
          cleanup();
        }
        await checkpoint("cleanup completed");
        mark("queue completion");
        let timer;
        try {
          await Promise.race([
            device.queue.onSubmittedWorkDone(),
            new Promise((_, reject) => {
              timer = setTimeout(
                () => reject(new Error("Queue completion timed out")),
                3000,
              );
            }),
          ]);
          result.queue = "resolved";
        } catch (error) {
          result.queue = String(error).slice(0, 1000);
        } finally {
          clearTimeout(timer);
        }
        await checkpoint("after queue completion");
        result.lost_before_device_destroy = lost;
      } catch (error) {
        result.failure = String(error).slice(0, 1500);
      } finally {
        // Attach handlers to every popped scope even on synchronous API failure.
        if (!pendingScopes) {
          try {
            result.scopes = storeScopes(await popScopes());
          } catch (error) {
            result.scope_cleanup_failure = String(error).slice(0, 1000);
          }
        }
        try {
          cleanup();
        } catch (error) {
          result.context_cleanup_failure = String(error).slice(0, 1000);
        }
        mark("explicit device cleanup");
        device.destroy();
      }
      return result;
    },
    { descriptor, mode },
  );
}

const output = {
  schema: "etalon.mobile-browser-diagnostic.v1",
  diagnostic_only: true,
  source_revision: REVISION,
  variant: args["--variant"],
  flags,
  scenarios: [],
};
let descriptor;
try {
  for (const scenario of [
    "canvas-alone",
    "full-suite",
    "raw-baseline",
    "raw-hdr-configure",
    "raw-hdr-configure-settle",
    "raw-hdr-no-destroy",
    "raw-hdr-acquire-settle",
    "raw-hdr-original",
  ]) {
    if (scenario.startsWith("raw-")) {
      await context.close();
      context = await launch(`-${scenario}`);
    }
    const page = await context.newPage();
    const record = { scenario, errors: [] };
    page.on("pageerror", (error) => {
      if (record.errors.length < 8)
        record.errors.push(String(error).slice(0, 1000));
    });
    try {
      await instrument(page);
      await page.goto(address.href, { waitUntil: "load", timeout: 30000 });
      const revision = await page
        .locator('meta[name="etalon-revision"]')
        .getAttribute("content");
      if (revision !== REVISION) throw new Error("Page source mismatch");
      const execute = async () => {
        if (scenario === "canvas-alone") return sessionScenario(page);
        if (scenario.startsWith("raw-")) {
          if (!descriptor)
            throw new Error(
              "Missing actual Rust device descriptor; refusing guessed raw control",
            );
          return rawScenario(page, descriptor, scenario);
        }
        await page.locator("#feature-level").selectOption("core");
        await page.locator("#run-tests").click();
        await page.waitForFunction(
          () => document.body.dataset.etalonState === "complete",
          null,
          { timeout: 60000 },
        );
        return page.evaluate(() =>
          JSON.parse(
            document.querySelector("#qualification-report").textContent,
          ),
        );
      };
      let timer;
      try {
        record.result = await Promise.race([
          execute(),
          new Promise((_, reject) => {
            timer = setTimeout(
              () => reject(new Error("Diagnostic scenario timed out")),
              scenario.startsWith("raw-") ? 15000 : 65000,
            );
          }),
        ]);
      } finally {
        clearTimeout(timer);
      }
    } catch (error) {
      record.failure = String(error).slice(0, 2000);
    } finally {
      try {
        const snapshot = await page.evaluate(() => ({
          trace: window.__trace,
          truncated: Boolean(window.__traceTruncated),
          descriptors: window.__deviceDescriptors,
          browser: navigator.userAgent,
        }));
        record.trace = snapshot.trace;
        record.trace_truncated = snapshot.truncated;
        record.browser = snapshot.browser;
        descriptor ||= snapshot.descriptors?.[0];
      } catch (error) {
        record.snapshot_failure = String(error).slice(0, 1000);
      }
      await page.close();
    }
    output.scenarios.push(record);
  }
} finally {
  await context.close();
}
const encoded = JSON.stringify(output);
if (Buffer.byteLength(encoded) > MAX_OUTPUT)
  throw new Error("Diagnostic output exceeds 500 KiB budget");
process.stdout.write(encoded);
