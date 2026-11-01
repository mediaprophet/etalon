// Browser-boundary fixtures for the bounded Rust WebGL renderer collector.
export async function runEnvironmentWebglContracts(page) {
  return page.evaluate(async () => {
    const revision = document.querySelector(
      'meta[name="etalon-revision"]',
    ).content;
    const wasm = await import(
      `./browser/etalon_browser_demo.js?rev=${revision}`
    );
    const originalCanvas = Object.getOwnPropertyDescriptor(
      globalThis,
      "OffscreenCanvas",
    );
    const originalNavigator = Object.getOwnPropertyDescriptor(
      globalThis,
      "navigator",
    );
    const evidence = [];
    const assert = (ok, message) => {
      if (!ok) throw new Error(message);
    };
    const restore = (key, descriptor) => {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    };
    const install = (descriptor) =>
      Object.defineProperty(globalThis, "OffscreenCanvas", {
        configurable: true,
        ...descriptor,
      });
    const read = async () => {
      const result = JSON.parse(await wasm.browser_environment_json()).webgl;
      assert(
        result?.scope === "WebGL renderer declarations only",
        "WebGL probe missing or mislabeled",
      );
      assert(
        result.webgpu_adapter_established === false &&
          result.webgpu_driver_established === false &&
          result.browser_gate_established === false,
        "WebGL strings fabricated WebGPU/driver/gate evidence",
      );
      assert(
        result.canvas.width === 1 &&
          result.canvas.height === 1 &&
          result.bounds.max_context_attempts === 2 &&
          result.bounds.string_bytes === 512,
        "Renderer probe exceeded declared bounds",
      );
      return result;
    };
    try {
      Object.defineProperty(globalThis, "navigator", {
        configurable: true,
        value: {},
      });
      for (const [mode, descriptor, status] of [
        ["constructor-unavailable", { value: undefined }, "unavailable"],
        ["constructor-null", { value: null }, "unavailable"],
        [
          "constructor-getter-error",
          {
            get() {
              throw new DOMException("canvas-getter", "SecurityError");
            },
          },
          "error",
        ],
        [
          "constructor-call-error",
          {
            value: function () {
              throw new Error("canvas-constructor");
            },
          },
          "error",
        ],
      ]) {
        install(descriptor);
        const result = await read();
        assert(
          result.status === status &&
            result.stage.startsWith("OffscreenCanvas."),
          `Canvas failure hidden: ${mode}`,
        );
        evidence.push({ mode, status: result.status, stage: result.stage });
      }
      const modes = [
        "context-null",
        "context-getter-error",
        "available",
        "webgl-fallback",
        "debug-null",
        "debug-undefined",
        "debug-call-error",
        "parameter-getter-error",
        "unmasked-parameter-error",
        "extension-getter-error",
        "cleanup-unavailable",
        "cleanup-getter-error",
        "cleanup-call-error",
        "bounded-strings",
      ];
      for (const mode of modes) {
        const contexts = [],
          parameters = [];
        let constructed = 0,
          loseRequests = 0;
        const lose = {
          loseContext() {
            assert(this === lose, "Cleanup lost extension receiver");
            loseRequests++;
            if (mode === "cleanup-call-error")
              throw new Error("lose-context-error");
          },
        };
        if (mode === "cleanup-getter-error")
          Object.defineProperty(lose, "loseContext", {
            get() {
              throw new Error("lose-getter-error");
            },
          });
        const gl = {
          getParameter(token) {
            assert(this === gl, "getParameter lost WebGL receiver");
            parameters.push(token);
            if (mode === "unmasked-parameter-error" && token === 0x9246)
              throw new Error("renderer-read-error");
            const values = {
              0x1f00: "masked-vendor",
              0x1f01: "masked-renderer",
              0x1f02: "fixture-WebGL-version",
              0x8b8c: "fixture-shading-version",
              0x9245: "unmasked-vendor",
              0x9246:
                mode === "bounded-strings"
                  ? String.fromCodePoint(0x1f600).repeat(512)
                  : "unmasked-renderer",
            };
            assert(Object.hasOwn(values, token), "Unexpected WebGL parameter");
            return values[token];
          },
          getExtension(name) {
            assert(this === gl, "getExtension lost WebGL receiver");
            if (name === "WEBGL_debug_renderer_info") {
              if (mode === "debug-null") return null;
              if (mode === "debug-undefined") return undefined;
              if (mode === "debug-call-error")
                throw new Error("debug-extension-error");
              return {};
            }
            assert(name === "WEBGL_lose_context", "Unexpected WebGL extension");
            return mode === "cleanup-unavailable" ? null : lose;
          },
        };
        if (mode === "parameter-getter-error")
          Object.defineProperty(gl, "getParameter", {
            get() {
              throw new Error("parameter-getter-error");
            },
          });
        if (mode === "extension-getter-error")
          Object.defineProperty(gl, "getExtension", {
            get() {
              throw new Error("extension-getter-error");
            },
          });
        class Canvas {
          constructor(width, height) {
            constructed++;
            assert(width === 1 && height === 1, "Probe created larger canvas");
          }
          getContext(name) {
            contexts.push(name);
            return mode === "context-null" ||
              (mode === "webgl-fallback" && name === "webgl2")
              ? null
              : gl;
          }
        }
        if (mode === "context-getter-error")
          Object.defineProperty(Canvas.prototype, "getContext", {
            get() {
              throw new Error("context-getter-error");
            },
          });
        install({ value: Canvas });
        const result = await read();
        assert(
          constructed === 1 && contexts.length <= 2,
          `Canvas/context budget violated: ${mode}`,
        );
        if (mode === "context-null") {
          assert(
            result.status === "unavailable" &&
              result.context_attempts.length === 2 &&
              result.context_attempts.every(
                (attempt) => attempt.result === "null",
              ) &&
              loseRequests === 0,
            "Null contexts were not explicit",
          );
        } else if (mode === "context-getter-error") {
          assert(
            result.status === "error" &&
              result.stage === "getContext.lookup" &&
              loseRequests === 0,
            "getContext getter exception was hidden",
          );
        } else {
          assert(
            result.context_type ===
              (mode === "webgl-fallback" ? "webgl" : "webgl2"),
            "Wrong WebGL context path",
          );
          const cannotLose = [
            "extension-getter-error",
            "cleanup-unavailable",
            "cleanup-getter-error",
          ].includes(mode);
          assert(
            loseRequests === (cannotLose ? 0 : 1),
            `Context cleanup skipped or repeated: ${mode}`,
          );
          if (mode === "cleanup-unavailable")
            assert(
              result.cleanup.status === "unavailable",
              "Unavailable cleanup extension hidden",
            );
          else if (
            [
              "extension-getter-error",
              "cleanup-getter-error",
              "cleanup-call-error",
            ].includes(mode)
          )
            assert(
              result.cleanup.status === "error",
              "Cleanup exception hidden",
            );
          else
            assert(
              result.cleanup.status === "requested",
              `Context loss not requested after probe: ${mode}`,
            );
          if (["debug-null", "debug-undefined"].includes(mode))
            assert(
              result.debug_renderer_info.status === "unavailable" &&
                !parameters.includes(0x9245) &&
                !parameters.includes(0x9246),
              "Unavailable debug extension was used or guessed",
            );
          if (mode === "parameter-getter-error")
            assert(
              result.status === "error" &&
                result.parameters.renderer.stage === "getParameter.lookup",
              "Parameter getter failure hidden",
            );
          if (mode === "unmasked-parameter-error")
            assert(
              result.status === "error" &&
                result.debug_renderer_info.unmasked_renderer.stage ===
                  "getParameter.call",
              "Unmasked parameter error hidden",
            );
          if (mode === "debug-call-error")
            assert(
              result.status === "error" &&
                result.debug_renderer_info.stage === "getExtension.call",
              "Debug extension error hidden",
            );
          if (mode === "available")
            assert(
              result.parameters.vendor.value === "masked-vendor" &&
                result.parameters.renderer.value === "masked-renderer" &&
                result.parameters.version.value === "fixture-WebGL-version" &&
                result.debug_renderer_info.unmasked_renderer.value ===
                  "unmasked-renderer",
              "Raw WebGL strings were changed",
            );
          if (mode === "bounded-strings")
            assert(
              new TextEncoder().encode(
                result.debug_renderer_info.unmasked_renderer.value,
              ).length <= 512 &&
                result.debug_renderer_info.unmasked_renderer.truncated,
              "Renderer string exceeded byte budget",
            );
        }
        evidence.push({
          mode,
          status: result.status,
          cleanup: result.cleanup?.status ?? "no-context",
          lose_requests: loseRequests,
          context_attempts: contexts.length,
        });
      }
      return evidence;
    } finally {
      restore("OffscreenCanvas", originalCanvas);
      restore("navigator", originalNavigator);
    }
  });
}
