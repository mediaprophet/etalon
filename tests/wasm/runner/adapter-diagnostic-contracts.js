// Mutate only the browser boundary; exercise the exported Rust diagnostic.
export async function runAdapterDiagnosticContracts(page) {
  return page.evaluate(async () => {
    const revision = document.querySelector(
      'meta[name="etalon-revision"]',
    ).content;
    const wasm = await import(
      `./browser/etalon_browser_demo.js?rev=${revision}`
    );
    if (typeof wasm.adapter_diagnostic_errors !== "function")
      throw new Error("Rust errors-only adapter diagnostic export is missing");
    const originalNavigator = Object.getOwnPropertyDescriptor(
      globalThis,
      "navigator",
    );
    const originalSetTimeout = Object.getOwnPropertyDescriptor(
      globalThis,
      "setTimeout",
    );
    const originalClearTimeout = Object.getOwnPropertyDescriptor(
      globalThis,
      "clearTimeout",
    );
    const setTimeout = globalThis.setTimeout.bind(globalThis);
    const clearTimeout = globalThis.clearTimeout.bind(globalThis);
    const pendingTimers = new Set();
    const evidence = [];
    const restore = (key, descriptor) => {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    };
    const navigatorFixture = (get) =>
      Object.defineProperty(globalThis, "navigator", {
        configurable: true,
        get,
      });
    const assert = (condition, message) => {
      if (!condition) throw new Error(message);
    };
    const diagnostic = async () => {
      const result = JSON.parse(await wasm.adapter_diagnostic_errors());
      assert(
        result.schema === "etalon.adapter-diagnostic.v1" &&
          result.schema_version === 1 &&
          result.automatic_upload === false &&
          result.diagnostic === "adapter-acquisition-errors",
        "Wrong diagnostic schema",
      );
      assert(
        typeof result.source_revision === "string" &&
          typeof result.library_version === "string",
        "Missing revision/version evidence",
      );
      assert(
        Number.isFinite(result.elapsed_ms) && result.elapsed_ms >= 0,
        "Invalid elapsed duration",
      );
      assert(
        result.gpu_execution_tested === false &&
          !Object.hasOwn(result, "attempts") &&
          !Object.hasOwn(result, "cases"),
        "Diagnostic fabricated GPU coverage or included a full case report",
      );
      assert(
        Array.isArray(result.errors) &&
          result.error_count === result.errors.length,
        "Incorrect errors-only count",
      );
      assert(
        result.errors.every(
          (error) =>
            typeof error.test_id === "string" &&
            typeof error.stage === "string" &&
            typeof error.api_path === "string" &&
            error.status !== "adapter-returned",
        ),
        "Error lacks step identity or includes a successful probe",
      );
      assert(
        pendingTimers.size === 0,
        "A completed diagnostic retained its timeout timers",
      );
      return result;
    };
    try {
      Object.defineProperty(globalThis, "setTimeout", {
        configurable: true,
        value(callback, delay, ...args) {
          const id = setTimeout(callback, delay, ...args);
          pendingTimers.add(id);
          return id;
        },
      });
      Object.defineProperty(globalThis, "clearTimeout", {
        configurable: true,
        value(id) {
          pendingTimers.delete(id);
          return clearTimeout(id);
        },
      });
      const early = [
        [
          "navigator-null",
          () => null,
          1,
          "adapter-navigator",
          "navigator.lookup",
          "globalThis.navigator",
          "gpu-unavailable",
          "null",
        ],
        [
          "navigator-undefined",
          () => undefined,
          1,
          "adapter-navigator",
          "navigator.lookup",
          "globalThis.navigator",
          "gpu-unavailable",
          "undefined",
        ],
        [
          "navigator-throws",
          () => {
            throw new DOMException("navigator-fixture", "SecurityError");
          },
          1,
          "adapter-navigator",
          "navigator.lookup",
          "globalThis.navigator",
          "access-threw",
        ],
        [
          "gpu-null",
          () => ({ gpu: null }),
          2,
          "adapter-navigator-gpu",
          "gpu.lookup",
          "navigator.gpu",
          "gpu-unavailable",
          "null",
        ],
        [
          "gpu-undefined",
          () => ({}),
          2,
          "adapter-navigator-gpu",
          "gpu.lookup",
          "navigator.gpu",
          "gpu-unavailable",
          "undefined",
        ],
        [
          "gpu-throws",
          () => ({
            get gpu() {
              throw new DOMException("gpu-fixture", "SecurityError");
            },
          }),
          2,
          "adapter-navigator-gpu",
          "gpu.lookup",
          "navigator.gpu",
          "access-threw",
        ],
        [
          "method-null",
          () => ({ gpu: { requestAdapter: null } }),
          3,
          "adapter-requestAdapter",
          "requestAdapter.lookup",
          "navigator.gpu.requestAdapter",
          "method-unavailable",
          "null",
        ],
        [
          "method-undefined",
          () => ({ gpu: {} }),
          3,
          "adapter-requestAdapter",
          "requestAdapter.lookup",
          "navigator.gpu.requestAdapter",
          "method-unavailable",
          "undefined",
        ],
        [
          "method-throws",
          () => ({
            gpu: {
              get requestAdapter() {
                throw new DOMException("method-fixture", "OperationError");
              },
            },
          }),
          3,
          "adapter-requestAdapter",
          "requestAdapter.lookup",
          "navigator.gpu.requestAdapter",
          "method-unavailable",
        ],
      ];
      for (const [
        mode,
        get,
        steps,
        testId,
        stage,
        path,
        status,
        valueKind,
      ] of early) {
        navigatorFixture(get);
        const result = await diagnostic();
        const error = result.errors[0];
        assert(
          result.errors.length === 1 &&
            result.executed.preflight_steps === steps &&
            result.executed.request_shapes === 0 &&
            result.executed.adapter_requests === 0,
          `Early fixture did not stop at its failing step: ${mode}`,
        );
        assert(
          error.test_id === testId &&
            error.stage === stage &&
            error.api_path === path &&
            error.status === status &&
            (!valueKind || error.value_kind === valueKind),
          `Early fixture lost its exact API failure: ${mode}`,
        );
        if (mode.endsWith("throws"))
          assert(
            error.name ===
              (mode === "method-throws" ? "OperationError" : "SecurityError"),
            `Exception name lost: ${mode}`,
          );
        evidence.push({
          mode,
          stage,
          api_path: path,
          errors: 1,
          adapter_requests: 0,
        });
      }
      const ids = [
        "adapter-no-arguments",
        "adapter-empty-options",
        "adapter-explicit-core",
        "adapter-explicit-compatibility",
        "adapter-force-fallback",
      ];
      let calls = [];
      const gpu = {
        requestAdapter(...args) {
          assert(this === gpu, "Diagnostic lost the GPU request receiver");
          calls.push(args);
          switch (calls.length) {
            case 1:
              return Promise.resolve(null);
            case 2:
              return Promise.resolve(undefined);
            case 3:
              return Promise.reject(
                new DOMException("diagnostic-rejection", "OperationError"),
              );
            case 4:
              return Promise.resolve({
                info: { vendor: "diagnostic-fixture" },
              });
            case 5:
              throw new DOMException("diagnostic-call", "SecurityError");
            default:
              throw new Error("Diagnostic exceeded its five request shapes");
          }
        },
      };
      navigatorFixture(() => ({ gpu }));
      const mixed = await diagnostic();
      assert(
        calls.length === 5 &&
          calls[0].length === 0 &&
          calls[1].length === 1 &&
          Object.keys(calls[1][0]).length === 0 &&
          JSON.stringify(calls[2][0]) === '{"featureLevel":"core"}' &&
          JSON.stringify(calls[3][0]) === '{"featureLevel":"compatibility"}' &&
          JSON.stringify(calls[4][0]) === '{"forceFallbackAdapter":true}',
        "Diagnostic changed the five shared request shapes",
      );
      assert(
        mixed.executed.request_shapes === 5 &&
          mixed.executed.adapter_requests === 5 &&
          mixed.executed.successful_requests === 1 &&
          mixed.errors.length === 4,
        "Mixed outcomes produced incorrect executed/error counts",
      );
      assert(
        JSON.stringify(mixed.errors.map((error) => error.test_id)) ===
          JSON.stringify([ids[0], ids[1], ids[2], ids[4]]) &&
          JSON.stringify(mixed.errors.map((error) => error.status)) ===
            '["null","undefined","rejected","call-threw"]',
        "Mixed outcomes were attributed to the wrong individual probe",
      );
      assert(
        mixed.errors
          .slice(0, 3)
          .every(
            (error) =>
              error.api_path === "navigator.gpu.requestAdapter.Promise.result",
          ) && mixed.errors[3].api_path === "navigator.gpu.requestAdapter",
        "Promise and call failures collapsed into a generic stage",
      );
      assert(
        mixed.errors[2].name === "OperationError" &&
          mixed.errors[3].name === "SecurityError",
        "Mixed exception identities were lost",
      );
      evidence.push({
        mode: "mixed-individual-outcomes",
        requests: 5,
        successes: 1,
        errors: 4,
        request_shapes_verified: true,
      });

      let invalidIndex = 0;
      gpu.requestAdapter = () =>
        [undefined, null, 7, {}, Promise.resolve(false)][invalidIndex++];
      const invalid = await diagnostic();
      assert(
        invalid.errors.length === 5 &&
          invalid.errors
            .slice(0, 4)
            .every(
              (error) =>
                error.status === "non-promise-result" &&
                error.api_path === "navigator.gpu.requestAdapter.return",
            ) &&
          invalid.errors[0].value_kind === "undefined" &&
          invalid.errors[1].value_kind === "null" &&
          invalid.errors[4].status === "invalid-adapter-result",
        "Invalid call return and Promise result were conflated",
      );
      evidence.push({ mode: "invalid-call-and-promise-results", errors: 5 });

      let resolveLate;
      let timeoutCalls = 0;
      gpu.requestAdapter = () =>
        ++timeoutCalls === 1
          ? new Promise((resolve) => {
              resolveLate = resolve;
            })
          : Promise.resolve({ info: {} });
      const timed = await diagnostic();
      assert(
        timed.errors.length === 1 &&
          timed.errors[0].test_id === ids[0] &&
          timed.errors[0].status === "timeout" &&
          timed.errors[0].timeout_ms === timed.probe_timeout_ms &&
          timed.executed.successful_requests === 4,
        "An unresolved probe did not time out independently",
      );
      resolveLate({ info: {} });
      gpu.requestAdapter = () => Promise.resolve({ info: {} });
      const success = await diagnostic();
      assert(
        success.errors.length === 0 &&
          success.executed.successful_requests === 5 &&
          success.summary === "adapter-probes-completed",
        "Successful adapter acquisition was mislabeled as an error or GPU qualification",
      );
      evidence.push({
        mode: "per-probe-timeout-and-late-resolution",
        errors: 1,
        post_timeout_errors: 0,
        timer_cleanup_verified: true,
      });
      return evidence;
    } finally {
      restore("navigator", originalNavigator);
      restore("setTimeout", originalSetTimeout);
      restore("clearTimeout", originalClearTimeout);
      for (const id of pendingTimers) clearTimeout(id);
    }
  });
}
