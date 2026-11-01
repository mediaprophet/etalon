// Fixtures alter only the browser API boundary. Rust performs every mapping assertion.
export async function runBackendContracts(page) {
  return page.evaluate(async () => {
    const revision = document.querySelector(
      'meta[name="etalon-revision"]',
    ).content;
    const wasm = await import(
      `./browser/etalon_browser_demo.js?rev=${revision}`
    );
    if (typeof wasm.inspect_adapter_contract !== "function")
      throw new Error("Rust adapter contract export is missing");
    const gpu = navigator.gpu;
    const original = Object.getOwnPropertyDescriptor(gpu, "requestAdapter");
    const requestAdapter = gpu.requestAdapter.bind(gpu);
    const real = await requestAdapter();
    if (!real)
      throw new Error(
        "Backend contracts require a real browser adapter for limit fixtures",
      );
    const snapshot = {};
    for (
      let prototype = real.limits;
      prototype && prototype !== Object.prototype;
      prototype = Object.getPrototypeOf(prototype)
    ) {
      for (const key of Object.getOwnPropertyNames(prototype)) {
        if (key.startsWith("max") || key.startsWith("min")) {
          const value = real.limits[key];
          if (typeof value === "number") snapshot[key] = value;
        }
      }
    }
    const optional = [
      "maxStorageBuffersInVertexStage",
      "maxStorageBuffersInFragmentStage",
      "maxStorageTexturesInVertexStage",
      "maxStorageTexturesInFragmentStage",
      "maxBindGroupsPlusVertexBuffers",
      "maxColorAttachmentBytesPerSample",
    ];
    const evidence = [];
    try {
      for (const mode of ["missing", "zero", "description"]) {
        const limits = { ...snapshot };
        for (const name of optional) {
          if (mode === "zero") limits[name] = 0;
          else delete limits[name];
        }
        let deviceRequests = 0;
        const fixture = {
          limits,
          features: new Set(real.features),
          info: {
            vendor: "fixture-vendor",
            architecture: "fixture-architecture",
            device: "fixture-device",
            description: mode === "description" ? "fixture-description" : "",
            isFallbackAdapter: false,
            subgroupMinSize: 4,
            subgroupMaxSize: 128,
          },
          requestDevice() {
            deviceRequests++;
            return Promise.reject(
              new Error("Rust preflight reached browser requestDevice"),
            );
          },
        };
        Object.defineProperty(gpu, "requestAdapter", {
          configurable: true,
          value: () => Promise.resolve(fixture),
        });
        const result = JSON.parse(await wasm.inspect_adapter_contract(mode));
        if (deviceRequests !== 0)
          throw new Error("Oversized limits reached browser requestDevice");
        evidence.push({ ...result, browser_device_requests: deviceRequests });
      }
      for (const mode of ["auto-null-compatibility", "auto-core-rejection"]) {
        const calls = [];
        const rejection = "etalon-auto-core-rejection";
        Object.defineProperty(gpu, "requestAdapter", {
          configurable: true,
          value: (options) => {
            calls.push({ ...options });
            if (calls.length === 1) {
              if (mode === "auto-core-rejection")
                return Promise.reject(
                  new DOMException(rejection, "OperationError"),
                );
              return Promise.resolve(null);
            }
            if (calls.length !== 2 || options.featureLevel !== "compatibility")
              return Promise.reject(
                new Error("Unexpected automatic adapter request"),
              );
            return requestAdapter(options);
          },
        });
        let session;
        try {
          session = await wasm.create_session_for_profile("auto");
          const info = JSON.parse(session.info_json());
          if (
            info.requested_feature_level !== "auto" ||
            Object.hasOwn(calls[0] || {}, "featureLevel")
          )
            throw new Error("Auto negotiation did not begin with default Core");
          if (mode === "auto-null-compatibility") {
            const attempts = info.adapter_attempts;
            if (
              calls.length !== 2 ||
              info.gpu_status !== "available" ||
              info.selected_feature_level !== "compatibility" ||
              attempts?.length !== 2 ||
              attempts[0].feature_level !== "core" ||
              attempts[0].status !== "unsupported" ||
              attempts[1].feature_level !== "compatibility" ||
              attempts[1].status !== "available"
            )
              throw new Error(
                `Auto null fallback contract failed: ${JSON.stringify(info)}`,
              );
            evidence.push({
              mode,
              browser_adapter_requests: calls,
              selected_feature_level: info.selected_feature_level,
              gpu_status: info.gpu_status,
              adapter_attempts: attempts,
            });
          } else {
            const reason = info.gpu_reason;
            if (
              calls.length !== 1 ||
              info.gpu_status !== "fail" ||
              reason?.code !== "adapter-request-rejected" ||
              !reason.message?.includes(rejection) ||
              !reason.message?.includes("OperationError") ||
              reason.adapter_attempts?.length !== 1 ||
              reason.adapter_attempts[0].feature_level !== "core" ||
              reason.adapter_attempts[0].status !== "fail"
            )
              throw new Error(
                `Auto rejection preservation contract failed: ${JSON.stringify(info)}`,
              );
            evidence.push({
              mode,
              browser_adapter_requests: calls,
              gpu_status: info.gpu_status,
              rejection: reason,
            });
          }
        } finally {
          if (session) {
            try {
              session.destroy();
            } finally {
              session.free();
            }
          }
        }
      }
    } finally {
      if (original) Object.defineProperty(gpu, "requestAdapter", original);
      else delete gpu.requestAdapter;
    }
    return evidence;
  });
}
