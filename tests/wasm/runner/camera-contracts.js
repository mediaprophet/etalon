import assert from "node:assert/strict";

export async function runCameraContracts(page) {
  const report = await page.evaluate(async () => {
    const wasm = await (
      await import(new URL("./wasm-runtime.js", location.href).href)
    ).loadRuntime();
    const parse = (value) => JSON.parse(value);
    const manager = navigator.mediaDevices;
    if (!manager || !HTMLCanvasElement.prototype.captureStream)
      throw new Error(
        "Synthetic camera contracts require secure-context mediaDevices and canvas.captureStream",
      );
    const descriptors = Object.fromEntries(
      ["getUserMedia", "enumerateDevices"].map((name) => [
        name,
        Object.getOwnPropertyDescriptor(manager, name),
      ]),
    );
    const canvas = document.createElement("canvas");
    canvas.width = 16;
    canvas.height = 12;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "rgb(10,20,30)";
    ctx.fillRect(0, 0, 16, 12);
    let calls = [];
    let created = [];
    let resolveLate;
    let mode = "success";
    const makeStream = () => {
      const stream = canvas.captureStream(0);
      created.push(stream);
      return stream;
    };
    const substitute = (name, value) =>
      Object.defineProperty(manager, name, { configurable: true, value });
    substitute("enumerateDevices", async () => [
      {
        kind: "videoinput",
        deviceId: "synthetic-camera",
        label: "IR thermal camera (synthetic label)",
        groupId: "synthetic-group",
      },
      {
        kind: "audioinput",
        deviceId: "not-a-camera",
        label: "microphone",
        groupId: "synthetic-group",
      },
    ]);
    substitute("getUserMedia", async (constraints) => {
      calls.push(constraints);
      if (mode === "denied")
        throw new DOMException(
          "Synthetic permission rejection",
          "NotAllowedError",
        );
      if (mode === "late")
        return new Promise((resolve) => {
          resolveLate = resolve;
        });
      return makeStream();
    });
    const checks = {};
    try {
      parse(wasm.camera_stop());
      checks.capabilities = parse(await wasm.camera_capabilities());
      checks.noPromptFromEnumeration = calls.length === 0;
      checks.invalidFacing = parse(await wasm.camera_start("", "thermal"));
      checks.invalidId = parse(await wasm.camera_start("x".repeat(513), ""));
      checks.noPromptFromInvalidInput = calls.length === 0;
      checks.noActiveFrame = parse(
        wasm.camera_ingest_rgba(1, 1, new Uint8Array(4)),
      );
      mode = "denied";
      checks.denied = parse(await wasm.camera_start("", "user"));
      mode = "success";
      checks.started = parse(
        await wasm.camera_start("synthetic-camera", "environment"),
      );
      checks.constraints = calls.at(-1);
      checks.duplicate = parse(await wasm.camera_start("", ""));
      checks.oversize = parse(
        wasm.camera_ingest_rgba(641, 1, new Uint8Array(4)),
      );
      checks.badLength = parse(
        wasm.camera_ingest_rgba(2, 2, new Uint8Array(4)),
      );
      const pixels = ctx.getImageData(0, 0, 16, 12).data;
      checks.frame = parse(
        wasm.camera_ingest_rgba(16, 12, new Uint8Array(pixels.buffer)),
      );
      let hash = 0xcbf29ce484222325n;
      for (const byte of pixels)
        hash = BigInt.asUintN(64, (hash ^ BigInt(byte)) * 0x100000001b3n);
      checks.expectedHash = hash.toString(16).padStart(16, "0");
      checks.stopped = parse(wasm.camera_stop());
      checks.stoppedTracks = created.every((stream) =>
        stream.getTracks().every((track) => track.readyState === "ended"),
      );
      checks.frameAfterStop = parse(
        wasm.camera_ingest_rgba(1, 1, new Uint8Array(4)),
      );
      mode = "late";
      const pending = wasm.camera_start("", "user");
      for (let attempt = 0; attempt < 20 && !resolveLate; attempt++)
        await new Promise((resolve) => setTimeout(resolve, 0));
      if (!resolveLate)
        throw new Error("Camera acquisition did not reach getUserMedia");
      parse(wasm.camera_stop());
      const late = makeStream();
      resolveLate(late);
      checks.cancelled = parse(await pending);
      checks.lateStopped = late
        .getTracks()
        .every((track) => track.readyState === "ended");
      mode = "success";
      checks.restarted = parse(await wasm.camera_start("", "environment"));
      checks.facingConstraints = calls.at(-1);
      parse(wasm.camera_stop());
      substitute("enumerateDevices", async () =>
        Array.from({ length: 100 }, (_, i) => ({
          kind: "videoinput",
          deviceId: `synthetic-${i}`,
          label: "x".repeat(1000),
          groupId: "group",
        })),
      );
      checks.bounded = parse(await wasm.camera_capabilities());
      substitute("enumerateDevices", async () => {
        throw new DOMException("Device enumeration blocked", "SecurityError");
      });
      checks.enumerationError = parse(await wasm.camera_capabilities());
      return checks;
    } finally {
      wasm.camera_stop();
      for (const stream of created)
        for (const track of stream.getTracks()) track.stop();
      for (const [name, descriptor] of Object.entries(descriptors)) {
        if (descriptor) Object.defineProperty(manager, name, descriptor);
        else delete manager[name];
      }
    }
  });
  assert.equal(report.capabilities.ok, true);
  assert.equal(report.capabilities.cameras.length, 1);
  assert.match(
    report.capabilities.cameras[0].spectral_classification,
    /^unknown/,
  );
  assert.equal(report.capabilities.thermal_radiometry_verified, false);
  assert.equal(report.noPromptFromEnumeration, true);
  assert.equal(report.invalidFacing.ok, false);
  assert.equal(report.invalidId.ok, false);
  assert.equal(report.noPromptFromInvalidInput, true);
  assert.equal(report.noActiveFrame.ok, false);
  assert.match(report.denied.error, /NotAllowedError/);
  assert.equal(report.started.ok, true);
  assert.equal(report.constraints.audio, false);
  assert.deepEqual(report.constraints.video.deviceId, {
    exact: "synthetic-camera",
  });
  assert.equal(report.constraints.video.width.max, 640);
  assert.equal(report.constraints.video.height.max, 480);
  assert.equal(report.duplicate.ok, false);
  assert.equal(report.oversize.ok, false);
  assert.equal(report.badLength.ok, false);
  assert.equal(report.frame.ok, true);
  assert.equal(report.frame.bytes_ingested, 16 * 12 * 4);
  assert.equal(report.frame.checksum_fnv1a64, report.expectedHash);
  assert.deepEqual(report.frame.channel_min, [10, 20, 30, 255]);
  assert.deepEqual(report.frame.channel_max, [10, 20, 30, 255]);
  assert.deepEqual(report.frame.channel_mean, [10, 20, 30, 255]);
  assert.equal(report.frame.thermal_radiometry_verified, false);
  assert.equal(report.frame.gpu_qualification, false);
  assert.equal(report.stopped.ok, true);
  assert.equal(report.stoppedTracks, true);
  assert.equal(report.frameAfterStop.ok, false);
  assert.equal(report.cancelled.ok, false);
  assert.match(report.cancelled.error, /cancelled/);
  assert.equal(report.lateStopped, true);
  assert.equal(report.restarted.ok, true);
  assert.deepEqual(report.facingConstraints.video.facingMode, {
    ideal: "environment",
  });
  assert.equal(report.bounded.cameras.length, 32);
  assert.equal(report.bounded.enumeration_truncated, true);
  assert.equal(report.bounded.cameras[0].label.length, 512);
  assert.match(report.enumerationError.error, /SecurityError/);
  await runCameraUiContract(page);
  return {
    execution:
      "synthetic browser MediaStream through Rust/WASM and DOM video readback; no physical or IR camera qualification",
    frame_bytes: report.frame.bytes_ingested,
    ui_frame_bytes: 32 * 24 * 4,
  };
}

async function runCameraUiContract(page) {
  await page.evaluate(() => {
    const manager = navigator.mediaDevices;
    const descriptor = Object.getOwnPropertyDescriptor(manager, "getUserMedia");
    const canvas = document.createElement("canvas");
    canvas.width = 32;
    canvas.height = 24;
    const ctx = canvas.getContext("2d");
    const draw = () => {
      ctx.fillStyle = "rgb(10,20,30)";
      ctx.fillRect(0, 0, 32, 24);
    };
    draw();
    const stream = canvas.captureStream(20);
    const timer = setInterval(draw, 50);
    Object.defineProperty(manager, "getUserMedia", {
      configurable: true,
      value: async () => stream,
    });
    window.__etalonSyntheticCamera = {
      stream,
      cleanup() {
        clearInterval(timer);
        for (const track of stream.getTracks()) track.stop();
        if (descriptor)
          Object.defineProperty(manager, "getUserMedia", descriptor);
        else delete manager.getUserMedia;
      },
    };
  });
  try {
    await page.locator("#camera-start").click();
    await page.waitForFunction(
      () => !document.querySelector("#camera-capture").disabled,
      null,
      { timeout: 10000 },
    );
    await page.locator("#camera-capture").click();
    const frame = JSON.parse(
      await page.locator("#camera-frame-report").textContent(),
    );
    assert.equal(frame.ok, true);
    assert.equal(frame.width, 32);
    assert.equal(frame.height, 24);
    assert.equal(frame.bytes_ingested, 32 * 24 * 4);
    assert.equal(frame.thermal_radiometry_verified, false);
    for (const [index, expected] of [10, 20, 30].entries())
      assert.ok(Math.abs(frame.channel_mean[index] - expected) <= 4);
    await page.locator("#camera-stop").click();
    const cleaned = await page.evaluate(() => ({
      detached: document.querySelector("#camera-video").srcObject === null,
      stopped: window.__etalonSyntheticCamera.stream
        .getTracks()
        .every((track) => track.readyState === "ended"),
    }));
    assert.deepEqual(cleaned, { detached: true, stopped: true });
  } finally {
    await page.evaluate(() => {
      document.querySelector("#camera-stop").click();
      window.__etalonSyntheticCamera.cleanup();
      delete window.__etalonSyntheticCamera;
    });
  }
}
