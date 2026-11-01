import assert from "node:assert/strict";
import { test } from "node:test";
import { BrowserDiagnostics } from "./android-diagnostics.js";

test("captures Blink rendering warnings and runtime exceptions, not unrelated protocol payloads", () => {
  const log = new BrowserDiagnostics();
  log.capture({
    method: "Log.entryAdded",
    params: {
      entry: {
        source: "rendering",
        level: "warning",
        text: "Failed to create WebGPU Context Provider",
      },
    },
  });
  log.capture({
    method: "Runtime.exceptionThrown",
    params: {
      exceptionDetails: {
        exception: { description: "GPU initialization failed" },
      },
    },
  });
  log.capture({
    method: "Runtime.consoleAPICalled",
    params: { args: ["ignored"] },
  });
  const result = log.snapshot();
  assert.equal(result.events.length, 2);
  assert.equal(result.events[0].source, "rendering");
  assert.match(result.events[0].text, /Context Provider/);
  assert.equal(result.events[1].source, "runtime-exception");
  assert.equal(result.automatic_upload, false);
});

test("bounds escaped Unicode messages and reports truncation and dropped events", () => {
  const log = new BrowserDiagnostics();
  for (let i = 0; i < 150; i++)
    log.capture({
      method: "Log.entryAdded",
      params: {
        entry: {
          source: "rendering",
          level: "warning",
          text: "\u0001🛰".repeat(1000),
        },
      },
    });
  const result = log.snapshot();
  assert.ok(result.events.length <= result.bounds.events);
  assert.ok(
    Buffer.byteLength(JSON.stringify(result.events)) <=
      result.bounds.event_bytes + 2,
  );
  assert.ok(result.dropped_events > 0);
  assert.ok(result.events.every((e) => e.truncated && !e.text.includes("�")));
  assert.ok(
    result.events.every(
      (e) => Buffer.byteLength(e.text) <= result.bounds.text_bytes,
    ),
  );
});
