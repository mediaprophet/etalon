import assert from "node:assert/strict";
import { test } from "node:test";
import { runTestPage } from "./test-page.js";

function pageWithResult(result, error) {
  const events = new Map();
  return {
    closed: false,
    disposed: false,
    waits: 0,
    on(name, callback) {
      events.set(name, callback);
    },
    async goto() {
      events.get("console")({
        type: () => "error",
        text: () => "GPU diagnostic",
      });
    },
    async waitForFunction() {
      this.waits++;
      if (error) throw error;
      return {
        jsonValue: async () => result,
        dispose: async () => {
          this.disposed = true;
        },
      };
    },
    async close() {
      this.closed = true;
    },
  };
}

test("one completion waiter disposes its result and closes the page", async () => {
  const page = pageWithResult({ passed: true });
  assert.deepEqual(await runTestPage(page, "http://127.0.0.1/test"), {
    status: 200,
    body: "OK",
  });
  assert.equal(page.waits, 1);
  assert.equal(page.disposed, true);
  assert.equal(page.closed, true);
});

test("Rust test failure retains its message and browser diagnostics", async () => {
  const page = pageWithResult({
    passed: false,
    message: "clear range assertion",
  });
  const response = await runTestPage(page, "http://127.0.0.1/test");
  assert.equal(response.status, 500);
  assert.match(response.body, /clear range assertion\nerror: GPU diagnostic/);
  assert.equal(page.closed, true);
});

test("timeout closes its page and returns useful plain text", async () => {
  const page = pageWithResult(null, new Error("Timeout 10000ms exceeded"));
  const response = await runTestPage(page, "http://127.0.0.1/test");
  assert.equal(response.status, 500);
  assert.match(response.body, /Timeout 10000ms exceeded/);
  assert.match(response.body, /GPU diagnostic/);
  assert.equal(page.closed, true);
});

test("browser cleanup errors are reported instead of becoming Express HTML", async () => {
  const page = pageWithResult({ passed: true });
  page.close = async () => {
    throw new Error("browser connection lost");
  };
  const response = await runTestPage(page, "http://127.0.0.1/test");
  assert.equal(response.status, 500);
  assert.match(
    response.body,
    /Failed to close browser page: browser connection lost/,
  );
});
