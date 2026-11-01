import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { webcrypto } from "node:crypto";
const source = await readFile(
  new URL("../../../docs/site/service-worker.js", import.meta.url),
  "utf8",
);
const scope = "https://example.test/etalon/";
const prefix = `etalon-pwa-v1:${scope}:`;
const A = "a".repeat(40),
  B = "b".repeat(40),
  C = "c".repeat(40);
const key = (value) => value.url || String(value);
class Store {
  constructor() {
    this.entries = new Map();
  }
  async match(url) {
    return this.entries.get(key(url))?.clone();
  }
  async put(url, response) {
    this.entries.set(key(url), response.clone());
  }
}
function storage() {
  const values = new Map();
  return {
    values,
    async keys() {
      return [...values.keys()];
    },
    async open(name) {
      if (!values.has(name)) values.set(name, new Store());
      return values.get(name);
    },
    async delete(name) {
      return values.delete(name);
    },
  };
}
const worker = (revision) => ({
  postMessage(_message, ports) {
    ports[0].postMessage({ revision });
  },
});
async function harness(
  caches,
  revision,
  { active = null, waiting = null, invalid = false, oversize = false } = {},
) {
  const body = new TextEncoder().encode("bounded app");
  const hash = Buffer.from(
    await webcrypto.subtle.digest("SHA-256", body),
  ).toString("hex");
  const manifest = {
    revision,
    assets: [
      {
        path: "testing.html",
        bytes: oversize ? body.length - 1 : body.length,
        sha256: invalid ? "0".repeat(64) : hash,
      },
    ],
  };
  const handlers = {};
  const context = vm.createContext({
    URL,
    Response,
    Uint8Array,
    TextEncoder,
    TextDecoder,
    MessageChannel,
    setTimeout,
    clearTimeout,
    crypto: webcrypto,
    caches,
    fetch: async (url) =>
      new Response(
        new URL(url).pathname.endsWith("offline-assets.json")
          ? JSON.stringify(manifest)
          : body,
        { headers: { "Content-Type": "text/html" } },
      ),
    self: {
      registration: {
        scope,
        active: active ? worker(active) : null,
        waiting: waiting ? worker(waiting) : null,
      },
      clients: { claim: async () => {} },
      addEventListener: (name, handler) => (handlers[name] = handler),
    },
  });
  vm.runInContext(source.replaceAll("{{REVISION}}", revision), context);
  return {
    async request(url, mode) {
      let pending;
      handlers.fetch({
        request: { url, method: "GET", mode },
        respondWith(value) {
          pending = value;
        },
      });
      return pending;
    },
    async run(name) {
      let pending;
      handlers[name]({
        waitUntil(value) {
          pending = value;
        },
      });
      await pending;
    },
  };
}

test("interrupted install is recovered without deleting the active or unrelated cache", async () => {
  const caches = storage();
  const a = await harness(caches, A);
  await a.run("install");
  await caches.open(prefix + B);
  await caches.open("unrelated-user-cache");
  await caches.open(prefix + "not-a-revision");
  const c = await harness(caches, C, { active: A });
  await c.run("install");
  assert(caches.values.has(prefix + A));
  assert(caches.values.has(prefix + C));
  assert(!caches.values.has(prefix + B));
  assert(caches.values.has("unrelated-user-cache"));
  assert(caches.values.has(prefix + "not-a-revision"));
  await c.run("activate");
  assert(!caches.values.has(prefix + A));
  assert(caches.values.has(prefix + C));
  assert(caches.values.has("unrelated-user-cache"));
});
test("a complete abandoned cache is recovered but a waiting installed revision is preserved", async () => {
  const caches = storage();
  await (await harness(caches, A)).run("install");
  await (await harness(caches, B, { active: A })).run("install");
  await assert.rejects(
    (await harness(caches, C, { active: A, waiting: B })).run("install"),
    /Close application windows/,
  );
  assert(caches.values.has(prefix + A));
  assert(caches.values.has(prefix + B));
  assert(!caches.values.has(prefix + C));
  await (await harness(caches, C, { active: A })).run("install");
  assert(!caches.values.has(prefix + B));
});
for (const option of ["invalid", "oversize"]) {
  test(`failed ${option} bundle preserves active revision and removes partial cache`, async () => {
    const caches = storage();
    await (await harness(caches, A)).run("install");
    await assert.rejects(
      (await harness(caches, C, { active: A, [option]: true })).run("install"),
    );
    assert(caches.values.has(prefix + A));
    assert(!caches.values.has(prefix + C));
  });
}

test("active app revision stays coherent across a newer-version deep link", async () => {
  const caches = storage();
  const app = await harness(caches, A);
  await app.run("install");
  const navigation = await app.request(
    scope + "testing.html?rev=" + B,
    "navigate",
  );
  assert.equal(navigation.status, 200);
  assert.equal(await navigation.text(), "bounded app");
  const wasm = await app.request(
    scope + "browser/etalon_browser_demo_bg.wasm?rev=" + B,
    "cors",
  );
  assert.equal(wasm.status, 409);
  assert.equal(wasm.headers.get("Cache-Control"), "no-store");
});
