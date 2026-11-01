const REVISION = "{{REVISION}}";
const PREFIX = `etalon-pwa-v1:${self.registration.scope}:`;
const CACHE = PREFIX + REVISION;
const MAX_BYTES = 40 * 1024 * 1024;
const MAX_ASSETS = 160;
const MANIFEST = new URL("offline-assets.json", self.registration.scope);

async function boundedBytes(response, limit) {
  if (!response.ok || !response.body)
    throw new Error("Offline asset unavailable");
  const reader = response.body.getReader();
  const result = new Uint8Array(limit);
  let size = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > limit)
        throw new Error("Offline asset exceeds its byte budget");
      result.set(value, size - value.byteLength);
    }
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally {
    reader.releaseLock();
  }
  return result.subarray(0, size);
}

function assetUrl(path) {
  if (
    typeof path !== "string" ||
    !/^[a-zA-Z0-9_./-]+$/.test(path) ||
    path.startsWith("/") ||
    path.split("/").includes("..")
  ) {
    throw new Error("Invalid offline asset path");
  }
  const url = new URL(path, self.registration.scope);
  if (!url.href.startsWith(self.registration.scope))
    throw new Error("Asset outside application scope");
  return url;
}

async function protectedRevision(worker) {
  if (!worker) return null;
  return new Promise((resolve, reject) => {
    const channel = new MessageChannel();
    const finish = (error, revision) => {
      clearTimeout(timer);
      channel.port1.close();
      channel.port2.close();
      error ? reject(error) : resolve(revision);
    };
    const timer = setTimeout(
      () => finish(new Error("Cannot identify installed offline revision")),
      3000,
    );
    channel.port1.onmessage = (event) => {
      const revision = event.data?.revision;
      if (typeof revision !== "string" || !/^[0-9a-f]{40}$/.test(revision))
        finish(new Error("Invalid installed revision"));
      else finish(null, revision);
    };
    try {
      worker.postMessage({ type: "ETALON_APP_STATUS" }, [channel.port2]);
    } catch (error) {
      finish(error);
    }
  });
}

async function installBundle() {
  const manifestUrl = new URL(MANIFEST);
  manifestUrl.searchParams.set("rev", REVISION);
  const bytes = await boundedBytes(
    await fetch(manifestUrl, { cache: "no-store" }),
    65536,
  );
  const manifest = JSON.parse(new TextDecoder().decode(bytes));
  if (
    manifest.revision !== REVISION ||
    !Array.isArray(manifest.assets) ||
    manifest.assets.length < 1 ||
    manifest.assets.length > MAX_ASSETS
  ) {
    throw new Error("Offline manifest revision or bounds invalid");
  }
  let total = 0;
  const paths = new Set();
  for (const asset of manifest.assets) {
    assetUrl(asset.path);
    if (
      paths.has(asset.path) ||
      !Number.isSafeInteger(asset.bytes) ||
      asset.bytes < 0 ||
      !/^[0-9a-f]{64}$/.test(asset.sha256)
    )
      throw new Error("Invalid asset integrity record");
    paths.add(asset.path);
    total += asset.bytes;
    if (total > MAX_BYTES)
      throw new Error("Offline bundle exceeds its byte budget");
  }
  const protectedKeys = new Set([CACHE]);
  for (const worker of [self.registration.active, self.registration.waiting]) {
    const revision = await protectedRevision(worker);
    if (revision) protectedKeys.add(PREFIX + revision);
  }
  const owned = (key) =>
    key.startsWith(PREFIX) && /^[0-9a-f]{40}$/.test(key.slice(PREFIX.length));
  for (const key of await caches.keys()) {
    if (owned(key) && !protectedKeys.has(key)) await caches.delete(key);
  }
  const existing = (await caches.keys()).filter(owned);
  if (!existing.includes(CACHE) && existing.length >= 2)
    throw new Error(
      "Close application windows to complete the waiting update before downloading another revision",
    );
  const cache = await caches.open(CACHE);
  const completed = await cache.match(MANIFEST);
  if (completed) {
    const saved = await completed.json();
    if (
      saved.revision === REVISION &&
      JSON.stringify(saved.assets) === JSON.stringify(manifest.assets)
    ) {
      let intact = true;
      for (const asset of manifest.assets)
        if (!(await cache.match(assetUrl(asset.path)))) intact = false;
      if (intact) return;
    }
  }
  try {
    for (const asset of manifest.assets) {
      const url = assetUrl(asset.path);
      const requestUrl = new URL(url);
      requestUrl.searchParams.set("rev", REVISION);
      const response = await fetch(requestUrl, { cache: "no-store" });
      const body = await boundedBytes(response, asset.bytes);
      if (body.byteLength !== asset.bytes)
        throw new Error("Offline asset length mismatch");
      const digest = Array.from(
        new Uint8Array(await crypto.subtle.digest("SHA-256", body)),
        (value) => value.toString(16).padStart(2, "0"),
      ).join("");
      if (digest !== asset.sha256)
        throw new Error("Offline asset revision/integrity mismatch");
      await cache.put(
        url,
        new Response(body, {
          headers: {
            "Content-Type":
              response.headers.get("Content-Type") ||
              "application/octet-stream",
          },
        }),
      );
    }
    await cache.put(
      MANIFEST,
      new Response(JSON.stringify(manifest), {
        headers: { "Content-Type": "application/json" },
      }),
    );
  } catch (error) {
    await caches.delete(CACHE);
    throw error;
  }
}

self.addEventListener("install", (event) => event.waitUntil(installBundle()));
self.addEventListener("activate", (event) =>
  event.waitUntil(
    (async () => {
      for (const key of await caches.keys()) {
        if (
          key !== CACHE &&
          key.startsWith(PREFIX) &&
          /^[0-9a-f]{40}$/.test(key.slice(PREFIX.length))
        )
          await caches.delete(key);
      }
      await self.clients.claim();
    })(),
  ),
);
self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (
    request.method !== "GET" ||
    !url.href.startsWith(self.registration.scope) ||
    (url.searchParams.has("rev") && url.searchParams.get("rev") !== REVISION)
  )
    return;
  event.respondWith(
    (async () => {
      url.search = "";
      url.hash = "";
      if (url.href === self.registration.scope) url.pathname += "index.html";
      const cache = await caches.open(CACHE);
      return (await cache.match(url)) || fetch(request);
    })(),
  );
});
self.addEventListener("message", (event) => {
  if (event.data?.type !== "ETALON_APP_STATUS" || !event.ports[0]) return;
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      const response = await cache.match(MANIFEST);
      const manifest = response ? await response.json() : null;
      event.ports[0].postMessage({
        revision: REVISION,
        offline_ready: manifest?.revision === REVISION,
        cache_name: CACHE,
        asset_count: manifest?.assets?.length || 0,
        asset_bytes:
          manifest?.assets?.reduce((sum, asset) => sum + asset.bytes, 0) || 0,
      });
    })(),
  );
});
