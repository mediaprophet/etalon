import { pathToFileURL } from "node:url";
import { setTimeout as pause } from "node:timers/promises";

export const MAX_REPORT_BYTES = 2 * 1024 * 1024;

// This connection owns a page debugging socket, never a browser session.
export class CdpClient {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 0;
    this.pending = new Map();
    socket.addEventListener("message", ({ data }) => {
      try {
        if (
          typeof data !== "string" ||
          Buffer.byteLength(data) > MAX_REPORT_BYTES * 2
        )
          throw new Error("DevTools message byte budget exceeded");
        const message = JSON.parse(data);
        const request = this.pending.get(message.id);
        if (!request) return;
        this.pending.delete(message.id);
        clearTimeout(request.timer);
        if (message.error)
          request.reject(new Error(JSON.stringify(message.error)));
        else request.resolve(message.result);
      } catch (error) {
        this.fail(error);
      }
    });
    socket.addEventListener("close", () =>
      this.fail(new Error("Page debugging socket closed")),
    );
    socket.addEventListener("error", () =>
      this.fail(new Error("Page debugging socket failed")),
    );
  }

  static async connect(endpoint, timeoutMs = 5000, Socket = WebSocket) {
    const address = new URL(endpoint);
    if (
      address.protocol !== "ws:" ||
      address.hostname !== "127.0.0.1" ||
      !address.port ||
      !address.pathname.startsWith("/devtools/page/") ||
      address.search ||
      address.hash
    )
      throw new Error("Expected a forwarded loopback page endpoint");
    const socket = new Socket(endpoint);
    const client = new CdpClient(socket);
    try {
      await new Promise((resolve, reject) => {
        const timer = setTimeout(
          () => reject(new Error("Page socket connection timed out")),
          timeoutMs,
        );
        const finish = (error) => {
          clearTimeout(timer);
          error ? reject(error) : resolve();
        };
        socket.addEventListener("open", () => finish(), { once: true });
        socket.addEventListener(
          "error",
          () => finish(new Error("Page socket connection failed")),
          { once: true },
        );
        socket.addEventListener(
          "close",
          () => finish(new Error("Page socket closed before connection")),
          { once: true },
        );
      });
      return client;
    } catch (error) {
      client.close();
      throw error;
    }
  }

  call(method, params, timeoutMs) {
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`DevTools ${method} timed out`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      try {
        this.socket.send(JSON.stringify({ id, method, params }));
      } catch (error) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(error);
      }
    });
  }

  fail(error) {
    for (const request of this.pending.values()) {
      clearTimeout(request.timer);
      request.reject(error);
    }
    this.pending.clear();
  }

  close() {
    this.fail(new Error("Page debugging session ended"));
    this.socket.close();
  }
}

export async function collectReport(
  client,
  url,
  timeoutMs,
  wait = pause,
  now = Date.now,
) {
  const deadline = now() + timeoutMs;
  const expression = `(() => {
    if (location.href !== ${JSON.stringify(url)}) throw new Error("Qualification page navigated away");
    const state = document.body?.dataset.etalonState;
    if (state !== "complete") return { state };
    const report = document.querySelector("#qualification-report")?.textContent;
    if (!report) throw new Error("Completed page has no qualification report");
    if (new TextEncoder().encode(report).byteLength > ${MAX_REPORT_BYTES})
      throw new Error("Qualification report byte budget exceeded");
    return { state, report };
  })()`;
  while (now() < deadline) {
    const result = await client.call(
      "Runtime.evaluate",
      { expression, returnByValue: true },
      Math.min(5000, deadline - now()),
    );
    if (result.exceptionDetails)
      throw new Error(JSON.stringify(result.exceptionDetails));
    const value = result.result?.value;
    if (value?.state === "complete") {
      if (
        typeof value.report !== "string" ||
        Buffer.byteLength(value.report) > MAX_REPORT_BYTES
      )
        throw new Error("Invalid or oversized qualification report");
      const report = JSON.parse(value.report);
      if (!report || typeof report !== "object" || Array.isArray(report))
        throw new Error("Qualification report must be an object");
      return value.report;
    }
    await wait(Math.min(200, Math.max(0, deadline - now())));
  }
  throw new Error("Rust/WASM qualification did not complete before timeout");
}

async function main() {
  const raw = process.argv.slice(2);
  const allowed = new Set(["--endpoint", "--url", "--timeout"]);
  if (
    raw.length !== 6 ||
    raw.some((value, i) => i % 2 === 0 && !allowed.has(value))
  )
    throw new Error(
      "Required: --endpoint PAGE_WS --url OWN_PAGE_URL --timeout SECONDS",
    );
  const args = Object.fromEntries([0, 2, 4].map((i) => [raw[i], raw[i + 1]]));
  const seconds = Number(args["--timeout"]);
  const url = new URL(args["--url"]);
  if (
    !Number.isInteger(seconds) ||
    seconds < 1 ||
    seconds > 600 ||
    url.protocol !== "http:" ||
    url.hostname !== "127.0.0.1" ||
    !url.searchParams.get("etalon-run")
  )
    throw new Error("Invalid qualification URL or timeout");
  const client = await CdpClient.connect(args["--endpoint"]);
  try {
    process.stdout.write(await collectReport(client, url.href, seconds * 1000));
  } finally {
    client.close();
  }
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  main().catch((error) => {
    process.stderr.write(`${String(error).slice(0, 2000)}\n`);
    process.exitCode = 1;
  });
}
