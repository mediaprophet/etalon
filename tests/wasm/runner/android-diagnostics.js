const MAX_EVENTS = 128;
const MAX_BYTES = 64 * 1024;
const MAX_TEXT_BYTES = 1024;

function bounded(value) {
  const bytes = Buffer.from(typeof value === "string" ? value : "", "utf8");
  let end = Math.min(bytes.length, MAX_TEXT_BYTES);
  while (end < bytes.length && end > 0 && (bytes[end] & 0xc0) === 0x80) end--;
  return {
    text: bytes.subarray(0, end).toString("utf8"),
    truncated: end < bytes.length,
  };
}

export class BrowserDiagnostics {
  constructor() {
    this.events = [];
    this.bytes = 0;
    this.dropped = 0;
  }

  capture(message) {
    let source, level, raw;
    if (message.method === "Log.entryAdded") {
      const entry = message.params?.entry;
      source = bounded(entry?.source).text;
      level = bounded(entry?.level).text;
      raw = entry?.text;
    } else if (message.method === "Runtime.exceptionThrown") {
      const details = message.params?.exceptionDetails;
      source = "runtime-exception";
      level = "error";
      raw = details?.exception?.description ?? details?.text;
    } else return;
    const event = { source, level, ...bounded(raw) };
    const size = Buffer.byteLength(JSON.stringify(event)) + 1;
    if (this.events.length >= MAX_EVENTS || this.bytes + size > MAX_BYTES) {
      this.dropped++;
      return;
    }
    this.events.push(event);
    this.bytes += size;
  }

  snapshot() {
    return {
      schema: "etalon.browser-cdp-log.v1",
      automatic_upload: false,
      bounds: {
        events: MAX_EVENTS,
        event_bytes: MAX_BYTES,
        text_bytes: MAX_TEXT_BYTES,
      },
      dropped_events: this.dropped,
      events: this.events.slice(),
    };
  }
}
