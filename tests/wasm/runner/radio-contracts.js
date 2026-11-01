export async function runRadioContracts(page) {
  return page.evaluate(async () => {
    const { loadRuntime } = await import("./wasm-runtime.js");
    const wasm = await loadRuntime();
    const original = Object.getOwnPropertyDescriptor(navigator, "bluetooth");
    const assert = (x, m) => {
      if (!x) throw Error(m);
    };
    const parse = (x) => JSON.parse(x);
    const calls = [];
    let reject = false;
    let delayed = null;
    let choose = null;
    let devices = [];
    class Characteristic extends EventTarget {
      properties = { read: true, write: true, notify: true };
      value = new DataView(new Uint8Array([99, 1, 2, 3, 88]).buffer, 1, 3);
      async readValue() {
        return this.value;
      }
      async writeValueWithResponse(data) {
        calls.push(["write", Array.from(data)]);
      }
      async startNotifications() {
        calls.push(["start"]);
        return this;
      }
      async stopNotifications() {
        calls.push(["stop"]);
        return this;
      }
    }
    function device() {
      const d = new EventTarget();
      d.id = "opaque-origin-id";
      d.name = "Synthetic BLE";
      d.c = new Characteristic();
      d.gatt = {
        connected: false,
        connect: async () => {
          if (delayed) await delayed;
          d.gatt.connected = true;
          return d.gatt;
        },
        disconnect() {
          d.gatt.connected = false;
          calls.push(["disconnect"]);
        },
        async getPrimaryService() {
          return { getCharacteristic: async () => d.c };
        },
      };
      devices.push(d);
      return d;
    }
    const bt = {
      getAvailability: async () => true,
      requestDevice(options) {
        calls.push(["chooser", options]);
        if (reject)
          return Promise.reject(
            new DOMException("Denied for test", "NotAllowedError"),
          );
        if (choose) return choose;
        return Promise.resolve(device());
      },
    };
    Object.defineProperty(navigator, "bluetooth", {
      configurable: true,
      value: bt,
    });
    let session = new wasm.BrowserBluetooth();
    try {
      const cap = parse(await wasm.radio_capabilities());
      assert(cap.bluetooth.available === true, "availability not exposed");
      assert(
        cap.limitations.wifi_scan === false &&
          cap.limitations.bluetooth_mac_address === false,
        "unsupported radio capabilities overstated",
      );
      let r = parse(await session.connect("bad!", "2a19"));
      assert(!r.ok && !calls.length, "invalid UUID opened chooser");
      reject = true;
      r = parse(await session.connect("180f", "2a19"));
      assert(
        !r.ok && r.error.includes("NotAllowedError"),
        "permission failure hidden",
      );
      reject = false;
      r = parse(await session.connect("180f", "2a19"));
      assert(r.ok && r.connected, "connection failed");
      const opts = calls.filter((x) => x[0] === "chooser").at(-1)[1];
      assert(
        opts.filters[0].services[0] === "0000180f-0000-1000-8000-00805f9b34fb",
        "service filter lost",
      );
      r = parse(await session.read());
      assert(r.hex === "010203" && r.bytes === 3, "DataView offset mishandled");
      r = parse(await session.write("ff00"));
      assert(r.ok && calls.at(-1)[1].join(",") === "255,0", "write corrupted");
      const before = calls.length;
      r = parse(await session.write("ff".repeat(513)));
      assert(
        !r.ok && calls.length === before,
        "oversized write crossed boundary",
      );
      r = parse(await session.start_notifications());
      assert(r.ok && r.subscribed, "subscription failed");
      for (let i = 0; i < 40; i++)
        devices.at(-1).c.dispatchEvent(new Event("characteristicvaluechanged"));
      r = parse(session.drain_notifications());
      assert(
        r.notifications.length === 32 &&
          r.dropped === 8 &&
          r.notifications[0].hex === "010203",
        "notification bounds/data wrong",
      );
      assert(
        parse(session.drain_notifications()).notifications.length === 0,
        "drain not consuming",
      );
      devices.at(-1).c.value = new DataView(new ArrayBuffer(513));
      devices.at(-1).c.dispatchEvent(new Event("characteristicvaluechanged"));
      assert(
        parse(session.drain_notifications()).errors === 1,
        "oversized notification not counted",
      );
      await session.stop_notifications();
      session.disconnect();
      let finish;
      delayed = new Promise((resolve) => (finish = resolve));
      const pending = session.connect("180f", "2a19");
      await new Promise((r) => setTimeout(r, 0));
      r = parse(await session.connect("180f", "2a19"));
      assert(!r.ok, "concurrent chooser admitted");
      session.disconnect();
      delayed = null;
      const newer = parse(await session.connect("180f", "2a19"));
      assert(newer.ok, "new connection after cancellation failed");
      const old = devices.at(-2),
        current = devices.at(-1);
      finish();
      assert(!parse(await pending).ok, "cancelled acquisition accepted");
      assert(
        !old.gatt.connected && current.gatt.connected,
        "late completion leaked or disconnected newer device",
      );
      session.disconnect();
      choose = Promise.resolve(current);
      delayed = new Promise((resolve) => (finish = resolve));
      const samePending = session.connect("180f", "2a19");
      await new Promise((r) => setTimeout(r, 0));
      session.disconnect();
      delayed = null;
      assert(
        parse(await session.connect("180f", "2a19")).ok,
        "same-device reconnect failed",
      );
      finish();
      assert(
        !parse(await samePending).ok && current.gatt.connected,
        "stale operation disconnected reused GATT",
      );
      choose = null;
      session.disconnect();
      choose = Promise.resolve(current);
      delayed = new Promise((resolve) => (finish = resolve));
      const retiring = session;
      const retiringPending = retiring.connect("180f", "2a19");
      await new Promise((r) => setTimeout(r, 0));
      retiring.disconnect();
      delayed = null;
      session = new wasm.BrowserBluetooth();
      assert(
        parse(await session.connect("180f", "2a19")).ok,
        "replacement owner failed",
      );
      finish();
      assert(
        !parse(await retiringPending).ok && current.gatt.connected,
        "retiring owner disconnected replacement owner",
      );
      retiring.free();
      choose = null;
      current.gatt.connected = false;
      current.dispatchEvent(new Event("gattserverdisconnected"));
      r = parse(session.drain_notifications());
      assert(!r.connected && !r.subscribed, "remote disconnect not reported");
      const remotelyDisconnected = session;
      choose = Promise.resolve(current);
      session = new wasm.BrowserBluetooth();
      assert(
        parse(await session.connect("180f", "2a19")).ok,
        "remote-disconnect transfer failed",
      );
      remotelyDisconnected.free();
      assert(
        current.gatt.connected,
        "retired owner teardown disconnected successor",
      );
      session.disconnect();
      session.free();
      session = null;
      return {
        status: "pass",
        execution: "Rust/WASM with synthetic Bluetooth browser boundary",
        physical_device_tested: false,
        cases: 12,
      };
    } finally {
      if (session) {
        session.disconnect();
        session.free();
      }
      if (original) Object.defineProperty(navigator, "bluetooth", original);
      else delete navigator.bluetooth;
    }
  });
}
