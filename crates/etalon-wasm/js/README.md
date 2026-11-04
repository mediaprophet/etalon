`worker_transport.js` owns the browser worker wire transport. Consumers copy this
module into their served asset tree; the Qualia deployment copy is
`docs/js/etalon-worker-transport.js`. Refresh from the repository root with:

```powershell
Copy-Item -LiteralPath coordination/worktrees/etalon-migration/crates/etalon-wasm/js/worker_transport.js -Destination docs/js/etalon-worker-transport.js
```

The module uses actual `crossOriginIsolated` and `SharedArrayBuffer` availability.
The hosting server must supply COOP/COEP headers (or its existing isolation service
worker) for shared rings. Missing isolation or rejected SAB cloning selects the
transferable ArrayBuffer path. Ring overflow also transfers a packet; one wake per
shared slot preserves ordering with transferred packets. No blocking waits run on
the browser thread.

Each direction has 32 slots of 8192 bytes and a 192-byte `SabRingHeader` matching
the Rust ABI. Packets use the 16-byte Rust `MessageHeader` layout, message type 1
for JSON, with a 64 KiB payload limit. Binary model buffers keep their existing
direct ownership-transfer contract, with a 1 GiB default host budget. Consumers
may select a different model budget with the client constructor; its initial
configuration message installs that explicit budget on the worker before model
admission. Checks precede transfer/WASM copies.

Transport admission and stale-completion guards do not grant inference authority.
Hosts continue calling their existing inference export and governance gates.
Qualia's CPU worker binds admission, streaming publication, and completion to the
shared WASM steward. GPU inference remains on its existing browser host.
