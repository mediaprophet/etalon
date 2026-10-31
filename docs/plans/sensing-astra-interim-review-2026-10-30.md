# Interim Astra review — Etalon sensing

This is an independent source-only review of the isolated sensing worktree, not
the final review requested after implementation. No sensing builds, tests,
hardware or network actions were used. Source changes were in progress during
the review, so findings below must be rechecked against the current tree.

## Priority findings

1. A guarded stop consumed its session even when stopping failed, preventing
   retry. The coordinator changed the stop functions to borrow the wrapper;
   a source regression case now covers a failed stop followed by retry.
2. The Windows Wi-Fi BSS information-element offset is entry-relative. The
   source must use checked entry-relative arithmetic before bounds validation;
   fixed with checked addition; local Windows SDK headers confirm the offset basis.
3. The Windows camera byte-copy path allocates WinRT staging objects per frame.
   It must reject strict zero-heap admission or be redesigned around bounded
   reusable/native buffers; it remains feature-gated and unqualified. The copy
   path now rejects hot-zero-heap admission and checks a payload byte budget;
   that budget does not cover WinRT object overhead.
4. WASAPI's per-wait timeout does not bound a whole capture call, and a packet
   larger than the remaining caller frame currently faults rather than buffering
   its remainder. A total deadline/cancellation contract and a bounded packet
   strategy are needed. A host-clock capture-duration budget and retryable
   shutdown path have been added. Oversized packets remain unsupported; a
   post-GetBuffer size mismatch faults and may discard that packet tail.
5. Windows camera time must preserve available raw native ticks without
   claiming a clock mapping. The coordinator added a separate optional native
   time field; source and device behavior remain unverified. Asynchronous
   cancellation and stop still need a complete host-owned lifecycle.
6. The GPU route is real WGSL source but currently rejects strict hot zero-heap,
   uses allocation-permitting command/mapping operations and blocking readback,
   and covers only I/Q power. Signal, runtime and SDR APIs now expose explicit
   allocation admission; the zero-heap backend itself remains outstanding.

## Scope still outstanding

SDR depends on a host-supplied reader rather than a concrete driver; no local
SoapySDR, RTL-SDR or UHD SDK was found. Wi-Fi BSS and BLE advertisements report
RSSI rather than CSI. Cellular measurements remain limited to host declarations
and OS signal bars. A concrete DSP/NPU execution backend was not found. Windows
camera acquisition is explicitly unqualified and feature-gated. Optional SDK
reexports do not by themselves provide an integrated acquisition scheduler.

These findings are implementation work, not device-qualification results.
The final Astra review remains pending after the remaining work is completed.
