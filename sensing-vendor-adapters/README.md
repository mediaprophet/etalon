# Isolated sensing vendor adapters

This separate Cargo workspace contains OpenVINO NPU and Qualcomm QNN HTP
model-execution adapters, portable DSP routing, a native Hexagon IQ-power kernel,
and synchronous FastRPC host/service and RTL-SDR receive implementations. Applications select
these packages directly. QNN HTP model execution is distinct from native DSP
kernels. Compatible local runtimes, drivers and hardware remain required.
Hard memory and no-egress enforcement remain host duties.

The adapters pin ORT `2.0.0-rc.15`, with dynamic loading and
without requesting the binary-download feature. Cargo feature unification in a
larger consuming workspace can still enable other features, including downloads.
The independent workspace now retains its own lockfile for reproducible CI.
Rust compilation and mock tests are tracked separately from native runtime and
device qualification; CI success does not establish physical hardware support. Public SDK-interface
reference sources were downloaded with authorization; their pinned provenance
and licenses are retained in `sdk-reference`. No project data was uploaded.

This workspace is separate from the main Etalon wildcard workspace. Earlier
isolated drafts are retained in `sensing-vendor-drafts-archive`; the main facade
and agent do not depend on or re-export these vendor crates. Applications select
the adapters directly from this workspace. Workspace isolation does not itself
enforce runtime network or memory limits.

`etalon-sense-deployment` provides an opt-in typed deployment manifest and
bounded streaming artifact admission before explicit runtime loading. See
`DEPLOYMENT.md`. It validates declared file integrity, not loaded-library or
hardware identity, and performs no downloads.

The FastRPC path shares an original pointer-free wire schema between host and
DSP service. The host uses the pinned public `remote_handle64` C ABI, admits an
explicit runtime and module, and declares native runtime allocation unbounded.
The C skeleton and Rust FFI boundary require a platform SDK image/link/sign
recipe and qualified URI/export resolution. No packaged DSP image, working
board connection or physical execution is claimed. See
`etalon-sense-dsp-service/README.md` and `etalon-sense-fastrpc/README.md`.

`etalon-sense-rtlsdr` is the concrete receive path for an explicitly selected
local librtlsdr runtime/device. It preserves raw offset-binary I/Q and exposes
common observation contracts with unknown physical calibration and timing.
Native synchronous reads may block indefinitely, so this path does not claim
the generic SDR reader's hard-duration contract. Its deployment admission is
separate from permission to enumerate, configure or acquire a device.
