# Native Hexagon sensing provider

This is a concrete Rust implementation for execution **on a Qualcomm Hexagon
DSP target**. Construction fails on other target architectures. It does not
implement a Windows/ARM host RPC loader or infer DSP execution from a device
name. A target artifact may also run in an emulator, so architecture gating is
not hardware attestation.

The provider implements `LiveProcessingProvider` and is consumed by the
multi-target `etalon-sense-dsp` dispatcher. It executes the existing revision-1
uncalibrated complex-i16 power operation: little-endian I/Q pairs produce
little-endian f64 values of `I² + Q²`. The full signed-i16 range is computed
exactly using i64 arithmetic; its maximum sum is `2^31`, exactly representable
as f64. There are no heap allocations, SDK calls, hidden sample buffers or CPU
fallbacks in dispatch. Input and output are bounded caller slices, and every
shape/extent/grant check precedes output mutation. Output clearing after a
dispatch failure is provided by the shared dispatcher. This is a scalar
native kernel; it does not claim HVX vectorization or measured performance.

Provider ID and generation are explicit; stop is terminal for an instance.
Restart requires a new instance with a fresh generation and a new binding.
Vendor/model/device registry IDs remain host assertions and must be governed
by the host's identity and policy mechanisms.

## Locally available authoritative platform sources

Installed Rust documentation at
`C:/Users/tim_h/.rustup/toolchains/1.96.1-x86_64-pc-windows-msvc/share/doc/rust/html/rustc/platform-support/hexagon-unknown-none-elf.html`
describes the tier-3 bare-metal Hexagon DSP target, no `std`, no default
allocator, Hexagon calling convention and PIC ELF output. The adjacent
`hexagon-unknown-qurt.html` and `hexagon-unknown-linux-musl.html` describe other
runtime targets. Installed Rust source also contains Hexagon architecture and
compiler-builtins support. These files establish a concrete target interface;
they do not establish a qualified toolchain or device deployment here.

The bare-metal target does not ship prebuilt `core` artifacts according to the
installed documentation. A supported toolchain/core build, compatible target
CPU/runtime, linker/startup, firmware loader or host RPC transport and a host
policy/permission boundary are deployment prerequisites. This library can be
linked into a target-side service using the host's transport; it neither
defines that service ABI nor downloads its dependencies. It deliberately
uses no unsourced vendor FFI.

## Qualification status

Source-only implementation. No compiler, build, test, emulator, DSP, device,
runtime library or network operation has been run. Host tests exercise the
arithmetic kernel and unsupported-target refusal as source specifications;
the target dispatch test requires a suitable native test runtime/harness.
Allocation measurement, cross-target compilation, CPU-oracle comparison,
firmware/RPC integration, policy and timing/cancellation qualification remain
deferred. No claim is made that this DSP can be loaded from the current host.
