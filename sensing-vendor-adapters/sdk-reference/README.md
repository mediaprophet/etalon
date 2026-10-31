# Public DSP SDK and FastRPC source review

Reviewed 2026-10-31 for Etalon sensing. This directory contains public source
references, not an installed SDK, executable toolchain, firmware, or a qualified
hardware backend. Nothing downloaded here was executed, compiled, linked, or
loaded onto a device. No project data was uploaded.

Upstream files here keep their own licenses and are ring-fenced from the rest of
Etalon. See [`THIRD-PARTY-LICENSES.md`](THIRD-PARTY-LICENSES.md).

## Sources retained locally

The initial nine downloaded files are pinned to immutable upstream commits. Exact
URLs, byte lengths, SHA-256 digests and download UTC are in
[`provenance.json`](provenance.json). The source payload totals **181,061 bytes**;
downloads were limited to 196,608 bytes per file and 524,288 bytes in total.

An additional 54,111 bytes of pinned Qualcomm BSD-3-Clause source establish the
public skeleton signatures, lifecycle scalar layout, and QAIC sequence
length-buffer convention: `adspmsgd_apps_skel.c`, `adspmsgd_apps.idl`,
`AEEStdDef.idl`, `mod_table.c`, and `mod_table.h`. Their exact URLs,
lengths and SHA-256 digests are recorded separately in
[`service-reference-provenance.json`](qualcomm-fastrpc/service-reference-provenance.json).
No additional reference source was compiled or executed. Etalon's original
byte-buffer service deliberately uses a distinct explicit ABI rather than
claiming compatibility with generated QAIC sequence marshalling. See
[`etalon-sense-dsp-service`](../etalon-sense-dsp-service/README.md) for the
implementation and remaining DSP loader/signing qualification boundary.

The official public QAIC source is also retained at immutable commit
`a66a0019fb488abae955bbc9b3a9de3dc15962d0`: nine source/documentation files totaling
129,286 bytes, under its distinct BSD-3-Clause-Clear license. No QAIC compiler
or code generation ran. Exact pins and hashes are in
[`qualcomm-qaic/provenance.json`](qualcomm-qaic/provenance.json), and the
[forward ABI review](qualcomm-qaic/ABI-REVIEW.md) records the important distinction
between handle-free and handle-aware loader entries.

The official Rust 1.95 Hexagon target descriptor and Apache/MIT license texts
add 12,035 bytes, pinned in `rust-target/provenance.json`. The descriptor declares
32-bit maximum atomic width and panic-abort, supporting the service's explicit
atomic32 target gate; selected SDK/toolchain/hardware qualification is deferred.
Total retained payload across all four provenance records is 376,493 bytes.

| Source | Pin | Retained material | License boundary |
| --- | --- | --- | --- |
| [Qualcomm FastRPC](https://github.com/qualcomm/fastrpc/tree/a56e9d4de3614a8d8f4b0e15ff561350c4b686e3) | `a56e9d4de3614a8d8f4b0e15ff561350c4b686e3` | README, `remote.h`, `rpcmem.h`, vendor `fastrpc_ioctl.h`, license | BSD-3-Clause; retain source notices and license when redistributing |
| [Linux v6.18](https://github.com/torvalds/linux/tree/7d0a66e4bb9081d75c82ec4957c50034cb0ea449) | `7d0a66e4bb9081d75c82ec4957c50034cb0ea449` | UAPI `fastrpc.h`, driver `fastrpc.c`, GPL-2.0 text, Linux syscall exception | UAPI: GPL-2.0 WITH Linux-syscall-note. Driver: GPL-2.0, reference only |

The GPL driver implementation must remain a review reference; it is not copied
into an Etalon implementation or included as a Rust build input. The UAPI has
its own explicit syscall exception. Keep these distinctions visible when
packaging this reference directory; do not label the complete directory MIT.

## Suitable concrete DSP platform

Qualcomm Hexagon with Linux FastRPC is a viable concrete target for a flexible
DSP interface. It has public, authoritative host transport definitions and a
vendor-supported DSP toolchain path. This is a platform selection, not evidence
that a compatible DSP, runtime, firmware, SDK or signing arrangement is present
on this Windows development machine.

The [official Hexagon NPU SDK page](https://www.qualcomm.com/developer/software/hexagon-npu-sdk)
requires a Qualcomm login to download and presents a product license agreement.
No login, license acceptance, installer, SDK archive, toolchain, proprietary
runtime, or firmware was obtained. The public FastRPC sources do not constitute
the full Hexagon SDK. SDK acquisition and use must follow Qualcomm's normal
access and license process; download authorization does not confer acceptance
of additional SDK terms.

## What the public interfaces establish

The [Qualcomm README](qualcomm-fastrpc/README.md) describes IDL compiled with
QAIC into a CPU stub and DSP skeleton. The stub and skeleton marshal the same
interface; the host userspace library communicates through the kernel driver
and RPMsg with the DSP runtime. The DSP skeleton must be linked with the DSP
implementation. A freestanding Rust Hexagon function is not automatically a
loadable FastRPC module.

[`remote.h`](qualcomm-fastrpc/remote.h) establishes these host ABI facts:

- `remote_handle64` is `uint64_t`; `remote_handle64_open`,
  `remote_handle64_invoke`, and `remote_handle64_close` return C `int` status.
- Despite its name, `remote_handle64_invoke` takes `remote_arg *`, not
  `remote_arg64 *`. `remote_arg` contains a pointer/`size_t` buffer, 32-bit and
  64-bit handles, and a DMA descriptor union. Host pointer width and alignment
  therefore matter.
- Scalar bits encode method attributes [31:29], method [28:24], input buffers
  [23:16], output buffers [15:8], input handles [7:4], and output handles [3:0].
  Argument order is input buffers, output buffers, input handles, output handles.
- The header advises applications to use generated QAIC stubs. These public
  generic declarations do not supply Etalon's method ordinals, parameter
  marshaling, skeleton symbol, result encoding, or module manifest.
- Module open/close requires synchronization; a handle is process-local. Opening
  an unsigned module requires a separate explicit runtime control. Etalon must
  never silently enable that control or downgrade a signed deployment policy.

[`linux/fastrpc.h`](linux/fastrpc.h) establishes the upstream Linux syscall
surface: DMA allocation, invocation, protection-domain initialization, mapping,
unmapping and capability requests. The invoke structure uses a 32-bit handle,
32-bit scalar and 64-bit argument-array address. Its arguments use 64-bit
pointer/length, signed 32-bit DMA fd and 32-bit attributes. This kernel-facing
32-bit handle is not interchangeable with a Qualcomm userspace 64-bit handle.

The vendor ioctl header and upstream Linux UAPI are separate interfaces; do not
mix their ioctl numbers or structures merely because both refer to FastRPC.
Presence of a device node or an API symbol also does not prove availability of
a permitted DSP module. Public transport definitions do not establish board
firmware compatibility or signing authority.

## Implementation boundary and next integration

The portable Etalon DSP contract should route by explicit target, provider and
generation, with bounded caller buffers and an explicit operation capability.
A native Hexagon IQ-power implementation can provide one concrete DSP kernel;
its native-target status must remain distinct from host-offload deployment.
Neither a QNN NPU backend nor a CPU implementation should advertise itself as
DSP execution. `Require(Dsp)` must fail when the selected DSP is unavailable.

A Linux host backend is viable after these concrete inputs are supplied:

1. A selected board, DSP domain and supported kernel/userspace/runtime versions,
   together with a lawful local SDK/toolchain and its applicable terms.
2. An Etalon-owned, versioned service definition for the exact kernel operation,
   matching CPU stub and DSP skeleton, and exported implementation with checked
   C ABI. A generated QAIC pair or a separately qualified original pair must
   share exactly one wire contract. Record interface hash, method ordinals and
   bounded request/response formats. Original service source now exists, with
   platform loader/link/signing qualification deferred.
3. An explicitly selected local runtime library and DSP module with digest,
   signer/verification result, target ISA, domain and firmware/runtime binding.
   A path or configured hash by itself is not signature verification.
4. A synchronized owner for session open/invoke/close and all DMA/cache lifetimes.
   Keep mapped input/output buffers alive through completion; use the runtime's
   required cache maintenance and supported allocator rather than assuming any
   Rust slice is shareable DSP memory.
5. Checked sizes and signed integer conversions before FFI, exact output length
   and operation validation after return, timeout/cancellation semantics that do
   not release in-flight buffers, and explicit device-loss failure.
6. Deployment receipts which distinguish requested target, verified module,
   actual execution backend, completion and transport failure. QualiaDB owns
   semantic permissions, identity correlation, uncertainty and custody rules;
   Etalon supplies verifiable measurements and execution evidence.

Do not invent a generic scalar/method pair and call it a sensing wire protocol.
Direct ioctl invocation without process/module initialization and matching DSP
runtime dispatch is insufficient. No host FastRPC transport is claimed complete
by this reference import; a fabricated deployment path would obscure the real
remaining integration.

Qualification remains deferred as requested. Later qualification needs a real
supported device, ABI/layout checks against the selected SDK, stub/skeleton
round trips, signed-module rejection cases, DMA/coherency checks, device-loss
handling and comparison with a bounded CPU oracle. No build or test was run for
this source survey.
