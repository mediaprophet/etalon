# DSP host RPC and deployment implementation

Status: source implemented; scoped Astra source re-review complete. Builds, tests, code
generation, native library loading and device calls remain deferred. Work is
confined to the isolated sensing worktree and vendor workspace. The original
sensing plans and the shared accelerator/compiler APIs are preserved.

## Source composition

| Library                            | Responsibility                                                                                                      | Allocation boundary                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `etalon-sense-dsp`                 | Exact host-registry target and operation/revision routing, pinned provider descriptor, no alternate-device fallback | Caller-owned bounded bindings and buffers                           |
| `etalon-sense-dsp-hexagon`         | Native Hexagon complex-i16 IQ power into little-endian f64 output                                                   | Zero-heap source path; measurement deferred                         |
| `etalon-sense-dsp-wire`            | Versioned 64-byte header, canonical lengths, echoed identity and numeric-result checks                              | Caller-owned byte slices                                            |
| `etalon-sense-dsp-service`         | Native dispatch, image target/capacity, bounded session lifecycle and Rust/C skeleton boundary                      | Bounded caller buffers and atomic32 context; SDK obligations remain |
| `etalon-sense-fastrpc`             | Linux/Android dynamic host transport, synchronous handle ownership and live-processing provider                     | `ColdUnbounded` native runtime; bounded sample/workspace slices     |
| `etalon-sense-deployment::fastrpc` | Local artifact length/hash admission, authenticated declarations and authority revalidation                         | Caller-owned streaming scratch and aggregate artifact-byte budget   |

The concrete operation is revision 1 of `0x4951504f57455231`: each signed
little-endian i16 I/Q pair becomes its exact integer-valued f64 `I² + Q²`.
It does not infer watts, location, identity, species or calibrated uncertainty.
Etalon's observation/calibration contracts and the QualiaDB sensing bridge
supply the separate physical, semantic, uncertainty and permission layers.

The service has an original direct one-input/one-output byte-buffer ABI.
The handle-aware skeleton reserves methods 0/1 for native open/close; IQ-power
uses method 2, scalar `0x02010100`. Open/close scalars are `0x00020001` and
`0x01000010`. Both endpoints share the exact versioned schema and digest.
This ABI is distinct from QAIC sequence marshalling.
The host uses native `remote_arg`, with pointer plus `size_t` length, for
`remote_handle64_invoke`; the kernel ioctl and `remote_arg64` layouts are not
substitutes.

## Admission and ownership

The authority authenticates the complete runtime/module/firmware dependency
closure, exact module URI/export, domain/session, target and schema. Hashes from
an untrusted manifest do not authenticate code. A protected immutable-artifact
and policy lease must cover validation, loading and every synchronous call.
Native loading remains an explicit unsafe trust boundary.

Requests require exact shape, encoding, provider generation and operation.
Response identity and lengths must match before output publication. Authority
is rechecked around native execution; used workspace and failure output are
cleared. Borrowed buffers remain alive until synchronous return. There is no
timeout that frees an in-flight buffer, CPU fallback or unsigned-module
downgrade. Unknown native open/close outcomes retain the runtime mapping rather
than guessing whether remote/global resources have stopped.

QualiaDB owns human-centered authorization, Sanctuary/guardianship, ontology,
inference and evidence custody. These transport declarations neither replace
those mechanisms nor turn a device identifier into authenticated entity identity.

## SDK and deployment prerequisites

Public Qualcomm FastRPC interface sources and examples are retained with pinned
commits, source URLs, byte counts, hashes and licenses under
`sensing-vendor-adapters/sdk-reference`. Linux GPL driver sources are reference
material, not linked implementation. Only public reference downloads were
performed; project data was not uploaded and downloaded code was not executed.

A real deployment still needs a target-compatible Hexagon compiler/toolchain,
SDK/runtime, firmware, board driver and native image/link/sign recipe. The
native Rust target source is distinct from a packaged loadable SDK module.
The authority must qualify exact loader/export resolution, signing policy,
buffer registration/cache behavior, processor lifecycle and dependency closure.
No board, signed image or hardware support has been qualified.

## Review disposition

Astra identified runtime-global open/close concurrency and missing enforced
host/image generation/capacity compatibility. Both corrections passed scoped
source re-review: runtime lifecycle calls are serialized across crate sessions,
and host generation/capacity are checked against authenticated declarations
before loading. This does not serialize unrelated native bindings; the unsafe
constructor explicitly requires host-wide coordination for those bindings.

Subsequent official QAIC source inspection established that the documented
forward-loader path uses a three-argument handle-aware skeleton and reserves
methods 0/1 for native open/close. The initial two-argument skeleton was based
on a reverse-call example and is not established compatible with that path.
The corrected three-argument `_skel_handle_invoke` follows the documented
forward-loader ABI. The native context admits one active generation through a
bounded atomic32 try-lock, holds its lease through dispatch, rejects stale/full
64-bit-invalid handles, and retains generation high-water across close. A new
admitted `etalon_generation` URI value reuses the image without recompilation.
Cross-module-reload freshness remains the host authority's responsibility.
Astra re-reviewed the forward correction and found no remaining concrete
blocker by inspection. Per-context request high-water strengthening is also
implemented and re-reviewed: complete wire/identity/size checks precede admission,
IDs are consumed before kernel work even on failure, and duplicate/out-of-order
requests are rejected. Fresh generation admission resets only the request counter.
No deployment claim relies on the initial example.

The final schema SHA-256 is
`bee23c17499ab0db8a7309372ac8b2d09855dcedb8258bda1dbd3a04771da029`.
Static reference-file hashing matched all 26 pinned reference files (376,493
bytes total) and the schema constant; this did not run target code or tests.
The broader review and improvement instructions are retained in the isolated
QualiaDB worktree's `docs/plans/sensing-astra-review-2026-10-31.md`.

## Deferred verification

After the cleanup hold is lifted, qualification must cover dependency and target
compilation, Rust/C size/alignment comparison with the selected SDK, wire and
scalar rejection, CPU-oracle comparisons including extreme I/Q values, stale
generation/replay/reset/revocation, concurrent session cleanup, failure and unwind
wiping, signed-image refusal, and real device/firmware behavior. Allocation and
latency measurements must distinguish the native kernel from the unbounded host
runtime. Source inspection does not establish any of these execution results.
