# Native Hexagon IQ power service

This directory implements an original bounded byte-buffer service and DSP-side
skeleton source. `etalon-sense-dsp-wire` is the shared executable interface
definition: its `service-schema.json` fixes byte offsets, method scalars,
encoding and numeric results. This is not QAIC-generated sequence marshalling.
No generated code, packaged DSP image, installed SDK, validated loader on a
specific device, signing result or hardware qualification is claimed.

The safe `HexagonIqPowerService` rejects non-Hexagon execution and dispatches
through the native `HexagonIqPowerProvider`. The kernel computes integer
`I²+Q²` in i64, then writes its exact f64 value in little-endian form. It supplies
no calibration or physical-unit interpretation. It uses caller buffers, no
heap, one synchronous dispatch and no workspace.

Each message has a 64-byte fixed header. Every payload length must match the
nonzero pair count, bounded at 1,048,576 pairs. Maximum request/response sizes
are 4,194,368 and 8,388,672 bytes. Version, reserved fields, operation/revision,
provider/generation and echoed request ID must match. The response header is
published only after native execution succeeds. Every nonzero remote status
invalidates all output. Identity echoes establish correlation, not entity
identity, signature verification, policy approval or trusted execution.

## Forward-loader ABI

`native/etalon_iq_power_skel.c` exports the handle-aware signature established
by official QAIC source:

```c
int etalon_iq_power_skel_handle_invoke(remote_handle64 handle,
    uint32_t scalars, remote_arg *args);
```

The original skeleton implements three exact scalar shapes, with zero method
attributes and no extended-scalar variant:

| Method | Scalar | Native arguments |
| --- | --- | --- |
| Open, 0 | `0x00020001` | int32 URI-byte-length buffer, NUL-terminated URI buffer, one output handle |
| Close, 1 | `0x01000010` | one input handle |
| IQ power, 2 | `0x02010100` | one input byte buffer, one output byte buffer |

Open/close ignore the leading native handle, matching QAIC-generated behavior;
close's target handle is the explicit input handle argument. The opened numeric
handle is supplied as the first entry parameter on operation two. The C shim reads the possibly unaligned open
length using memcpy, checks exact primitive size and exact URI length, and uses
a local output handle. Rust validates the final single NUL and bounded URI.

The shared URI parser accepts this documented forward-loader form:

```text
file:///<module>?etalon_iq_power_skel_handle_invoke&_modver=1.0&etalon_generation=<canonical positive u32 decimal>
```

It allows at most 512 printable nonspace ASCII bytes excluding the C NUL,
with no percent escaping or fragment. The export and module version must match
exactly. Duplicate query keys, abbreviated `_modver` aliases, leading zeroes,
signs, zero/overflowing generation, embedded NULs and trailing/empty attributes
are rejected. Extra nonempty `key=value` attributes may select an explicitly
admitted domain/session; their interpretation is runtime-specific. No domain
name, board or file path is selected automatically. The host independently
parses the same URI before native loading and compares its generation with the
admitted deployment. `etalon_generation` is nonreserved: official remote.idl
explicitly permits forwarding unknown URI keys while reserving underscore keys.

The application byte-buffer schema is original. Generated QAIC sequence
callers are incompatible: their separate primitive length buffers do not match
this service. The native open/close marshalling uses the public lifecycle ABI;
the IQ-power message uses the separate original pointer-free LE schema.

## Native lifecycle and safety boundary

One active context is permitted per resident module image. Open admits a
strictly greater nonzero u32 generation and returns it as an opaque u64 token;
the token is never dereferenced. A full 64-bit comparison rejects upper-bit
aliases. Close clears the active generation while retaining the highest seen
generation. Reopen needs a fresh greater generation, and does not need an image
rebuild. Exhaustion fails rather than wrapping.

A single 32-bit atomic try-lock protects open, close and the entire native
kernel invocation. It never spins or waits: concurrent lifecycle/operation
requests fail. An RAII guard releases the lock on each Rust return. The official
Rust 1.95 `hexagon-unknown-none-elf` target source declares maximum atomic width
32 and abort-on-panic. The `ffi` feature explicitly requires target atomic32
support at compile time; it uses no C11 or 64-bit atomics. This source evidence
does not replace validation on the selected SDK, ISA, runtime or board.

Module unload/reload resets its in-memory high-water mark. The deployment
authority must hold a single active target lease and durably prevent generation
reuse across DSP/process/firmware restarts. It must enforce reset/revocation
serialization and authenticate exact module/URI/domain/session resolution.
The native context also keeps a u64 request-ID high-water mark as two u32
atomics under the same lock. It rejects zero, duplicate and out-of-order IDs;
gaps are allowed. Full wire/provider/generation/capacity validation precedes
admission, and IDs are consumed before kernel work even when that work fails.
Fresh native open resets this request counter. This local duplicate protection
is not persistent anti-replay protection or cryptographic attestation.

The `ffi` feature exports native open/close/process functions for the C shim.
Before creating slices it bounds lengths and rejects null, integer-overflowing
and overlapping ranges. Live allocation provenance, mapping/cache lifetime and
exclusive buffer ownership remain obligations of the unsafe SDK boundary.
Numeric pointer checks do not prove these properties. The final image must use
panic=abort and supply its platform panic handler; this library does not install
process-wide panic behavior. No native Rust/C struct is decoded from the IQ wire.

## Immutable image and deployment requirements

The image recipe defines `ETALON_PROVIDER_ID` and `ETALON_MAX_PAIRS`. The C source
fails compilation when either is absent or invalid. These image properties must
match the digest-bound deployment target and capacity. Session generation is
bound into the exact admitted loader URI and native context instead of being
compiled into the image. No request can choose provider, capacity or generation.

The signed deployment description must bind the image digest, configuration,
schema digest, Rust/C objects, compiler/runtime ABI, firmware, domain, session
and exact loader URI/export. A path or hash alone is not signature verification.
Do not enable unsigned modules automatically. Platform packaging must compile
the shim against the matching remote.h, link the Rust `ffi` implementation and
native provider, export the exact handle-aware entry and retain signing evidence.
Replacing the original skeleton with generated QAIC code requires re-admitting
both sides' method and message contract.

All builds, tests, native linking, runtime loading and device invocations remain
deferred at the user's request. Source-complete is not qualified deployment.
Qualification must cover native ABI size/alignment, malformed scalars and
messages, boundary IQ values, cache/mapping lifetime, native open/close/reopen,
concurrent close/dispatch, stale and upper-bit handles, wraparound, cross-reset
generation authority, revocation and CPU-reference comparison. Source tests are
prepared for the codec, URI grammar and context transitions; none were run.

## Authoritative references

The forward signature and URI distinction are recorded in
`../sdk-reference/qualcomm-qaic/ABI-REVIEW.md` with pinned public QAIC sources
and BSD-3-Clause-Clear license. FastRPC lifecycle marshalling, generated sequence
conventions and remote.h are retained under their upstream BSD-3-Clause notices.
Exact source URLs, lengths and hashes are in the respective provenance records.
The Rust target descriptor and Apache/MIT license texts are retained separately.
No reference source was executed, and no generated implementation was copied
into the original skeleton. Linux GPL driver material remains reference-only.
