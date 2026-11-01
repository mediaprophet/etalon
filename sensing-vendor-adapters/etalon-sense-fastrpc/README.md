# Etalon synchronous FastRPC host provider

Source implementation for a Linux/Android host and the original Etalon IQ-power
DSP service. This is a vendor workspace extension; it does not change the Etalon
root manifest, accelerator APIs or compiler code. CI compiles its Rust boundary
and runs mock tests; native library loading and DSP execution remain separate
deployment qualifications. It is not qualified hardware support.

`FastRpcIqPowerProvider::open` takes a validated
`etalon_sense_deployment::fastrpc::AdmittedFastRpcDeployment`, an unreused provider
generation and an explicit IQ-pair limit. It checks the service ID and exact
schema digest against `etalon-sense-dsp-wire`, parses the exact service export
and module version from its canonical URI, matches the URI session generation
against the admitted declaration and bounds capacity by the admitted image
capacity, then loads the admitted absolute runtime path with local, immediate
symbol resolution. Opening is explicitly
unsafe: artifact digests do not establish trustworthy native code or a correct
C ABI. The host must authenticate the complete runtime/module/firmware closure,
signatures, domain/session policy, image compatibility, URI-to-export binding,
library dependency resolution and an immutable authority lease. Deployment
admission does not reserve a session or configure a protection domain.
The service image has fixed provider ID/capacity. A canonical
`etalon_generation` URI attribute binds each session to its admitted generation.
The service's bounded single-context lifecycle rejects concurrent duplicate
sessions and nonincreasing generations. Its high-water state only persists for
the loaded module's lifetime; authority must prevent generation reuse across
module reloads, process/device restart and rollback.

The ABI re-expresses the pinned BSD-licensed Qualcomm
[`remote.h`](../sdk-reference/qualcomm-fastrpc/remote.h):
`remote_handle64_open`, `remote_handle64_invoke`, `remote_handle64_close`.
The invoke uses **native `remote_arg`**, whose buffer is `void*` plus `size_t`;
it does not use `remote_arg64`, the kernel ioctl ABI or guessed QAIC sequence
marshalling. Attribution and upstream license are retained in
[`sdk-reference`](../sdk-reference/README.md). The direct Etalon service uses
the source-backed module-version-1 remote-handle skeleton export, with methods
zero/one for native open/close and method two for IQ-power. IQ invocation uses
attributes zero, one input and one output, no handles, scalar
`0x02010100`; matching DSP skeleton source is supplied separately. Generating,
linking, signing and qualifying the final module requires the selected local
SDK and board integration.

Dispatch validates the exact operation/revision, current provider generation,
rank, encoding, contiguity and lengths. The response must match the request's
provider/generation/nonwrapping request ID and complete wire schema before any
payload is published. These echoed identifiers are consistency checks, not
cryptographic execution attestation. Inputs and outputs remain borrowed through
the entire synchronous call. The provider uses caller workspace for encoded
request and response, scrubs that used region on return/unwind, and clears output
on failure. Authority is rechecked before and after native execution and before
output publication. The authority must hold a lease during the call; these
checks cannot retroactively stop a revoked call already executing on hardware.

The session is thread-affine and serializes invocation against its own cleanup
through exclusive ownership. A shared process-wide mutex serializes native
load/open/close/unload across this crate's independent instances; poisoning
prevents new opens while cleanup still attempts once. Native invocation across
different handles is permitted by the pinned header. Unrelated native bindings
need host coordination around all lifecycle calls/drops or process isolation.
Stop/drop attempts one close. A nonzero close status means
unknown remote outcome: the handle is never retried or reused and the runtime
mapping is deliberately retained for process lifetime. Failure after library
loading (missing symbols, revoked authority, lock poisoning or unsuccessful
open) also retains its runtime mapping, since global initialization may persist.
Successful handle cleanup still depends on the trusted runtime's contract.
Call `stop` explicitly when a cleanup outcome must be recorded: `Drop` can
attempt cleanup but cannot report its result to the caller. Revocation after
open preserves a failed cleanup status in the returned admission error.

The declaration is `AllocationClass::ColdUnbounded`: FastRPC and native runtime
allocations are not bounded by caller workspace. No latency bound is claimed.
The synchronous C call can block indefinitely; no timeout releases live buffers.
Hard memory, deadline and no-egress constraints require host process isolation.
There is no CPU fallback, runtime discovery, unsigned-module enablement, assumed
available hardware, or claim that an echoed target proves physical DSP execution.

Native target qualification remains deferred. CI host tests cover scalar/layout invariants,
one-close ownership, independent-session close serialization, revoked admission,
generation mismatch, failed invocation/argument mutation, allocation
classification and scratch cleanup. Later checks must include C ABI
layout comparison with the selected SDK, signed service-image rejection,
stub/skeleton round trips, runtime buffer registration/cache requirements,
real device loss and cleanup, and IQ-power comparison with the CPU oracle.
