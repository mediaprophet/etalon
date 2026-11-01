# Local vendor deployment admission

The Rust deployment contracts and mock native boundaries are validated by the
Etalon CI vendor jobs. Native runtime loading and physical device execution need
separate deployment qualification. See [release validation](../docs/sensing-release.md).

The `etalon-sense-deployment` library accepts a typed `DeploymentManifest`
containing an opaque nonzero deployment ID and target ID, a protected local
root, an exact ORT library path/length/SHA-256 and, where applicable, an exact
provider backend path/length/SHA-256. QNN HTP requires its own backend entry.
An OpenVINO provider linked into ORT can omit a separate provider entry.
The hashes must be supplied by an authenticated manifest authority; computing
them from arbitrary untrusted files and approving them proves no provenance.

The host's `DeploymentAuthority` approves the entire declaration before any
artifact I/O, and is rechecked before admission is returned. The host must
validate the target/provider compatibility, provenance, redistribution terms,
root ownership, local volume, access controls and no-egress isolation. It must
hold an immutable-artifact and policy lease through validation, initialization
and all runtime use. The library does not install an SDK, driver or runtime.

Validation rejects obvious remote, device, alternate-stream and parent paths
before probing. Final-component no-follow/reparse checks and regular-file
checks protect supported Windows/Unix paths. Mapped volumes, mounts, ancestor
redirections, disguised device nodes and concurrent directory replacement
still require a protected environment. Native dependent-library loading and
provider network behavior are outside this boundary.

`validate_deployment` checks the aggregate declared artifact byte budget before
opening files. It streams through caller-owned scratch capped at 64 KiB, checks
exact lengths and digests, and wipes supplied scratch on return or unwinding.
One extra byte per artifact detects file growth. Its byte budget does not bound
all OS allocations, time spent in filesystem calls, or provider memory use.
It validates only the two explicitly listed files; the host must independently
admit any transitive native library dependencies and trusted self-contained
model bytes.

The returned `AdmittedDeployment` denotes matched declared artifacts at that
check. `NpuRuntime::initialize_from_deployment` rechecks authority and requests
the admitted ORT path. The OpenVINO/QNN `prepare_from_deployment` entry points
check provider kind and requested-path consistency and recheck host authority.
The retained requested path is an input declaration. ORT's process-global
loader may silently reuse its first loaded library; initialize before any
other ORT use. Neither the token nor path equality proves which library was
loaded, what processor executes a graph or which driver is in use. CPU graph
fallback remains disabled in the explicit providers.

The separate adapter workspace pins cached ORT rc.15 with dynamic loading and
does not request binary-download features. A consuming workspace can still
unify features that enable downloads. This workspace retains its own
`Cargo.lock`, and CI checks its Rust dependency graph. Hosts must still establish
offline/no-download dependency and artifact controls for their chosen deployment.

## FastRPC DSP deployment boundary

`etalon_sense_deployment::fastrpc::validate_fast_rpc_deployment` admits an
exact local FastRPC runtime, a DSP service module and up to 64 additional
firmware/dependency artifacts. Every entry requires a nonzero declared length
and SHA-256 under the protected local root. Duplicate paths, missing identities,
invalid paths, overlong/nonprintable module URIs and aggregate byte-budget
overflow are rejected before file reads. Verification reuses the bounded
no-follow, regular-file and exact-stream checks, and clears caller scratch on
every return or unwind. This is a separate manifest from ORT deployment; no
provider-kind reinterpretation is used.

`FastRpcAuthority` must authenticate **every** declaration against trusted
signatures, authorize the exact target/domain/session and service schema, and
hold the immutable no-egress artifact/policy lease. Hashes supplied by an
untrusted manifest are not an acceptable signature or provenance check. The
authority must reject a missing firmware/dependency closure and prevent the
runtime from resolving undeclared libraries, firmware or modules, including
via search paths, environment variables, ancestor replacement or mapped roots.
The listed files must cover everything actually loaded; an empty additional
artifact list is valid only if the authority verifies no additional files are
needed or they are already covered by the protected deployment authority.

The manifest binds the image's provider ID (the target ID), exact service
capacity (1 through 1,048,576 IQ pairs), and nonzero admitted session generation.
The generation must match the canonical `etalon_generation` URI parameter;
keys starting with `_` are reserved by the vendor ABI. The host derives/checks
its descriptor from these declarations before loading. The authority verifies
the signed image's target/capacity, holds one active session for the generation,
and never reuses that generation after close, reset or module reload. The
service retains a generation high-water mark across close while loaded;
freshness across image unload or processor restart needs durable host authority.
An image can accept later fresh generations without recompilation.

The exact NUL-terminated module URI is authenticated as supplied; admission
does not invent a QAIC URI or infer domain/session from its spelling. The
authority must bind URI resolution to the admitted module digest, the service
entrypoint/schema, and the selected DSP domain/session. Session IDs and domain
IDs are policy assertions; this function neither reserves a session nor obtains
the runtime's effective domain. Runtime-specific DSP signing and firmware trust
chains remain prerequisites checked by the host authority. File hashing does
not attest the remote processor or establish which firmware/module it runs.

The returned token exposes only the authenticated declaration and matched-byte
count. `revalidate_authority` checks lease revocation; it does not rehash files.
The host transport must recheck it at load/open/invoke boundaries and compare
the exact service ID and schema digest before making RPCs. Loading and calling
the native runtime still requires an explicit unsafe trust boundary even when
artifact admission succeeds. Admission does not install firmware, create a DSP
protection domain, load libraries or execute module code.

Mock tests cover refusal before artifact I/O, scratch wiping, identity and
repeated-path rejection, and lease revocation. CI runs these separately from
physical FastRPC execution.

## RTL-SDR deployment boundary

`rtlsdr::validate_rtl_sdr_deployment` checks an explicit local runtime and up to
64 declared dependencies using bounded caller scratch and an aggregate file-byte
budget. Paths, nonzero lengths/digests, target/generation, exact enumeration index
and sample capacity are authenticated through `RtlSdrAuthority`. Invalid or
duplicate paths, overflowing budgets and capacity outside 1..=1,048,576 IQ pairs
are refused. File verification performs no library loading or radio access.

The token is consumed by the separate concrete RTL-SDR library, which rechecks
the authority lease around loading and device operations. The authority must
protect ancestors and native dependency/search resolution, verify ABI/driver
compatibility and enforce no-egress isolation. Hashing a declared library does
not establish its loaded identity or trustworthiness. Native execution remains
an explicit unsafe boundary. An empty dependency list requires an authority
that verifies the closure is complete or covered by its protected environment.

An enumeration index and USB identity strings are unverified selectors, not
physical identity proofs. The device-operation gate must approve the actual
selected handle identity and exact configuration before tuning/acquisition.
Human-centered permissions, environmental-use policies and evidence custody
remain composable host decisions. Generations must not be reused across device
reopen, process restart or rollback.

The synchronous native read has no hard-deadline or cancellation interface.
The backend must not claim the existing bounded `SdrReader` contract; callers
needing enforceable deadlines must establish process/device isolation or use a
separately implemented cancellable backend. Caller I/Q buffers are bounded,
while libusb/driver/runtime allocations remain unbounded. Rust/mock checks run
in CI; native runtime loading, device execution and deployment packaging have
not been qualified.

## GPU deployment boundary

The ordinary sensing GPU provider remains explicitly `ColdUnbounded` for live
dispatch. Its buffer mapping and command recording/submission use ordinary
Etalon APIs with internal allocation. The prepared sensing provider instead
uses a private Vulkan lane with fixed resources and caller readback storage;
its live allocation class also remains `ColdUnbounded` pending qualification.
See [prepared compute](../docs/prepared-compute.md) for the implemented ownership,
policy, timeout and allocation boundaries. File admission establishes neither
a hot zero-heap execution path nor hardware identity.
