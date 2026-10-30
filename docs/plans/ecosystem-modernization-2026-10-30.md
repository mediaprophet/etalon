# Etalon, QualiaDB and VibeScript modernization

Status: implementation in progress; validation results are not yet acceptance.

## Objective and boundaries

Review the implementation and dependencies before further broad testing, apply
justified updates and upstream correctness fixes, complete the consuming
QualiaDB/VibeScript harness, preserve existing Rust suites, document lost CTS
coverage, then commit, push and resolve the resulting GitHub Actions failures.
CPU, integrated/discrete GPU and NPU evidence must remain distinguishable.

This plan covers build/runtime dependencies, shader translation and validation,
resource correctness, accelerator discovery/routing, test infrastructure and CI.
It does not assert that a review of these boundaries certifies every algorithm
in either repository. Unavailable hardware cannot establish execution coverage.

## Review baseline

- Etalon starts at `a5bf2501e` on `dev`; QualiaDB is on `0.0.40.8-dev`.
- The common ancestor with upstream wgpu is
  `e084b66aab767940ff1236604d194c9e1c2844b0` (18 October 2026).
- The reviewed upstream endpoint is
  `49325ca0a13bdc0787b1601138096cce743dd29e` (29 October 2026),
  comprising 62 commits after that ancestor. Commit dates alone are insufficient
  because some later-merged changes have earlier author dates.
- The existing Deno removal and initial consuming harness are retained work.
  The initial harness compiled, but its CPU/iGPU execution had not been qualified
  when the user requested review before further testing.
- The laptop exposes Intel UHD Graphics 620. NPU presence must be probed rather
  than inferred from its GPU or CPU model name.

## Supervision and ownership

The supervising agent owns integration, public contracts, lockfiles, upstream
core/API ports, harness receipts, CI wiring and final acceptance. Workers may
edit only their assigned files, preserve other agents' changes and report
incompatible upstream changes. They do not independently commit or push.

| Workstream                    | Assigned role     | Bounded ownership                                                                                                                                    |
| ----------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dependency/build safety       | Dependency agent  | Workspace dependency declarations, optional ORT consumers, real static-DXC feature, portable GPU/Forge feature wiring, unused tensor CUDA dependency |
| Accelerator discovery         | Accelerator agent | `etalon-agent::accelerator`, agent manifest/reexports and HAL capability state transitions                                                           |
| Upstream shader/Metal         | Shader agent      | Reviewed MSL/compact prerequisites and fixtures, Metal capability/drawable/mapping fixes                                                             |
| Integration and upstream core | Supervisor        | Core validation/resource/API fixes, Qualia routing/cache projection, harness, reporting, CI, docs, lockfiles and release checks                      |

Independent edits may proceed in parallel. Dependency resolution, formatting,
builds and hardware execution are supervised after the combined source batch
is reviewed. Conflicting file ownership must be resolved before editing.

## Wave 1: storage and reproducible build boundaries

Inventory before deletion; remove only resolved build-cache paths inside the
named repositories. The stopped Etalon `target/debug/incremental` cache was
3.01 GiB and has been removed. Preserve compiled dependency artifacts, the
fresh Qualia harness target, downloaded SDKs and local nextest tooling.

Use bounded build concurrency, no incremental cache and reduced debug output
for the new harness. Keep production lockfiles separate from local-checkout
test overrides. Do not delete source, user assets, live application state or
unidentified temporary directories. Recheck available space before expensive
builds and retain small qualification receipts rather than duplicate targets.

## Wave 2: dependency and feature corrections

1. Upgrade optional ORT consumers to the reviewed rc.15 API: an explicit
   environment, shared version selection and preserved CPU/model-validation
   behavior. Review both core super-resolution and `qualia-vision` consumers.
   Upstream reports safety fixes in rc.15; this is not a cosmetic version bump.
2. Apply Tokio 1.54.1's timer regression fix. Retain current supported Windows,
   Ash, GPU allocator, Bytemuck and Serde versions unless review finds a concrete
   reason to change them.
3. Make portable defaults independent of CUDA. Keep CUDA explicitly selectable;
   remove unused CUDA wrapper dependencies where the existing shared driver
   probe supplies the required behavior. Driver initialization is not graph
   instantiation or successful model execution.
4. Repair native GPU/Forge/Naga feature composition, preserving validation and
   existing CPU fallbacks. A disabled feature must not report successful work.
5. Restore a functioning optional static-DXC dependency and supported target
   boundary. Review toolset requirements and binary license notices before
   selecting a newer release; do not silently disable a public feature.
6. Refresh justified build/transitive patch versions and retained browser-runner
   packages selectively. Do not copy upstream's lockfile or restore Deno.

Sources: [ORT releases](https://github.com/pykeio/ort/releases),
[Tokio 1.54.1](https://github.com/tokio-rs/tokio/releases/tag/tokio-1.54.1),
[static DXC build source](https://github.com/DouglasDwyer/mach-dxcompiler-rs/blob/master/build.rs).

## Wave 3: upstream issue audit and equivalent fixes

Maintain a commit-level review ledger for the 62 upstream commits. Assess the
underlying defect against the renamed/diverged stack; do not equate copying a
patch with proving a fix. Preserve Etalon extensions and safe-boundary checks.

Prioritize all applicable resource and shader correctness fixes: external-plane
same-device validation, all-aspect texture clears, multiplanar zero initialization,
device-loss/mapping ordering, surface callbacks, pipeline overrides and resource
validation, successful-submission index semantics, texture blitter sampling,
MSL gradient dimensionality and continuing-block temporary lifetimes, Metal
sampler limits and initially occluded surfaces. Include associated regression
tests/fixtures. Evaluate performance changes and Vulkan external textures with
their prerequisites and explicit compatibility decisions.

Deno-only changes and retired CTS digest updates do not apply. Community/merge
automation that this fork intentionally removed is not reinstated. Existing
browser Rust tests and useful CI dependency updates remain in scope.

Use the [WebGPU specification](https://gpuweb.github.io/gpuweb/) and
[WGSL specification](https://gpuweb.github.io/gpuweb/wgsl/) to establish expected
behavior, with named section anchors in the issue ledger.

## Wave 4: truthful accelerator inventory and routing

Expose separate hardware-discovery, runtime-initialization, workload-support and
execution-verification outcomes. Successful OS enumeration with zero NPUs is
different from an unavailable API, denied access, unknown driver or unprobed
platform. Bound enumeration and property reads; sort results deterministically.

Windows uses dynamically loaded system DXCore and its documented NPU hardware
attribute; Linux distinguishes recognized NPU drivers from other accel devices.
Unsupported platform probes report that limitation explicitly. Neither a GPU
enumeration nor a DirectML DLL proves NPU availability.

Submitting a GPU command does not promote a capability to verified. Require a
completed workload and CPU-oracle comparison. Unified physical memory does not
prove a host mmap was imported without a copy. Detected NPUs are not routable
until an appropriate provider, workload and execution path are established.
Keep runtime/provider fingerprints separate from stale GPU-only passport data.

Sources: [DXCore hardware types](https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore-adapter-attribute-guids),
[Windows ML providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers),
[Linux accelerator subsystem](https://docs.kernel.org/accel/introduction.html).

## Wave 5: consuming harness and conditional coverage

Keep Etalon independent of QualiaDB. The consuming harness executes checked
VibeScript through the canonical Qualia host, uses real GPU upload/dispatch/
readback and compares deterministic results to CPU references. Include boundary
sizes, changed-input reuse, initialization, lifecycle and rejection behavior.

Provide lightweight discovery separately from full product integration, explicit
CPU-only execution, per-backend hardware execution and a require-NPU mode.
Record structured, bounded receipts with the exact source/dependency revision,
adapter identity/class/backend, workload, discovery/runtime outcomes, executed
case counts, verification results and limitations. Missing NPU execution is
`not_run/unavailable`, never an early-returning hardware test counted as a pass.
An available NPU without a supported initialized provider is detected/unverified.
Provider-specific tests must disable unnoticed CPU/GPU fallback and retain
placement/equivalent execution evidence before reporting an NPU pass.

Support phone and other browser diagnostics through GitHub Pages and an explicit
local network static server. Build a separate source-linked Qualia diagnostics
WASM bundle for canonical VibeHost CPU checks and an asynchronous Etalon Rust browser bridge;
the browser supplies WebGPU shader translation and HAL execution. Keep this
coverage separate from the native Render.gpu capabilities and native backends.
Verify the downloaded WASM against a build manifest containing both reviewed
repository revisions and asset hashes. Run in a terminable worker, preserve CPU
results across GPU failures, and offer a bounded downloadable JSON report with
no automatic upload. Browser NPU visibility remains unknown rather than absent.
Plain LAN HTTP may lack the secure context needed for WebGPU; provide explicit
certificate/key HTTPS hosting and retain the resulting availability diagnostics.

The harness does not implement upstream CTS. Preserve the old revision and
enabled/failing/skipped selector inventories; document every removed CI lane and
the JavaScript/WebIDL/shader/API coverage that is no longer exercised.

## Wave 6: validation gate

Freeze and review the combined source batch before broad execution. Run:

1. Formatting, manifest/lockfile/dependency/license checks, generated-file sync
   checks and source-level checks that Deno/V8/cooked-waker remain absent.
2. Focused regression tests for each changed safety/validation path and shader
   fixtures; optional ORT on/off checks and explicit CUDA compile checks.
3. Lightweight CPU/discovery checks, full checked-Vibe CPU integration, then
   Intel iGPU execution in isolated bounded processes. Run other local backends
   only when available; identify the actual backend used.
4. Conditional NPU discovery and supported provider execution, preserving
   unavailability/unsupported/error results without fabricated passes.
5. Existing relevant Rust suites, full Etalon `cargo xtask test`, Qualia core
   library tests, retained browser/WASM checks and the supported CI matrix.
6. Browser harness report/failure behavior, WASM build and actual browser
   CPU/GPU execution; local server path/method restrictions and asset serving.
   Actual phone execution is a separate user-supplied diagnostic outcome.

Report every failure, including earlier failed build attempts. Determine whether
it comes from the changes, an existing defect or the environment using evidence;
do not suppress failed tests or relabel them as success. No trunk baseline run
is authorized by this plan. No CTS execution is claimed.

## Wave 7: documentation, commits, push and Actions

Reconcile the implementation ledger, commands, capability limits and conformance
loss documentation with the final code. Sync generated Etalon agent files and
update the Qualia session inventory. Do not edit generated copies directly.

Commit reviewed Etalon changes first, push the current authorized development
branch, then pin the consumer/harness CI to that reviewed Etalon revision and
commit/push QualiaDB. Preserve attribution and third-party licenses. Never
force-push over concurrent work. Track Actions for the exact pushed revisions,
repair failures and repeat only affected checks. A queued/skipped workflow or
unavailable self-hosted accelerator does not establish hardware coverage.

Completion requires successful applicable GitHub Actions, an honest hardware
coverage report, and documented platform/provider limitations. If credentials,
runner availability or an external service blocks a required check, report the
specific blocker and leave that acceptance item open.
