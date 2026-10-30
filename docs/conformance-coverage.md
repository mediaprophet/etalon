# QualiaDB/VibeScript integration and retired CTS coverage

The embedded Deno runner (`cts_runner`), local Deno WebGPU bindings
(`deno_etalon`), `cargo xtask cts`, and Deno-specific CI jobs have been removed.
Their Rust dependencies, including `deno_core` and MPL-2.0 `cooked-waker`, are
no longer part of this workspace. The narrow MPL exception has been removed.
The bincode advisory exception remains because Rust tests/tools still use it.

## Replacement integration harness

The harness belongs to QualiaDB, which consumes Etalon, avoiding a dependency
from Etalon back into QualiaDB. In a sibling QualiaDB checkout, run:

```powershell
pwsh -File scripts/test-etalon-vibe.ps1 -EtalonPath ../etalon -Backend dx12
pwsh -File scripts/test-etalon-vibe.ps1 -EtalonPath ../etalon -Backend vulkan
pwsh -File scripts/test-etalon-vibe.ps1 -EtalonPath ../etalon -CpuOnly
```

Run each GPU backend in a fresh process. Metal is available on supported Apple
hosts. An integrated GPU is a GPU and is eligible for the GPU mode; no NVIDIA
card or CUDA SDK is required. CPU-only mode explicitly filters out GPU tests
and cannot establish GPU coverage. Missing hardware fails the GPU test instead
of silently passing. The wrapper places a time limit on test execution.
Software CPU adapters cannot satisfy this hardware check, and an explicitly
requested backend must match the selected adapter's backend.

The runner builds a separate workspace under QualiaDB's ignored `target`
directory, patches the Etalon git dependencies to the selected local checkout,
and keeps its own lockfile. It does not rewrite QualiaDB's production dependency
pins. It prints the checkout paths and GPU adapter information so evidence can
be attributed to the implementation and backend actually tested.

The tests parse and check VibeScript and execute it through QualiaDB's canonical
`VibeHost`, rather than replacing that host with a mock:

| Area           | Replacement coverage                                                            |
| -------------- | ------------------------------------------------------------------------------- |
| Semantic graph | Detached graph transaction, commit, canonical Quin parity                       |
| GPU compute    | Real WGSL dispatch, buffer upload, staging copy and byte readback               |
| CPU comparison | Vector addition at 1, 63, 64, 65 and 257 elements, repeated with changed inputs |
| Initialization | Sparse writes leave untouched output elements zero                              |
| Lifecycle      | Empty readback, consumed readback, explicit GPU destruction                     |
| Rejection      | Invalid/destroyed handles and unsupported binding kinds                         |

These are focused integration tests, not exhaustive validation, rendering,
performance, zero-allocation, device-loss, security or conformance certification.
They cover the selected adapter/backend, not every adapter automatically.

## Coverage removed

The retired CTS revision is recorded in [retired-cts/revision.txt](retired-cts/revision.txt).
The exact former CI selectors and backend-specific expectations are retained in
[test.lst](retired-cts/test.lst), with former failing and skipped selectors in
[fail.lst](retired-cts/fail.lst) and [skip.lst](retired-cts/skip.lst). These files
are historical inventories, not active tests or evidence that tests passed.
The inventories contain 495 enabled selector entries, 138 failing entries and
36 skipped entries, excluding blank lines and comments. Selectors expand to
multiple cases; these counts are not individual test counts or pass rates.

Removing the runner loses direct execution of those upstream CTS selectors
against Etalon, including their JavaScript/WebIDL binding behavior. It removes
the Windows DX12, Linux Vulkan and macOS Metal CTS CI matrix. Areas formerly
selected include API operations and validation, resource creation and mapping,
command encoding and copies, render/compute pipelines, synchronization, shader
execution and language cases. See the retained lists for the exact scope;
neither the old selected subset nor the new harness establishes full conformance.

The Vibe harness does not test JavaScript promise/error behavior, WebIDL
conversions, browser DOM/canvas integration, browser worker exposure, or the
full upstream WGSL shader and API parameter matrices. The retired native Deno
runner itself did not establish real-browser DOM/worker coverage either.

Etalon's existing Rust unit, compile, dependency, validation, GPU, example,
benchmark-smoke, trace, Naga snapshot and browser-WASM tests are retained.
Their commands remain in [testing.md](testing.md). No Rust tests outside the
retired Deno runner/bindings have been deleted or disabled.

There is currently no replacement upstream CTS runner in this repository.
A future browser or native CTS bridge must prove it runs this Etalon build;
testing an unrelated browser's WebGPU implementation would not restore the
lost Etalon coverage.
