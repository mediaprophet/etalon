# Modernization qualification evidence — 2026-10-30

This is an interim evidence ledger, not release acceptance. The reviewed changes
remain development work. Final commit pins, successful GitHub Actions, the full
Qualia core library suite and a successful full Etalon suite are not claimed.
See [the implementation plan](plans/ecosystem-modernization-2026-10-30.md),
[upstream dispositions](upstream-review-2026-10-30.md) and
[conformance limitations](conformance-coverage.md).

## Environment and scope

- Local compiler: Rust 1.96.1 (`31fca3adb`, LLVM 22.1.2),
  `x86_64-pc-windows-msvc`. This is not evidence for the repository MSRV matrix.
- Physical adapter: Intel UHD Graphics 620, integrated GPU, PCI `8086:5917`.
  DX12 reports driver `31.0.101.2141`; GL reports
  `4.6.0 - Build 31.0.101.2141`. Vulkan reports `Intel Corporation` / `Intel driver`.
  Microsoft Basic Render Driver results are software execution, not Intel hardware
  evidence. No NVIDIA card is required for the retained portable paths.
- The resolved Etalon graph contains 450 packages and no Deno/V8 packages.
  Native compiler dependency is `mach-dxcompiler-rs 0.1.7`; the separate official
  DXC reference is `1.9.2602.24`. They are separate compiler oracles, not interchangeable
  proof. See [native dependency notes](native-accelerator-dependencies.md).
- Browser receipts identify both repositories as dirty development sources and
  record source fingerprints plus exact JS/WASM SHA256 values. A commit hash alone
  does not identify those development artifacts.

Selected advertised capabilities from the fresh local `.gpuconfig` are below.
They are discovery results, not execution certification. All three Intel backends
advertise 1,024 compute invocations per workgroup and workgroup dimensions
`[1024, 1024, 64]`.

| Backend | Shader f16 | Subgroup feature | Subgroup size range                 | Nonuniform storage-texture arrays | Maximum buffer bytes |
| ------- | ---------- | ---------------- | ----------------------------------- | --------------------------------- | -------------------- |
| Vulkan  | Yes        | Yes              | 8–32                                | Yes                               | 4,294,967,295        |
| DX12    | Yes        | Yes              | 16–16                               | No, affected driver is guarded    | 2,147,483,647        |
| GL      | No         | No               | 4–128 reported, feature unavailable | No                                | 2,147,483,647        |

## Evidence collected

`cargo clippy --all-targets --all-features -- -D warnings` passes on the local
MSVC target in 8 minutes 43 seconds. This does not substitute for the other
platforms in CI. Rustfmt, Typos, Tombi (234 manifests) and generated-file sync
checks also pass.

| Check                        | Observed result                                                                                                           | Limit                                                                                                                              |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Naga library suite           | 151 passed, zero failures                                                                                                 | CPU shader-library tests; no Metal-device execution claim                                                                          |
| Focused native qualification | 94 nextest cases passed in 86.853 seconds                                                                                 | Includes unsupported-feature outcomes, expected failures and software adapters; not 94 hardware executions                         |
| GL mapping correction        | Three synthetic flag checks pass, with GL mapping regressions passing in native qualification                             | Synthetic access/storage-flag separation does not by itself prove driver execution                                                 |
| Trace player corpus          | DX12, GL and Vulkan execute successfully; Metal explicitly reports unavailable                                            | One adapter/instance reused per backend, fresh device per fixture; unavailable Metal is not Metal coverage                         |
| Full Etalon suite            | Completed 2,475 cases: 2,464 passed, 11 failed, 24 skipped; `compile_fail` passed in 151.114 seconds                      | Historical full-run failure count; see the fresh repair batch below for subsequent results                                         |
| Browser integration          | Two canonical Vibe CPU cases plus one Rust Etalon browser GPU case passed; GPU case performs 11 dispatch/readbacks        | Counts 1/63/64/65/257, repeated inputs, sparse-zero check; browser backend delegates to browser WebGPU, not native Etalon HAL/Naga |
| Browser report and email UI  | Actual Run/Download remains passing; mocked file-share, cancellation, draft fallback and invalid-recipient rejection pass | Share file bytes match JSON. No email sent; OS mail delivery is untested                                                           |
| Plain LAN HTTP diagnostic    | Two CPU cases pass; GPU unsupported, zero GPU/NPU passes; download matches UI; unrelated document returns 403             | Assets explicitly unverified over insecure HTTP. LAN server subsequently stopped; HTTPS Pages/email is the requested delivery path |
| NPU evidence parser          | Six pure GNU-target tests pass, including fallback/incomplete evidence rejection                                          | No actual NPU execution; MSVC execution/provider matrix remains pending CI                                                         |
| Local NPU discovery          | DXCore probe complete, no devices detected; execution `not_run`                                                           | Browser NPU visibility remains unknown; absence is never an execution pass                                                         |

## Failures and remediation

The subsequent fresh repair batch runs 438 cases: 436 pass and two Intel DX12
multiview cases fail. Vulkan's original and added MSAA regressions, unsafe opt-in
validation, Naga's six previously tool-blocked cases and diagnostic-caret test now
pass. The separate `etalon-types` run passes all 54 tests. These targeted results
do not replace a successful full-suite re-run.

The subsequent 30-case multiview/PSO-stream run passes 28 cases and reproduces
the same two Intel DX12 failures. Five PSO-stream serialization tests pass,
covering pointer alignment and deterministic padding; fixing those safety
defects did not fix the multiview output. A small binary using the original
multiview test module reproduces both failures for further controlled diagnosis.

| Observed failure                                                                      | Change and verification                                                                                                                                                        |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| GL buffer mapping received a storage-only flag                                        | Keep `DYNAMIC_STORAGE_BIT` in buffer-storage flags and out of map access flags; synthetic checks and native mapping cases cover the distinction                                |
| Mapping failure could lose the device while resource locks were held                  | Defer loss until locks are released; focused mapping-error regression passes                                                                                                   |
| Player's aggregate corpus exceeded the 90-second nextest limit                        | Split by backend and eliminate repeated backend initialization. Available backend cases pass within 7.4 seconds; timeout was not increased                                     |
| Cold compile-fail test competed with GPU work                                         | Reserve the configured test threads for compile-fail. The current full run passes it in 151.114 seconds without increasing its timeout                                         |
| Subgroup own-lane shuffle and Intel DX12 storage-texture array failures               | Portable subgroup regression passes; the storage-array guard targets the reproduced adapter/driver. Focused qualification is green; unsupported results remain distinguishable |
| Browser build failed on `InstanceDescriptor::default()` and exposed platform warnings | Use the explicit constructor, gate unused native implementations precisely, retain active validation and test-only helpers; actual WASM codegen and browser execution succeed  |
| Email UI module was rejected by the restricted server                                 | Add exactly `email.js` to the allowlist and test the eight permitted paths. Actual email UI smoke then passes; first interrupted attempt is retained                           |
| Full-run multiview, experimental-mask, SPIR-V tooling and diagnostic failures         | Fresh repair batch clears Vulkan multiview, experimental opt-in/mask, all six tooling failures and the diagnostic mismatch; two Intel DX12 multiview cases remain failing      |

A direct Vulkan probe isolated the MSAA failure: both multisampled layers contain
their expected colors, but implicit multiview resolve produces zero for layer one.
Explicit per-layer image resolve and an ordinary single-layer LOAD/resolve pass
both recover bytes 32/96. The latter preserves attachment-only image usages and
forms the scoped fallback for Intel `8086:5917`, driver version `1656925`
(`31.0.101.2141`), `INTEL_PROPRIETARY_WINDOWS`. Fresh HAL execution now passes the
original MSAA oracle and the added nonzero-base-layer/sparse-mask/StoreDiscard
regression, alongside normal, noncontiguous and single-view cases and the exact-driver
guard test. The separate viewport/scissor ordering fix follows the
[Vulkan multiview state rule](https://docs.vulkan.org/refpages/latest/refpages/source/VkRenderPassMultiviewCreateInfo.html):
initial dynamic state must be set after beginning the multiview subpass. That
validity correction is not claimed to explain the resolve failure.

The unused batched browser LLM argmax remains an explicitly staged
`wasm-llm-diagnostics` experiment, not production decode or harness execution.
Vulkan external-texture descriptor infrastructure remains gated until SPIR-V
external-image lowering exists. Neither item is claimed as usable completed coverage.

## Retained local receipts and remaining gates

Receipts below are ignored workspace artifacts, deliberately not committed links.
They must be retained or exported with their source/build identities for review;
a fresh clone will not contain them.

- Etalon `target/naga-library-tests-retry.log`,
  `target/modernization-focused-final.log`, `target/modernization-full-final.log`;
  the earlier failing run is `target/modernization-full-rust-tests.log`.
- Fresh repair and types results: `target/modernization-repair-batch.log` and
  `target/modernization-types-tests.log`.
- Strict Clippy: `target/modernization-clippy-strict.log`; subsequent PSO and
  multiview checks: `target/modernization-dx12-normalized-stream.log`.
- Qualia `target/qualification/browser-harness-20261030-first/`,
  `browser-harness-lan-first/`, `browser-harness-email-first/` and
  `browser-harness-email-retry/`; each completed smoke retains bounded JSON receipts.
- Qualia `target/qualification/browser-wasm-{check,check-retry,build}.log`,
  `npu-parser-tests.log`, and the harness run referenced by
  `target/qualification/language-discovery.json`.

Storage cleanup removed approximately 3.01 GiB of Etalon incremental state and
620,663,300 bytes (0.578 GiB) of identified GNU dependency artifacts, 511 files.
MSVC/WASM caches, downloaded SDKs and receipts were preserved. These are gross
reclaimed amounts; subsequent builds consume space again.

Remaining acceptance gates: resolve and re-run every full-suite failure, complete
Qualia core/default-feature and platform checks, exercise any available supported
NPU provider without silent fallback, commit and pin the final sources, rebuild
published browser artifacts against those pins, and obtain successful GitHub Actions
including HTTPS Pages deployment. Application integration does not restore the
retired WebGPU CTS coverage.
