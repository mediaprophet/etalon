# Change Log

<!--
Please add your PR to the changelog! Choose from a top level and bottom
level category, then write your changes like follows:

- Describe your change in a user friendly format. By @yourslug in [#99999](https://github.com/mediaprophet/etalon/pull/99999)

You can add additional user facing information if it's a major breaking change. You can use the following to help:

```diff
- Old code
+ New code
```

Top level categories:

- Major changes
- Added/New Features
- Changes
- Bug Fixes (that don't change API)
- Performance
- Documentation
- Dependency Updates
- Examples
- Testing/Internal

Bottom level categories:

- General
- naga
- Validation
- DX12
- Vulkan
- Metal
- GLES / OpenGL
- WebGPU
- Emscripten
- Hal
-->

This changelog covers Etalon only. Etalon was forked from [wgpu](https://github.com/gfx-rs/wgpu) 30. For the history of changes made before the fork, see the [upstream wgpu changelog](https://github.com/gfx-rs/wgpu/blob/trunk/CHANGELOG.md).

## 0.0.3

### Mobile browser portability and resource ownership

- Preserve optional browser limits and adapter diagnostics, move float-canvas probing to explicit asynchronous capability discovery, and add recorded Core/Compatibility/Auto qualification profiles without silently weakening caller requirements.
- Add portable GPU radix/sieve and token readback, bounded tensor-arena allocation, corrected 96-byte fused-attention uniforms, and top-k edge-case handling. Android NPU discovery remains distinct from runtime or execution verification.
- **Breaking:** construct submission leases through `HalComputeLane::lease_submission`; unrestricted construction/public resource fields are removed and reclamation is gated by issuing-queue completion. See the [mobile resolution disposition](docs/mobile-resolution-2026-11-01.md) for migration and native async polling requirements.
- Expand browser qualification to 20 cases, including 18 GPU cases, with optional ADB device selection. Qualification requires every declared case to execute successfully; physical-device results remain separate from hosted checks. No mobile performance or CTS result is claimed.

### Browser error reporting

- Restore upstream's default Core adapter-request shape while retaining explicit Compatibility requests; show failed GPU qualification prominently when the browser suite cannot execute GPU work.

- Preserve browser adapter-request rejection details instead of misreporting them as an unavailable adapter.

## 0.0.2

### Release preparation

- Prepare the 0.0.2 Git source release; [release notes](docs/releases/0.0.2.md) describe capabilities, usage, qualification boundaries and unchanged licensing.

### Added/New Features

- Added a revision-matched Rust/WASM browser test page with 13 portable/GPU cases, render readback and local reports, alongside a broader Etalon capability guide. Core/compatibility adapter selection and workload-sized required limits exercise actual device capabilities.

- Add private prepared Vulkan execution and a sensing IQ-power live provider
  that reuses upload, command, submission and readback resources. Caller sample
  buffers are bounded; target allocation and driver qualification remain explicit.

- Add an opt-in RTL-SDR receive adapter with explicit local runtime/dependency
  admission, device policy gates and caller-owned I/Q buffers. Native read
  deadlines and hardware qualification remain external requirements.

- Add an isolated sensing vendor workspace with portable DSP routing, a native
  Hexagon IQ-power service, synchronous FastRPC host transport and bounded local
  artifact admission. Hardware execution and deployment qualification are deferred.

## v0.0.1 (in development)

First Etalon version, forked from wgpu 30. The `v0.0.1` tag was created on 2026-10-21 before the version was complete; work on v0.0.1 continues on the `v0.0.1.x` branch.

### Major changes

- `etalon-tensor` residency-key sets now use `hashbrown::HashSet`: callers assigning `CudaMegaPassExecutor::resident_weights` or calling `collect_and_validate_layer_keys` must use that collection type. `register_resident_weight` is unchanged.
- `Queue::submit` now returns `Option<SubmissionIndex>`. A failed native submission has no waitable index; the browser backend always returns `None`, and synchronous browser waits return `PollError::CantWait`.
- Renamed the `wgpu` crates, modules and documentation to `etalon` (`etalon`, `etalon-core`, `etalon-hal`, `etalon-types`, `etalon-core-remote`, `etalon-core-remote-types`, `etalon-sync`, `etalon-naga-bridge`, `etalon-info`). `naga` and `naga-types` keep their names. (5e1908e38 and earlier)
- Set every workspace package to version `0.0.1` with `publish = false`. Etalon is consumed through git, not crates.io. (f20f5abaf, ed0b671d8)
- Licensing: Etalon modifications and additions are CC BY-NC-ND 4.0 (`LICENSE`, `LICENSE-QUALIA`). Upstream wgpu license texts are kept in `upstream-licenses/`. (bcfb954fb, 29df9cf5e)
- Environment variables use the upper-case `ETALON_` prefix (for example `ETALON_BACKEND`, `ETALON_ADAPTER_NAME`, `ETALON_POWER_PREF`, `ETALON_VALIDATION`). The initial rename had produced lower-case `etalon_*` names; those are no longer read, and neither are upstream's `WGPU_*` names.
- WGSL extensions specific to this implementation use the `etalon_` prefix (for example `enable etalon_ray_query;`, `enable etalon_cooperative_matrix;`). Upstream's `wgpu_*` spellings are not accepted.
- Branch model: `dev` (development), `latest` (experimental users, default branch), `v0.0.1.x` (the v0.0.1 version line and its fixes), and `stable` (general use, to be created when v0.0.1 is complete).

### Added/New Features

- Browser `Instance::request_webgpu_adapter` supports explicit compatibility-level adapter requests while retaining browser validation.
- Naga's HLSL `Options::force_vertex_view_index` forwards fragment view indices through a flat vertex output for drivers that mishandle pixel `SV_ViewID`.
- Extension crates under `crates/`: `etalon-agent`, `etalon-nquin`, `etalon-render`, `etalon-tensor`, `etalon-wasm` and `etalon-wgsl-forge`. (e1193443a)
- Implementation work across the foundation/types/sync, naga IR/frontends/validation, naga backends, `etalon-core`, `etalon-hal`, and surface/deno/examples layers. (9ebe40ef3, 2aa8295a5, baf80e4c5, 38c7237d2, e6ac82c3d, 38723252e)

### Bug Fixes

- Forge manifests only label matching, source-bound Naga reports as validated. Numerical comparisons reject invalid tolerances and nonfinite reference values.

#### General

- Sparse multiview clears, resolves and discards track only selected layers, preserving inactive user data and initializing unwritten layers before reads.
- Buffer-mapping failures invalidate the device immediately and deliver device-loss callbacks after resource locks are released, allowing callbacks to reenter the API safely.
- Experimental buffer device addresses and host-memory imports now require explicit unsafe experimental-feature opt-in.
- Identifiers broken by the rename are back to Rust naming conventions: `Etalon*` types and enum variants (for example `ImplementedEnableExtension::EtalonRayQuery`, `EtalonContext`) and upper-case constants and statics (`INSIDE_ETALON_PASSES`, `ETALON_CTS_XTASK`).
- `etalon-wgsl-forge`: generated shaders now use `enable etalon_cooperative_matrix;` and `enable etalon_ray_query;`, which Etalon's WGSL front end accepts. They previously used upstream's `wgpu_*` spellings, which it rejects.
- `etalon-core`: a range-tracking test called a non-existent `RangedStates::append` and stopped the crate's tests from compiling. It now uses `isolate` and `coalesce`.
- `etalon-render`: EMF pipeline entry points now match the shader's `vertex_main` and `fragment_main`. (4e77ca3f7)

#### WebGPU

- `Device::pop_error_scope` no longer panics on `wasm32` when the error scope caught no error and the browser resolves it to `null` or `undefined`. Unexpected JavaScript errors now surface as `Error::Internal` instead of a panic. (bcfb954fb)
- `request_device` omits `requiredLimits` keys that the adapter does not expose. This avoids `OperationError` when browsers remove limits such as `maxInterStageShaderComponents`. (bcfb954fb)

#### Vulkan

- Lavapipe multiview passes avoid driver defects in discarded color attachments and sparse resolves without disabling memory poisoning.
- Naga folds subgroup gathers from the current invocation into their evaluated input, preserving argument side effects and avoiding unnecessary self-shuffle instructions.
- `BFLOAT16_KHR` is matched by its raw discriminant (`1000141000`). (dfbaad4c3)

#### DX12

- Pipeline state streams now enforce native descriptor alignment and copy C-structure padding without reading uninitialized bytes.
- Intel device `0x5917` with driver `31.0.101.2141` no longer advertises non-uniform storage texture array indexing after reproduced incorrect reads. Uniform arrays remain supported; see `docs/dx12-intel-storage-array-limitation.md`.

#### Hal

- The `static-dxc` feature links the pinned `mach-dxcompiler-rs` 0.1.7 compiler and derives supported shader models from the compiler version. Native dependency provenance is documented separately.
- GLES buffer storage and mapping use separate access masks, fixing invalid readback mappings for buffers that also receive copies or query results.

### Documentation

- README: upstream attribution to [gfx-rs/wgpu](https://github.com/gfx-rs/wgpu), a git-dependency quickstart matching the current API, the homepage, the branch model, and an environment-variable section.
- `upstream-licenses/`: restored the upstream MIT copyright notice (`The gfx-rs developers`) and clarified which material each license covers.
- Workspace `homepage` is <https://mediaprophet.github.io/etalon/>.
- This changelog records Etalon changes only. Upstream history is linked above rather than copied.
- Removed dead links and links to crates.io/docs.rs for Etalon crates; Etalon is not published to crates.io.

### Dependency Updates

- Workspace dependencies harmonized with consuming projects. (0c9e6c466)

### Testing/Internal

- Reviewed all 62 upstream commits after the 18 October fork point and ported applicable resource-validation, pipeline-override, MSL, Metal, browser-cache, staging and texture-blit fixes with regression coverage. See `docs/upstream-review-2026-10-30.md`.
- Added bounded NPU discovery and separate discovery/runtime/workload/execution states in `etalon-agent`; submitted commands and scalar CUDA fallback work no longer claim verified hardware execution.
- Retired the embedded Deno WebGPU CTS tooling and its dependencies. QualiaDB now provides a VibeScript integration harness; the resulting conformance coverage loss is documented in `docs/conformance-coverage.md`. Existing Rust tests are retained.
- Removed wgpu's community, governance and contribution setup (contributing, governance, code of conduct and security policies; issue and PR templates; maintainer approval and changelog PR gates; merge-queue, review and release checklists).
- Removed legacy CI services and automation (Renovate, Codecov, daily scheduled runs, the `Lazy` workflow). Docs and examples deploy to this repository's `gh-pages` branch.
