# Etalon — EMF Synthesis, Sensing and Compute for QualiaDB

> Rust foundations for electromagnetic-field synthesis, sensing and computation, designed for QualiaDB and Webizen systems.

[![CI](https://github.com/mediaprophet/etalon/actions/workflows/ci.yml/badge.svg)](https://github.com/mediaprophet/etalon/actions/workflows/ci.yml)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](LICENSE)
[![Ecosystem: QualiaDB](https://img.shields.io/badge/Ecosystem-QualiaDB-blue.svg)](https://github.com/mediaprophet/qualiaDB)
[![Platform: Vulkan_|_Metal_|_DX12_|_WebGPU](https://img.shields.io/badge/Backends-Vulkan_%7C_Metal_%7C_DX12_%7C_WebGPU-orange.svg)](#architecture--workspace-structure)

**Etalon** is being developed as an EMF synthesis, sensing and compute library specifically for [QualiaDB](https://github.com/mediaprophet/qualiaDB) systems. It brings together device and observation contracts, signal processing, generated compute kernels, tensor operations, semantic data buffers and electromagnetic-field visualization. Graphics is one supporting execution and presentation capability within that broader direction.

Etalon supplies Rust execution, accelerator and device-facing building blocks. QualiaDB supplies semantic context, inference, q42 uncertainty representation, human-centered permissions and evidence custody. Their integration is explicit: Etalon does not replace QualiaDB's database or policy systems. Use by any project remains subject to the applicable licence terms.

Version 0.0.2 provides foundations for this direction. Implemented synthesis-related building blocks include deterministic shader/kernel generation and rendering; these do not establish a general waveform generator, physical RF transmitter or complete EMF simulation/inference system. Qualification is specific to the API, backend and device exercised.

Named after the optical Fabry-Pérot etalon, Etalon is an independent project founded and maintained by Timothy Charles Holborn ([mediaprophet](https://github.com/mediaprophet)). Its development began with an MIT-licensed fork of a wgpu source snapshot declaring version `30.0.0`, at revision [`e084b66aab767940ff1236604d194c9e1c2844b0`](https://github.com/gfx-rs/wgpu/tree/e084b66aab767940ff1236604d194c9e1c2844b0). That snapshot supplied some graphics, validation and shader-compiler foundations, including Naga.

Work to create Etalon began from that starting point on 20 October 2026. Etalon development combines substantial improvements to inherited functions with new libraries and systems for sensing, signal processing, generated compute kernels, tensor operations, semantic data and QualiaDB/Webizen integration. The wgpu reference identifies the origin of part of the codebase; Etalon's identity and scope are defined by its own development.

Etalon-specific work is licensed under **CC BY-NC-ND 4.0**, as set out in [`LICENSE`](LICENSE) and [`LICENSE-QUALIA`](LICENSE-QUALIA). The MIT reference identifies the terms of the original source snapshot and does not relicense Etalon's subsequent work.

Etalon, QualiaDB and other constituent projects in the ecosystem being developed share an interim licensing approach while the intended licensing and rights fabric and its supporting systems are developed. Each project's own licence notice identifies its applicable variant, version and covered material; Etalon-specific work currently uses CC BY-NC-ND 4.0. See [Licensing and rights fabric: purpose, development status and interim licensing](https://mediaprophet.github.io/license/) for the shared rationale and development status.

The intended framework supports cooperative work, contributor compensation and accountable licence transitions. Adoption depends on the required semantic and logic tooling, rights safeguards, contribution and settlement systems, and governance being ready for the defined use. A future transition requires an authorised decision for identified material; a compensation-based transition also requires full satisfaction of its defined agreed obligations. This statement of intent does not amend the licence attached to an existing release. Current project licences and third-party component notices continue to govern their respective material.

**Homepage:** https://mediaprophet.github.io/etalon/

**Capabilities:** [Etalon libraries and inherited-code improvements](https://mediaprophet.github.io/etalon/capabilities.html) | [Run the browser tests](https://mediaprophet.github.io/etalon/testing.html)

**Library guide:** [Getting started](https://mediaprophet.github.io/etalon/getting-started.html) | [Public API](https://mediaprophet.github.io/etalon/doc/etalon/) | [Sensing API](https://mediaprophet.github.io/etalon/doc/etalon_sense/)

The documentation site publishes the qualified `stable` release and records its exact source revision. Future development stays on `dev`; pin a release tag or full commit for reproducible consumers.

> **Project authorship and source history:** Etalon development and maintenance are independent of wgpu. Credits for inherited foundations describe their original authorship. See [Upstream & Attribution](#upstream--attribution) and [Licensing](#intellectual-property--licensing-notice).

---

## Hardened Capabilities & Fixes

### 1. Robust `pop_error_scope` Handling (Upstream Fix for Issue #7)

- **The Issue**: In upstream `wgpu` 30.0.0, calling `device.pop_error_scope().await` on `wasm32` panicked with `"Unexpected error: constructor= value="` whenever the error scope caught _no_ error. Modern `wasm-bindgen` (≥0.2.123) treats JS `null` as a present value in `into_option()`, turning `null` into `Some(null)` and triggering an internal panic that aborted the WebAssembly module and caused canvas viewports to hang or turn black.
- **The Fix**: Etalon inspects the resolved JS value in `future_pop_error_scope`, correctly returning `None` when the scope resolves to `null` or `undefined`.
- **Defensive Error Mapping**: Replaced the `panic!` in `error_from_js` with a graceful `crate::Error::Internal` variant, ensuring that unexpected JavaScript exceptions surface as typed Rust errors rather than process aborts.

### 2. Automatic Adapter Limits Sanitization

- **The Issue**: W3C WebGPU specification (§3.2.1) mandates that every key in `requiredLimits` passed to `requestDevice()` must exist in `adapter.limits`. When browsers deprecate or remove limits (e.g. `maxInterStageShaderComponents` removed by Chromium/Edge), or when running on constrained mobile/compatibility adapters, upstream unconditional limit serialization triggers a hard JavaScript `OperationError`.
- **The Fix**: Etalon queries the adapter's supported limits via `self.inner.limits()` and checks property presence (`Reflect::has`) before serializing each limit key into `requiredLimits`. Unrecognized or unsupported keys are omitted, eliminating device creation failures without requiring monkey-patching in host web pages.

### 3. Cooperative Matrix & Tensor-Core Support (#9741)

- **Integrated Memory Model Fix**: Incorporates `vulkanMemoryModelDeviceScope` for Vulkan cooperative matrices, enabling robust compilation and execution of tiled tensor-core GEMM compute kernels across modern GPUs in Qualia's WGSL Forge.

### 4. Zero-Heap and Bounded Cognitive Integration

- **Caller-Owned Sensing Buffers**: Portable sensing contracts and signal primitives use explicit caller buffers. The prepared native Vulkan compute lane reuses fixed resources; ordinary queue/map APIs and vendor runtimes may allocate. Strict hot zero-allocation admission requires evidence for the selected execution path and target.
- **Unified Native & Browser Rendering**: Directly bridges native desktop surfaces (Vulkan, DirectX 12, Metal) and browser WebGPU contexts within a single cohesive API.

---

## Architecture & Workspace Structure

```text
etalon/
├── etalon/                 # Safe, idiomatic Rust API for graphics and compute
│   └── src/backend/        # Native (etalon-core) and Browser (webgpu.rs) backends
├── etalon-core/            # Core state machine, validation, and resource management
├── etalon-hal/             # Hardware Abstraction Layer (Vulkan, DX12, Metal, GLES)
├── etalon-types/           # Common types, structures, flags, and device limits
├── etalon-naga-bridge/     # Maps etalon features/limits to naga validation capabilities
├── naga/                   # Shader translation and validation engine (WGSL, SPIR-V, MSL, HLSL)
├── crates/                 # Etalon extensions (agent, nquin, render, tensor, wasm, wgsl-forge)
├── examples/               # Standalone and feature examples
├── .github/workflows/      # CI pipelines for multiplatform verification
└── upstream-licenses/      # Upstream wgpu MIT and Apache-2.0 license texts
```

---

## Quickstart

Etalon is not published to crates.io (every workspace package sets `publish = false`). Depend on it through git:

```toml
[dependencies]
etalon = { git = "https://github.com/mediaprophet/etalon.git", tag = "v0.0.2-release" }
pollster = "1.0"
```

### Branches

Etalon is at an early stage and breaking changes are expected. Choose the branch that matches how much change your project can absorb:

| Target                        | Purpose                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| `v0.0.2-release` (tag)        | Immutable final 0.0.2 source release.                                                |
| `v0.0.2.x` (branch)           | Qualified 0.0.2 maintenance line.                                                    |
| `stable`, `latest` (branches) | Most recent qualified release; both remain on 0.0.2 until another release qualifies. |
| `dev` (branch)                | Further development; not a release guarantee.                                        |

Use `tag = "v0.0.2-release"` for 0.0.2, or `branch = "latest"` to follow qualified releases. Cargo.lock retains the selected commit until you explicitly update it. For an immutable build use the full release commit with `rev`. See the [0.0.2 final release notes](docs/releases/0.0.2-release.md).

The earlier `v0.0.2` tag remains the historical prerelease and does **not** contain the final fixes. It has not been moved. `trunk`, `main`, `qualia-main`, and `v0.0.1.x` are retained historical refs; they are not the current release channels.

> **Note:** Etalon's crates use the `etalon-*` names, or keep upstream names (`naga`, `naga-types`), at version `0.0.2`. A project that also depends on upstream `wgpu`/`naga` from crates.io gets two separate copies. Their types are not interchangeable.

Request an adapter and device:

```rust
fn main() {
    let instance = etalon::Instance::new(etalon::InstanceDescriptor::new_without_display_handle());

    let adapter = pollster::block_on(
        instance.request_adapter(&etalon::RequestAdapterOptions::default()),
    )
    .expect("Failed to find a suitable GPU adapter");

    let (device, _queue) = pollster::block_on(adapter.request_device(&etalon::DeviceDescriptor {
        label: Some("Etalon GPU Device"),
        required_features: etalon::Features::empty(),
        required_limits: etalon::Limits::downlevel_defaults(),
        default_queue: etalon::QueueDescriptor { label: None },
        experimental_features: etalon::ExperimentalFeatures::disabled(),
        memory_hints: etalon::MemoryHints::MemoryUsage,
        trace: etalon::Trace::Off,
    }))
    .expect("Failed to create device");

    println!("Etalon device initialized: {device:?}");
}
```

On the WebGPU (browser) backend, Etalon removes any requested limit keys the adapter does not expose before calling `requestDevice()` (see _Automatic Adapter Limits Sanitization_ above). More complete programs are in [`examples/standalone/`](examples/standalone/) and [`examples/features/`](examples/features/).

---

## Environment variables

Etalon uses `ETALON_*` environment variables wherever upstream wgpu uses `WGPU_*`. The `WGPU_*` names are **not** read, so projects migrating from wgpu need to rename them (for example `WGPU_BACKEND` → `ETALON_BACKEND`).

These variables take effect when an application opts in through the `from_env` / `with_env` helpers (for example `Backends::from_env()`, `InstanceFlags::with_env()`, `BackendOptions::from_env_or_default()`) or `etalon::util::initialize_adapter_from_env_or_default()`. The examples, tests and benchmarks in this repository use them.

| Variable                                                           | Effect                                                                       |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| `ETALON_BACKEND`                                                   | Comma-separated list of backends: `vulkan`, `dx12`, `metal`, `gl`, `webgpu`. |
| `ETALON_ADAPTER_NAME`                                              | Selects the first adapter whose name contains this value (case-insensitive). |
| `ETALON_POWER_PREF`                                                | Power preference: `low`, `high` or `none`.                                   |
| `ETALON_DEBUG`, `ETALON_VALIDATION`, `ETALON_GPU_BASED_VALIDATION` | Debugging and validation flags (`1` enables, `0` disables).                  |

Backend-specific variables (`ETALON_DX12_*`, `ETALON_GLES_MINOR_VERSION`, `ETALON_GL_*`, `ETALON_NOOP_BACKEND`) and the remaining instance flags are documented on `BackendOptions` and `InstanceFlags` in `etalon-types`.

---

## Verification & Build

### Desktop Build:

```bash
cargo build -p etalon
```

### WASM / Browser Build:

```bash
cargo check -p etalon --target wasm32-unknown-unknown
```

### Run Naga Shader Test Suite:

```bash
cargo test -p naga
```

Contributor documentation, including where tests live, is in [`docs/`](docs/) (see [`docs/testing.md`](docs/testing.md)).

---

## Upstream & Attribution

**Etalon founder and maintainer:** Timothy Charles Holborn ([mediaprophet](https://github.com/mediaprophet)). Etalon-specific contributions are distinct from the history of its inherited foundations. Contribution to wgpu before the separation does not, by itself, establish participation in the independent Etalon project.

The starting source was **[wgpu revision `e084b66aab767940ff1236604d194c9e1c2844b0`](https://github.com/gfx-rs/wgpu/tree/e084b66aab767940ff1236604d194c9e1c2844b0)**, whose workspace declared version **`30.0.0`**. It was forked under the MIT licence before the changes that began creating Etalon. The first Etalon development commit is [`bcfb954f`](https://github.com/mediaprophet/etalon/commit/bcfb954fbc2f35b28c1cd2ea6777311769c16b0c), dated 20 October 2026. The full source revision identifies the exact starting snapshot rather than assuming equivalence with a release tag.

The gfx-rs developers and contributors are credited for inherited wgpu/Naga material. Etalon's subsequent improvements to that material and its new libraries have their own development history. The starting snapshot offered the MIT or Apache-2.0 licence; both original texts are preserved:

- [Apache License 2.0](upstream-licenses/LICENSE-APACHE)
- [MIT License](upstream-licenses/LICENSE-MIT)

Material that originates from upstream wgpu remains available under those terms. The upstream copyright and license notices are preserved in [`upstream-licenses/`](upstream-licenses/). [`CHANGELOG.md`](CHANGELOG.md) records Etalon changes only. For pre-fork history, see the [wgpu changelog at the starting snapshot](https://github.com/gfx-rs/wgpu/blob/e084b66aab767940ff1236604d194c9e1c2844b0/CHANGELOG.md).

---

## Intellectual Property & Licensing Notice

QualiaDB, Webizen, Etalon, and all Qualia-specific architectures, integrations, modifications, and extensions are governed by:

**Copyright (c) 2026 Timothy Charles Holborn**  
**[LinkedIn](https://www.linkedin.com/in/ubiquitous/) | [Email](mailto:timothy.holborn@gmail.com)**  
**License:** Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)  
Full terms: [`LICENSE`](LICENSE) / [`LICENSE-QUALIA`](LICENSE-QUALIA) and [Creative Commons Legal Code](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).

Upstream [wgpu](https://github.com/gfx-rs/wgpu) code retains its Apache-2.0 and MIT licensing, as documented in [`upstream-licenses/`](upstream-licenses/).

All participating AI agents, contributors, and downstream integrators must strictly honor this copyright, moral rights, and non-commercial attribution. Unlawful extraction, stripping of provenance, or unauthorized commercial exploitation is strictly prohibited.
