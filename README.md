# Etalon — Fiduciary WebGPU & Graphics Standard for QualiaDB & Webizen

> High-performance, memory-safe, cross-platform graphics and GPU compute engine hardened for QualiaDB cognitive pipelines, Aether spatial hosts, and Webizen OS-Shell.

[![CI](https://github.com/mediaprophet/etalon/actions/workflows/ci.yml/badge.svg)](https://github.com/mediaprophet/etalon/actions/workflows/ci.yml)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](LICENSE)
[![Upstream: wgpu (MIT / Apache-2.0)](https://img.shields.io/badge/Upstream-gfx--rs%2Fwgpu_(MIT_%2F_Apache--2.0)-informational.svg)](https://github.com/gfx-rs/wgpu)
[![Ecosystem: QualiaDB](https://img.shields.io/badge/Ecosystem-QualiaDB-blue.svg)](https://github.com/mediaprophet/qualiaDB)
[![Platform: Vulkan_|_Metal_|_DX12_|_WebGPU](https://img.shields.io/badge/Backends-Vulkan_%7C_Metal_%7C_DX12_%7C_WebGPU-orange.svg)](#architecture--workspace-structure)

**Etalon** is the human-agency and fiduciary GPU acceleration and graphics standard powering the spatial computing environment of [Webizen OS-Shell](https://github.com/WebizenAI/webizen-os-shell), the [Aether](https://github.com/mediaprophet/rust-app-framework) human-agency application framework, and [QualiaDB](https://github.com/mediaprophet/qualiaDB)'s 10D manifold neural rendering and WGSL Forge compute kernels.

Named after the optical Fabry-Pérot etalon—the scientific archetype of precision measurement and wave interferometry—Etalon is maintained as an enhanced, hardened fork of [wgpu](https://github.com/gfx-rs/wgpu) 30 by the gfx-rs developers. It provides an authoritative reference implementation that resolves critical upstream edge cases, W3C specification churn, and browser/webview incompatibilities that disrupt production WebGPU and native graphics deployments.

**Homepage:** https://mediaprophet.github.io/etalon/

> **Upstream:** Etalon is derived from [gfx-rs/wgpu](https://github.com/gfx-rs/wgpu), which is dual-licensed under Apache-2.0 and MIT. See [Upstream & Attribution](#upstream--attribution) and [Licensing](#intellectual-property--licensing-notice).

---

## Hardened Capabilities & Fixes

### 1. Robust `pop_error_scope` Handling (Upstream Fix for Issue #7)
- **The Issue**: In upstream `wgpu` 30.0.0, calling `device.pop_error_scope().await` on `wasm32` panicked with `"Unexpected error: constructor= value="` whenever the error scope caught *no* error. Modern `wasm-bindgen` (≥0.2.123) treats JS `null` as a present value in `into_option()`, turning `null` into `Some(null)` and triggering an internal panic that aborted the WebAssembly module and caused canvas viewports to hang or turn black.
- **The Fix**: Etalon inspects the resolved JS value in `future_pop_error_scope`, correctly returning `None` when the scope resolves to `null` or `undefined`.
- **Defensive Error Mapping**: Replaced the `panic!` in `error_from_js` with a graceful `crate::Error::Internal` variant, ensuring that unexpected JavaScript exceptions surface as typed Rust errors rather than process aborts.

### 2. Automatic Adapter Limits Sanitization
- **The Issue**: W3C WebGPU specification (§3.2.1) mandates that every key in `requiredLimits` passed to `requestDevice()` must exist in `adapter.limits`. When browsers deprecate or remove limits (e.g. `maxInterStageShaderComponents` removed by Chromium/Edge), or when running on constrained mobile/compatibility adapters, upstream unconditional limit serialization triggers a hard JavaScript `OperationError`.
- **The Fix**: Etalon queries the adapter's supported limits via `self.inner.limits()` and checks property presence (`Reflect::has`) before serializing each limit key into `requiredLimits`. Unrecognized or unsupported keys are omitted, eliminating device creation failures without requiring monkey-patching in host web pages.

### 3. Cooperative Matrix & Tensor-Core Support (#9741)
- **Integrated Memory Model Fix**: Incorporates `vulkanMemoryModelDeviceScope` for Vulkan cooperative matrices, enabling robust compilation and execution of tiled tensor-core GEMM compute kernels across modern GPUs in Qualia's WGSL Forge.

### 4. Zero-Heap and Bounded Cognitive Integration
- **Zero Allocations in Hot Paths**: Compatible with QualiaDB's Rule 0 (*Zero heap in hot paths*), allowing uniform belts, vertex/index staging buffers, and compute passes to stream directly from fixed-size memory arenas and stack arrays (`[NQuin; N]`).
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
etalon = { git = "https://github.com/mediaprophet/etalon.git", branch = "latest" }
pollster = "1.0"
```

### Branches

Etalon is at an early stage and breaking changes are expected. Choose the branch that matches how much change your project can absorb:

| Branch | Purpose |
|---|---|
| `stable` | For general use. Not created yet; it will be added once v0.0.1 is complete. |
| `latest` | For experimental users who want the newest usable state. Can change without notice. |
| `v0.0.1.x` | The v0.0.1 version line and its fixes. Projects that need a specific version can depend on this branch. Future version lines follow the same pattern. |
| `dev` | Active development. May be broken at any time. |

For reproducible builds, pin a specific commit with `rev = "<commit>"` instead of `branch`. `trunk` and `qualia-main` are retained for existing consumers.

> **Note:** Etalon's crates use the `etalon-*` names, or keep upstream names (`naga`, `naga-types`), at version `0.0.1`. A project that also depends on upstream `wgpu`/`naga` from crates.io gets two separate copies. Their types are not interchangeable.

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

On the WebGPU (browser) backend, Etalon removes any requested limit keys the adapter does not expose before calling `requestDevice()` (see *Automatic Adapter Limits Sanitization* above). More complete programs are in [`examples/standalone/`](examples/standalone/) and [`examples/features/`](examples/features/).

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

Etalon is a fork of **[wgpu](https://github.com/gfx-rs/wgpu)**, the cross-platform, safe, pure-Rust graphics API developed by the gfx-rs developers and contributors. This includes its shader translator **naga**. Upstream wgpu is dual-licensed under:

- [Apache License 2.0](upstream-licenses/LICENSE-APACHE)
- [MIT License](upstream-licenses/LICENSE-MIT)

Material that originates from upstream wgpu remains available under those terms. The upstream copyright and license notices are preserved in [`upstream-licenses/`](upstream-licenses/). [`CHANGELOG.md`](CHANGELOG.md) records Etalon changes only. For pre-fork history, see the [upstream wgpu changelog](https://github.com/gfx-rs/wgpu/blob/trunk/CHANGELOG.md).

---

## Intellectual Property & Licensing Notice

QualiaDB, Webizen, Etalon, and all Qualia-specific architectures, integrations, modifications, and extensions are governed by:

**Copyright (c) 2026 Timothy Charles Holborn**  
**[LinkedIn](https://www.linkedin.com/in/ubiquitous/) | [Email](mailto:timothy.holborn@gmail.com)**  
**License:** Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)  
Full terms: [`LICENSE`](LICENSE) / [`LICENSE-QUALIA`](LICENSE-QUALIA) and [Creative Commons Legal Code](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).  

Upstream [wgpu](https://github.com/gfx-rs/wgpu) code retains its Apache-2.0 and MIT licensing, as documented in [`upstream-licenses/`](upstream-licenses/).

All participating AI agents, contributors, and downstream integrators must strictly honor this copyright, moral rights, and non-commercial attribution. Unlawful extraction, stripping of provenance, or unauthorized commercial exploitation is strictly prohibited.
