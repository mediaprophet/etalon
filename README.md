# Etalon — Fiduciary WebGPU & Graphics Standard for QualiaDB & Webizen

> High-performance, memory-safe, cross-platform graphics and GPU compute engine hardened for QualiaDB cognitive pipelines, Aether spatial hosts, and Webizen OS-Shell.

[![CI](https://github.com/mediaprophet/etalon/actions/workflows/ci.yml/badge.svg)](https://github.com/mediaprophet/etalon/actions/workflows/ci.yml)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](LICENSE-QUALIA)
[![Ecosystem: QualiaDB](https://img.shields.io/badge/Ecosystem-QualiaDB-blue.svg)](https://github.com/mediaprophet/qualiaDB)
[![Platform: Vulkan_|_Metal_|_DX12_|_WebGPU](https://img.shields.io/badge/Backends-Vulkan_%7C_Metal_%7C_DX12_%7C_WebGPU-orange.svg)](#architecture--workspace-structure)

**Etalon** is the human-agency and fiduciary GPU acceleration and graphics standard powering the spatial computing environment of [Webizen OS-Shell](https://github.com/WebizenAI/webizen-os-shell), the [Aether](https://github.com/mediaprophet/rust-app-framework) human-agency application framework, and [QualiaDB](https://github.com/mediaprophet/qualiaDB)'s 10D manifold neural rendering and WGSL Forge compute kernels.

Named after the optical Fabry-Pérot etalon—the scientific archetype of precision measurement and wave interferometry—Etalon is maintained as an enhanced, hardened fork of `etalon` 30. It provides an authoritative reference implementation that resolves critical upstream edge cases, W3C specification churn, and browser/webview incompatibilities that disrupt production WebGPU and native graphics deployments.

---

## Hardened Capabilities & Fixes

### 1. Robust `pop_error_scope` Handling (Upstream Fix for Issue #7)
- **The Issue**: In upstream `etalon` 30.0.0, calling `device.pop_error_scope().await` on `wasm32` panicked with `"Unexpected error: constructor= value="` whenever the error scope caught *no* error. Modern `wasm-bindgen` (≥0.2.123) treats JS `null` as a present value in `into_option()`, turning `null` into `Some(null)` and triggering an internal panic that aborted the WebAssembly module and caused canvas viewports to hang or turn black.
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
│   └── src/backend/      # Native (etalon-core) and Browser (webgpu.rs) backends
├── etalon-core/            # Core state machine, validation, and resource management
├── etalon-hal/             # Hardware Abstraction Layer (Vulkan, DX12, Metal, GLES)
├── etalon-types/           # Common types, structures, flags, and device limits
├── naga/                 # Shader translation and validation engine (WGSL, SPIR-V, MSL, HLSL)
├── .github/workflows/    # Clean CI pipelines for multiplatform verification
└── upstream-licenses/    # Upstream etalon MIT and Apache-2.0 license texts
```

---

## Quickstart

Add Etalon to your `Cargo.toml`:

```toml
[dependencies]
etalon = { git = "https://github.com/mediaprophet/etalon.git", branch = "qualia-main" }
```

Request an adapter and device with automatic limit filtering:

```rust
use etalon::{Instance, Backends, InstanceDescriptor, RequestAdapterOptions, DeviceDescriptor};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: etalon::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("Failed to find suitable GPU adapter");

    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            label: Some("Etalon GPU Device"),
            required_features: etalon::Features::empty(),
            // Automatically sanitized against adapter capabilities:
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await?;

    println!("Etalon Device initialized: {:?}", device);
    Ok(())
}
```

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

---

## Intellectual Property & Licensing Notice

QualiaDB, Webizen, Etalon, and all Qualia-specific architectures, integrations, modifications, and extensions are governed by:

**Copyright (c) 2026 Timothy Charles Holborn**  
**[LinkedIn](https://www.linkedin.com/in/ubiquitous/) | [Email](mailto:timothy.holborn@gmail.com)**  
**License:** Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)  
Full terms: [`LICENSE-QUALIA`](LICENSE-QUALIA) and [Creative Commons Legal Code](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).  

Upstream etalon framework code retains its respective Apache-2.0 and MIT licensing as documented in [`upstream-licenses/`](upstream-licenses/).

All participating AI agents, contributors, and downstream integrators must strictly honor this copyright, moral rights, and non-commercial attribution. Unlawful extraction, stripping of provenance, or unauthorized commercial exploitation is strictly prohibited.
