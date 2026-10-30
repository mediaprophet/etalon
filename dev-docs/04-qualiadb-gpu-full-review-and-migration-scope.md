# Etalon Architectural Audit: Complete QualiaDB GPU, CUDA, Vulkan & Inference Scope

> **Document ID:** `ETALON-AUDIT-04`  
> **Status:** READY FOR CONSULTANT AGENT REVIEW & IMPLEMENTATION PLANNING  
> **Author:** Timothy Charles Holborn & Antigravity  
> **Date:** October 2026  
> **Target System:** Etalon Engine (`https://github.com/mediaprophet/etalon`)  
> **Source System:** QualiaDB (`https://github.com/mediaprophet/qualiaDB`)  
> **Licensing Baseline:** CC BY-NC-ND 4.0 International; target transition to MediaProphet Permissive Commons

---

## 1. Executive Summary & Purpose

This audit serves as the comprehensive briefing package for **consultant agents, system architects, and core maintainers**. It provides an exhaustive, forensic inventory of all GPU, CUDA, Vulkan, DirectML, Metal, LLM inference, 3D volumetric, and shader compiler subsystems currently embedded within [QualiaDB](https://github.com/mediaprophet/qualiaDB).

### Why Etalon?

QualiaDB was originally designed as a high-integrity, paraconsistent knowledge database and semantic agent runtime. However, due to limitations in upstream `wgpu` (such as lack of zero-copy buffer device addresses [BDA], missing native cooperative matrix / tensor core access on Windows, and high API allocation overhead in hot loops), QualiaDB accumulated over **75 low-level GPU and compiler modules**.

**Etalon** (`wgpu` fork rebranded, hardened, and versioned at `0.0.1`) is being established as the dedicated, high-performance silicon substrate. Migrating these subsystems into Etalon allows:

1. **Clean Architectural Boundaries:** QualiaDB focuses on knowledge graphs, deontic logic, and agent governance; Etalon focuses on low-overhead hardware execution, tensor compute, and rendering.
2. **First-Class Tensor Cores & Compute Queues:** Direct support for `VK_KHR_cooperative_matrix`, CUDA WMMA, and asynchronous compute queues directly within the HAL.
3. **Zero-Alloc Hot Paths:** Enforcing a zero-heap execution contract across both graphics passes and transformer token decode loops.

---

## 2. Forensic Subsystem Inventory Across QualiaDB

The table below catalogs every GPU-related subsystem identified across QualiaDB's codebase:

| Subsystem                                            | Location in `qualiaDB`                                                     | Lines of Code / Modules            | Core Capabilities & Technologies                                                                                                                                                                                                                                                                                                                                           |
| :--------------------------------------------------- | :------------------------------------------------------------------------- | :--------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. WGSL Forge & Multi-Backend JIT**                | `crates/qualia-core-db/src/wgsl_forge/`                                    | 13 files + 8 subdirs (~28,000 LOC) | • Multi-backend emitters: CUDA-C, PTX, HLSL (DXC), MSL, SPIR-V, WGSL.<br>• Cooperative matrix / WMMA tensor-core tiled GEMM.<br>• Naga validation, CPU oracle verification, GPU hardware profiling.<br>• Adapter-keyed persistent shader cache.                                                                                                                            |
| **2. CUDA Native Execution Lane**                    | `crates/qualia-core-db/src/inference/cuda_lane/`                           | 5 files + 4 subdirs (~8,500 LOC)   | • Direct CUDA driver API via NVRTC & PTX streams.<br>• `mega_pass.rs`: all-28-layer fused transformer decode without CPU D2H stalls.<br>• Slab allocation, PagedAttention, and Q8_0/Q4_K weight caching.                                                                                                                                                                   |
| **3. Vulkan & Hardware Adapter Routing**             | `crates/qualia-core-db/src/gpu_context.rs` & `wgsl_forge/execute/wgpu.rs`  | ~6,500 LOC                         | • `create_instance_for_coopmat()`: un-gates `VK_KHR_cooperative_matrix` on NVIDIA on Windows by routing around DX12 limitations.<br>• Multi-circuit device registry for heterogeneous iGPU/dGPU roles.<br>• VRAM reservation ledger (`OperationalMode::Full`, `Eco`, `Reserve`).                                                                                           |
| **4. DirectML & Metal Acceleration Bridges**         | `crates/qualia-core-db/src/inference/directml_bridge.rs` & `metal_lane.rs` | ~4,200 LOC                         | • Windows DirectML COM pipeline for AMD, Intel, and Qualcomm Snapdragon NPUs.<br>• Apple Silicon Metal Performance Shaders (MPS) unified memory zero-copy lane.                                                                                                                                                                                                            |
| **5. Local LLM Tensor Engine & P64 Container**       | `crates/qualia-core-db/src/inference/` & `gguf_bridge/`                    | ~32,000 LOC                        | • Autoregressive token decode loop (`resident_decode.rs`).<br>• Standard 64-byte P64 weight container (`p64_weight.rs`) with per-tensor CRC-32C.<br>• Zero-copy token embedding dequantization (`ggml_quants.rs`).<br>• Ternary weights (`ternary_gpu.rs`), GPU Top-K selection (`topk_gpu.rs`), Speculative decode.                                                       |
| **6. Volumetric 10D Manifold & Portal GPU Renderer** | `crates/qualia-core-db/src/render/` & `crates/webizen-render/`             | ~65,000 LOC across 81+ modules     | • `PortalGpu` offscreen & canvas volumetric renderer.<br>• TAA / Temporal resolve (`temporal_resolve_gpu.rs`).<br>• GPU-driven indirect draw & occlusion culling (`indirect_cull.rs`).<br>• Zero-alloc UniformBelt ring buffers (`uniform_belt.rs`).<br>• Multi-channel Signed Distance Field text (`ui_msdf_text.rs`).<br>• Quantum field visualizer (`emf_pipeline.rs`). |
| **7. 3D Simulation & Physics Compute**               | `crates/qualia-sim/`                                                       | 20 modules (~12,000 LOC)           | • Atmospheric scattering, optics, geomechanics, kinematics, and ripple water compute.                                                                                                                                                                                                                                                                                      |
| **8. Computer Vision & Super-Resolution**            | `crates/qualia-vision/`                                                    | 20 modules (~9,500 LOC)            | • Edge-directed super-resolution, bicubic filtering, spatial feature extraction.                                                                                                                                                                                                                                                                                           |

---

## 3. Critical Bug Fixes, Workarounds & Hardening Insights Discovered in QualiaDB

During the development and testing of QualiaDB across real hardware (NVIDIA RTX A2000 12GB, Apple M-series, Intel iGPU, Chrome WebGPU, and mobile Safari), numerous critical hardware and API bugs were discovered and mitigated. **These must be codified directly into Etalon's core architecture:**

### A. WebGPU & Browser Runtime Hardening

1. **`mappedAtCreation` Unreachable Trap on WASM (Commit `774e5d8a`):**
   - _Issue:_ Wrapping mapped uniform rings by allocating dynamic 256-byte `mappedAtCreation` buffers caused browser WebGPU implementations to reject the buffer and abort execution.
   - _Fix in QualiaDB:_ WASM runtime was branched to upload portal uniforms via `queue.write_buffer`, while native desktop continues to use mapped memory rings.
   - _Etalon Action:_ Build this detection directly into `etalon::Queue` staging pools.
2. **Browser Adapter Probe Black-Hole (Commit `014a5298`):**
   - _Issue:_ Chrome can leave `navigator.gpu.requestAdapter()` pending indefinitely if hardware acceleration is blocked or drivers are blacklisted.
   - _Fix in QualiaDB:_ Implemented an explicit 3000ms timeout with automatic fallback to WebGL2/Naga ES 300.
   - _Etalon Action:_ Include watchdog timer and graceful fallback in `etalon::Instance::request_adapter`.
3. **Pop Error Scope Null Pointer Panic (Etalon Commit `bcfb954fb`):**
   - _Issue:_ Newer `wasm-bindgen` (>= 0.2.123) treats JS `null` as present in `Option<GpuError>`, causing unwrap panics when popping empty error scopes.
   - _Fix in Etalon:_ Explicitly check `is_null() || is_undefined()` and return `None`.
4. **W3C Deprecated Limit Rejection (Etalon Commit `bcfb954fb`):**
   - _Issue:_ Requesting deprecated limits like `maxInterStageShaderComponents` triggers `OperationError` on modern Chrome/Firefox.
   - _Fix in Etalon:_ Reflection-checked limit filtering in `map_js_sys_limits`.

### B. Graphics & Rendering Engine Hardening

5. **Pick Readback 256-Byte Row Alignment (`render/gpu/mod.rs`):**
   - _Issue:_ Reading a single 1x1 pixel texture for object picking crashed because `bytes_per_row` was set to 4, violating wgpu's `COPY_BYTES_PER_ROW_ALIGNMENT` (256 bytes).
   - _Fix in QualiaDB:_ Pad staging buffer to 256 bytes; slice only `[0..4]` upon mapped readback.
6. **Water Pass / Transparent Draw Index Buffer Collision (Commit `9cd20212`):**
   - _Issue:_ Binding a separate index buffer during water pass before completing transparent draw passes caused state clobbering in the render pipeline encoder.
   - _Fix in QualiaDB:_ Reordered frame graph passes to ensure transparent sorting concludes prior to secondary refractive passes.
7. **Uniform Control Flow in Viewport WGSL Shaders (Commit `8b908293`):**
   - _Issue:_ Texture sampling functions with implicit LOD (e.g. `textureSample`) inside non-uniform conditional branches caused shader stage validation panics and driver faults on Intel and Mali GPUs.
   - _Fix in QualiaDB:_ Enforced explicit gradient / LOD sampling (`textureSampleLevel`) or hoisted sampling out of conditional blocks.

### C. Compute, CUDA & Tensor Hardening

8. **Vulkan vs. DX12 Tensor Core Un-gating on Windows (`execute/wgpu.rs`):**
   - _Issue:_ On Windows with NVIDIA drivers, DX12 wgpu backend does NOT expose cooperative matrix (`coopmat=false`), reducing decode speeds to ~4 tok/s.
   - _Fix in QualiaDB:_ Added `create_instance_for_coopmat()` which specifically requests Vulkan (`VK_KHR_cooperative_matrix`), unlocking hardware tensor cores and achieving ~83 tok/s.
   - _Etalon Action:_ Allow `etalon-hal` to natively expose cooperative matrix across all supported backends, with automatic backend priority for compute workloads.
9. **Elimination of Per-Layer D2H Stalls (`cuda_lane/mega_pass.rs`):**
   - _Issue:_ Naive layer-by-layer CUDA execution suffered 84 serial CPU stalls per token (28 layers × 2 D2H readbacks + CPU RMSNorm/residual).
   - _Fix in QualiaDB:_ Fused all 28 layers into a single continuous CUDA stream launch, with only 4 bytes (the argmax token) read back to the host at the end of the frame.
10. **Zero-Alloc Uniform Belt (`commit 80914f3e`):**
    - _Issue:_ Standard `queue.write_buffer` allocated internal staging memory on every frame (~313 heap allocations/frame).
    - _Fix in QualiaDB:_ Introduced `UniformBelt` ring of pre-mapped buffers rotated per frame, dropping CPU overhead by 40%.

---

## 4. Proposed Etalon Target Crate Architecture

To cleanly ingest these capabilities, Etalon will be structured with the following modular crate hierarchy:

```
etalon/ (Workspace Root, v0.0.1)
├── etalon/                        # Core WebGPU Safe API & Facade
├── etalon-core/                   # State Tracker, Resource Registry & Validation
├── etalon-hal/                    # Hardware Abstraction Layer (Vulkan, DX12, Metal, WebGPU)
├── etalon-types/                  # Fixed Layout POD Structures & Enums
├── etalon-sync/                   # Lock-Free Sync Primitives & SPSC Rings
├── etalon-naga-bridge/            # Shader Translation Pipeline
├── naga/                          # Naga Shader IR & Translators
│
│   === NEW CRATES MIGRATED FROM QUALIADB ===
│
├── crates/etalon-agent/           # Direct Silicon Steward: BDA, Memory Import, Zero-Copy Host Mmap
├── crates/etalon-forge/           # Multi-Target Shader Forge (CUDA-C, PTX, HLSL, MSL, SPIR-V, WGSL)
├── crates/etalon-tensor/          # Local LLM Tensor Engine, Mega-Pass Scheduler, P64 Container & GEMV
├── crates/etalon-nquin/           # 48-byte Super-Quin GPU Accelerators, Graph Scans & Radix Joins
├── crates/etalon-render/          # Volumetric Portal Renderer, Temporal Resolve, Mesh Slab & MSDF UI
└── compat/wgpu/                   # Zero-Overhead Backwards Compatibility Facade for Upstream Ecosystem
```

---

## 5. Specific Questions for Consultant Agents

Before finalizing the migration execution schedule, we invite architectural feedback on the following key questions:

1. **HAL Integration vs. Separate Crate for CUDA / PTX:**
   - _Option A:_ Keep CUDA/PTX in `crates/etalon-forge` and `crates/etalon-tensor` as a dedicated compute path alongside `etalon-hal`.
   - _Option B:_ Integrate CUDA as a first-class backend inside `etalon-hal` implementing `etalon_hal::Api`.
   - _Discussion Point:_ Is the divergence between graphics render-pass abstractions and raw stream compute worth forcing into the `hal::Api` trait, or is a dedicated compute-lane preferable?

2. **Buffer Device Address (BDA) & Zero-Copy Mmap Exposure:**
   - For QualiaDB's 48-byte Quins and P64 weight containers, should Etalon expose raw 64-bit GPU virtual addresses (BDA) in safe Rust via a new `BufferAddress` capability, or remain bound to buffer/offset bind groups?

3. **Cooperative Matrix Standardization:**
   - How should Etalon expose cooperative matrix operations so that both WGSL compute shaders and native SPIR-V/DXC pipelines can use the same high-level matrix multiply abstractions without vendor-specific dialect fragmentation?

4. **WASM Multi-threading & SharedArrayBuffer Strategy:**
   - Given browser restrictions on WebGPU multithreading, what is the recommended architecture for running the token decode loop on a dedicated Web Worker while presenting rendered frames on the main thread canvas?
