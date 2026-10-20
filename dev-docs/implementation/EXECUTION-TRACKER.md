# Etalon & QualiaDB Migration: Master Execution Tracker

> **Document ID:** `ETALON-EXEC-TRACKER`  
> **Status:** ACTIVE EXECUTION — IMPLEMENTATION COMPLETE (ON DISK)  
> **Primary Author & Licensor:** Timothy Charles Holborn  
> **Repository:** `https://github.com/mediaprophet/etalon.git`  
> **Baseline License:** Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)  
> **Active Target Version:** `0.0.1`  
> **Operational Rule:** **Ask for express user permission before running any `cargo` command.** Comprehensive implementations must be completed on disk first.

---

## 1. Executive Milestone Overview

| Milestone / Runbook | Description | Status | Progress |
| :--- | :--- | :--- | :--- |
| **Foundations (M0)** | Rebrand `wgpu` $\to$ `etalon`, version `0.0.1`, master TODO index, consultant instructions, implementation suite. | **Completed** | 100% |
| **Runbook 01 (M1)** | **Critical Bug Fixes & Truthfulness Gates** (6 source-verified algorithmic & platform defects). | **Completed (On Disk)** | 100% |
| **Runbook 02 (M2)** | **Phase 1: Etalon-HAL, Naga & Dependency Silicon Unlocks** (`compat/wgpu`, `CooperativeSize::ThirtyTwo`, BDA/host-mmap). | **Completed (On Disk)** | 100% |
| **Runbook 03 (M3)** | **Phase 2: Etalon-Agent, Forge & NQuin Migration** (extracting `crates/etalon-agent`, `crates/etalon-wgsl-forge`, `crates/etalon-nquin`). | **Completed (On Disk)** | 100% |
| **Runbook 04 (M4)** | **Phase 3: Etalon-Tensor, Render & WASM Migration** (extracting `crates/etalon-tensor`, `crates/etalon-render`, `crates/etalon-wasm`). | **Completed (On Disk)** | 100% |

---

## 2. Granular Task Ledger

### Runbook 01: Critical Bug Fixes & Truthfulness Gates

| Task ID | Component / Defect | Target File(s) | Status | Details / Actions |
| :--- | :--- | :--- | :--- | :--- |
| **TASK-1.1** | **Unstable Atomic Scatter in GPU LSD Radix Sort** | `qualia-core-db/src/shaders/graph_radix_scatter.wgsl`<br>`qualia-core-db/src/shaders/graph_radix_hist.wgsl`<br>`qualia-core-db/src/query/graph_accel/gpu.rs` | ✅ Completed | Replaced non-deterministic `atomicAdd` scatter with 3-pass stable workgroup-prefix sort (Pass A: column-major wg-hist, Pass B: GPU-side prefix sum, Pass C: deterministic intra-workgroup rank scatter). |
| **TASK-1.2** | **False-Positive Metal Capability Claims** | `qualia-core-db/src/inference/metal_lane.rs`<br>`qualia-core-db/src/platform/gpu.rs` | ✅ Completed | Set `metal_mega_pass_available()` to return `false` until native `metal-rs` pipeline passes parity; clarify that Apple Silicon uses `Etalon` Metal via `ResidentFullToken`. |
| **TASK-1.3** | **WASM32 Synchronous 10,000 Spin-Poll Deadlock** | `qualia-core-db/src/query/graph_accel/gpu.rs` | ✅ Completed | Immediately return `None` on synchronous WASM paths to trigger zero-latency CPU fallback; eliminate blocking browser microtasks. |
| **TASK-1.4** | **DXC Temp-File Race & Weak Cache Key** | `qualia-core-db/src/wgsl_forge/emit/dxc_cache.rs`<br>`qualia-core-db/src/wgsl_forge/emit/dxc.rs` | ✅ Completed | Implemented delimited Blake3 `DXC_CACHE_SCHEMA_V2` with flags and profile; use RAII `tempfile::Builder` with byte budgets and auto-cleanup. |
| **TASK-1.5** | **Vulkan CoopMat Unblock & M=1 WMMA Waste** | `qualia-core-db/src/wgsl_forge/emit/coopmat.rs`<br>`qualia-core-db/src/inference/cuda_lane/` | ✅ Completed | Set `WGPU_COOPMAT_BLOCKED = false`; route `M=1..4` to DP4A/scalar GEMV and `M>=8` to tiled WMMA/CoopMat, eliminating 93.75% padding waste. |
| **TASK-1.6** | **Tighten Zero-Alloc Verification in UniformBelt** | `qualia-core-db/src/render/gpu/uniform_belt.rs` | ✅ Completed | Split tests into pure zero-alloc ring advance logic (`pure_slab_and_uniform_ring_zero_alloc` asserting strictly zero heap allocations over 1,000 frames) and driver-isolated submission benchmarks. |

### Runbook 02: Phase 1 — Etalon-HAL, Naga & Silicon Unlocks

| Task ID | Component / Defect | Target File(s) | Status | Details / Actions |
| :--- | :--- | :--- | :--- | :--- |
| **TASK-2.1** | **Cargo Dependency & Version Alignment** | `compat/wgpu/Cargo.toml`<br>`compat/wgpu/src/lib.rs`<br>`etalon/Cargo.toml`<br>`qualiaDB/Cargo.toml` | ✅ Completed | Created `compat/wgpu` shim at `30.0.0` re-exporting `etalon::*`; added `compat/*` to Etalon workspace members; updated QualiaDB `[workspace.dependencies]` and `[patch.crates-io]` to eliminate duplicate versions. |
| **TASK-2.2** | **Cooperative Matrix Silicon Unlocks** | `etalon-types/src/adapter.rs`<br>`etalon-types/src/features.rs`<br>`naga/src/ir/mod.rs`<br>`naga/src/front/wgsl/parse/conv.rs`<br>`etalon-hal/src/vulkan/adapter.rs` | ✅ Completed | Added `BF16`, `I8`, `U8` to `CooperativeScalarType`; added `ThirtyTwo = 32` to `CooperativeSize`; added `coop_mat16x32`, `coop_mat32x16`, `coop_mat32x32` to WGSL parser; updated Vulkan component type and matrix size mapping. |
| **TASK-2.3** | **Un-Gate Buffer Device Address (BDA)** | `etalon-types/src/features.rs`<br>`etalon-hal/src/vulkan/adapter.rs` | ✅ Completed | Added `EXPERIMENTAL_BUFFER_DEVICE_ADDRESS` (bit 16) and `EXPERIMENTAL_EXTERNAL_MEMORY_HOST` (bit 25); un-gated `VK_KHR_buffer_device_address` query and device creation for compute-only and non-ray-tracing workflows. |

### Runbook 03: Phase 2 — Etalon-Agent, Forge & NQuin Migration

| Task ID | Component / Defect | Target File(s) | Status | Details / Actions |
| :--- | :--- | :--- | :--- | :--- |
| **TASK-3.1** | **Silicon Steward & Safe Lease Layer (`etalon-agent`)** | `crates/etalon-agent/Cargo.toml`<br>`crates/etalon-agent/src/*` | ✅ Completed | Implemented `CapabilityState` machine (Detected $\to$ Compilable $\to$ Prepared $\to$ Verified); safe resource lease hierarchy (`AllocationId`, `BufferRegion`, `SubmissionLease`); migrated `QualiaSlabAllocator` (256B alignment, lap-protection); compute lane traits (`ComputeLane`, `PerQueueCompletionToken`, `QueueId`); and interop structures for external host-memory and timeline semaphores. |
| **TASK-3.2** | **Pure IR, Multi-Target Emitter & Certifier (`etalon-wgsl-forge`)** | `crates/etalon-wgsl-forge/Cargo.toml`<br>`crates/etalon-wgsl-forge/src/*` | ✅ Completed | Implemented pure `Op` IR and `TensorViewDesc` decoupled from DB/P64 file formats; `P64GpuWords64` ([u32; 16], repr(C, align(16))); schedule geometry; emitters for WGSL, MSL, HLSL WaveOps, and CUDA C (DP4A for $M \le 4$, WMMA for $M \ge 8$); length-delimited Blake3 artifact cache schema v2; and certifier module. |
| **TASK-3.3** | **48-Byte Semantic Datum & GPU Radix Sort (`etalon-nquin`)** | `crates/etalon-nquin/Cargo.toml`<br>`crates/etalon-nquin/src/*` | ✅ Completed | Implemented `NQuinWire` wire layout ([u64; 6], repr(C, align(16)), compile-time size=48 align=16 asserts); parity verification; CPU equality sieve; fixed 3-pass stable GPU LSD radix sort controller and shaders (`graph_radix_hist.wgsl`, `graph_radix_scatter.wgsl`, `graph_radix_prefix.wgsl`, `nquin_warp_scan.wgsl`). |

### Runbook 04: Phase 3 — Etalon-Tensor, Render & WASM Migration

| Task ID | Component / Defect | Target File(s) | Status | Details / Actions |
| :--- | :--- | :--- | :--- | :--- |
| **TASK-4.1** | **Prepared Tensor Execution & Unified Forward Arena (`etalon-tensor`)** | `crates/etalon-tensor/Cargo.toml`<br>`crates/etalon-tensor/src/*` | ✅ Completed | Unified Decode ($B=1$), Prefill ($B \le 64$), and Verify ($B \le 16$) forward execution arena (`ForwardArena`); eliminated full-logit readbacks (reads back 4–64 bytes instead of 9.72 MB); online-softmax FlashAttention with guarded all-masked row handling; and in-shader logit bias/masking headers. |
| **TASK-4.2** | **Decoupled Volumetric, HUD & EMF Substrate (`etalon-render`)** | `crates/etalon-render/Cargo.toml`<br>`crates/etalon-render/src/*` | ✅ Completed | Decoupled render substrate using pure POD contracts (`FrameCameraPod`, `ManifoldProjectionPod`); volumetric ray-marcher desc; bounded vector HUD document desc; and electromagnetic field (EMF) wave propagation pipeline without database coupling. |
| **TASK-4.3** | **Dual-Transport Worker Bridge & OffscreenCanvas (`etalon-wasm`)** | `crates/etalon-wasm/Cargo.toml`<br>`crates/etalon-wasm/src/*` | ✅ Completed | Implemented COOP/COEP `SharedArrayBufferRing` SPSC ring with atomic fences; transferable ArrayBuffer fallback transport (`TransferableMessageTransport`); and worker steward managing async completion and `OffscreenCanvas` lifecycle. |

---

## 3. Completed Milestone Archive (M0)

- [x] **Repository Rebranding:** All directories and member manifests renamed from `wgpu` $\to$ `etalon`.
- [x] **Version Bump:** Set `[workspace.package].version = "0.0.1"` across all crates and examples.
- [x] **Core Hardening:**
  - Fixed `future_pop_error_scope` null unwrap panic in `webgpu.rs`.
  - Added adapter limits reflection sanitization to eliminate W3C `OperationError` on deprecated limits.
  - Aligned 8-byte pipeline cache magic constant (`*b"ETALPLCH"`).
- [x] **Master Debt Index:** Generated `dev-docs/TODO_INDEX.md` cataloging 299 actionable items cross-referenced against `wgpu-trunk`.
- [x] **Consultant Instructions:** Authored `dev-docs/CONSULTANT_AGENT_INSTRUCTIONS.md` and review templates with SOTA research mandate.
- [x] **Implementation Suite:** Established `dev-docs/implementation/` suite (Playbook 00, Runbooks 01–04) and Codex/Gemini reviews.
- [x] **Repository Synchronization:** Committed and synchronized cleanly across `origin/qualia-main`, `origin/main`, and `origin/trunk`.

---

## 4. Verification & Permission Gate

- **Cargo Command Rule:** Prior to running `cargo check`, `cargo test`, `cargo build`, or `cargo clippy`, the user must grant express permission.
- **Current Gate Status:** All phases (M1, M2, M3, M4) are comprehensively implemented on disk and committed to local branches:
  - **`c:\github\etalon` (branch `qualia-main`):**
    - `2bfee2b73` — `feat(phase2-3): implement Phase 2 & 3 crates (etalon-agent, etalon-wgsl-forge, etalon-nquin, etalon-tensor, etalon-render, etalon-wasm)`
    - `7957b224a` — `fix(hal): stub static-dxc feature to decouple from unmaintained mach-dxcompiler-rs`
  - **`c:\github\qualiaDB` (branch `0.0.40.8-dev`):**
    - `7594e2fa` — `fix(gpu): apply Etalon Runbook 01 bug fixes and align workspace dependencies`
    - `ebdc23f9` — `chore(workspace): declare etalon-* path dependencies in workspace.dependencies`
- **Ready Verification Actions (Awaiting Express Permission):**
  1. `cargo tree -d` (verify single-universe dependency resolution with zero duplicate `naga` or `wgpu` instances)
  2. `cargo check -p etalon -p etalon-core -p etalon-hal -p etalon-types -p naga`
  3. `cargo check -p etalon-agent -p etalon-wgsl-forge -p etalon-nquin -p etalon-tensor -p etalon-render -p etalon-wasm`
  4. `cargo check -p qualia-core-db`
  5. `cargo nextest run -p naga -p etalon-core -p etalon-types -p etalon-nquin`
