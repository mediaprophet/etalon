# Master Implementation Playbook for Execution Agents

- **Target Workspace:** `C:\github\etalon` (and consumer integration in `C:\github\qualiaDB`)
- **Consolidated Sources:**
  - Foundational Plans: [`initial-doc.md`](../initial-doc.md), [`01-wgpu-to-etalon-refactor-plan.md`](../01-wgpu-to-etalon-refactor-plan.md), [`02-qualiadb-migration-inventory-and-roadmap.md`](../02-qualiadb-migration-inventory-and-roadmap.md), [`03-licensing-and-rights-architecture.md`](../03-licensing-and-rights-architecture.md), [`04-qualiadb-gpu-full-review-and-migration-scope.md`](../04-qualiadb-gpu-full-review-and-migration-scope.md), [`TODO_INDEX.md`](../TODO_INDEX.md)
  - Consultant Review A (Codex): [`2026-10-20-codex-cross-platform-migration-strategy-review.md`](../consultant-reviews/2026-10-20-codex-cross-platform-migration-strategy-review.md)
  - Consultant Review B (Gemini/Antigravity): [`2026-10-20-gemini4-sota-silicon-architecture-review.md`](../consultant-reviews/2026-10-20-gemini4-sota-silicon-architecture-review.md)
- **Intended Audience:** Lower-cost / high-speed implementation agents executing scoped coding, bug-fix, and migration tasks.

---

## 1. How to Use This Implementation Suite

Do **not** guess architecture or invent ad-hoc abstractions. Every design decision has already been resolved by the Principal Consultant Reviews and distilled into four actionable runbooks in this folder:

| Runbook | Scope | Prerequisite |
| :--- | :--- | :--- |
| **[`01-CRITICAL-BUG-FIXES-AND-TRUTHFULNESS-GATES.md`](./01-CRITICAL-BUG-FIXES-AND-TRUTHFULNESS-GATES.md)** | Fixes 6 source-verified correctness bugs and truthfulness defects in `QualiaDB` and `Etalon` (unstable GPU radix scatter, false-positive Metal availability, WASM spin-poll deadlock, DXC temp-file race & weak cache key, $93.75\%$-wasted `M=1` WMMA GEMV, and loose zero-alloc assertions). | None — execute first or alongside Phase 1. |
| **[`02-PHASE1-ETALON-HAL-AND-NAGA-SILICON-UNLOCKS.md`](./02-PHASE1-ETALON-HAL-AND-NAGA-SILICON-UNLOCKS.md)** | Unlocks `etalon-types`, `naga`, `etalon-hal`, and `etalon-core`: dependency unification, `16x16x32` INT8 Cooperative Matrix (`CooperativeSize::ThirtyTwo`), safe lease-backed BDA & host-mmap import, `ReusableComputeBundle`, and per-queue timeline synchronization. | None. |
| **[`03-PHASE2-ETALON-AGENT-FORGE-AND-NQUIN-MIGRATION.md`](./03-PHASE2-ETALON-AGENT-FORGE-AND-NQUIN-MIGRATION.md)** | Extracts `crates/etalon-agent`, `crates/etalon-wgsl-forge`, and `crates/etalon-nquin` with immediate vertical-slice verification against `QualiaDB`. | Runbook `01` & `02`. |
| **[`04-PHASE3-ETALON-TENSOR-RENDER-AND-WASM-MIGRATION.md`](./04-PHASE3-ETALON-TENSOR-RENDER-AND-WASM-MIGRATION.md)** | Extracts `crates/etalon-tensor`, `crates/etalon-render` (including audio/physics/EMF boundaries), and `crates/etalon-wasm` (worker & `SharedArrayBuffer` SPSC bridge). | Runbook `03`. |

---

## 2. The 10 Immovable Rules for Execution Agents

1. **Preserve the Safe API Boundary (`AGENTS.md`):**
   - `etalon-core`, `etalon-hal`, `etalon-types`, and `naga` process untrusted input when used in WebGPU/browser contexts.
   - Never expose a bare `u64` physical pointer as a safe API in `etalon`. Safe code must use lifetime- and generation-checked resource views (`OwnedAllocation` $\rightarrow$ `BufferRegion` $\rightarrow$ `PreparedPlan` $\rightarrow$ `SubmissionLease` retained until hardware queue completion). Raw BDA dereferencing and passthrough shaders (`Features::PASSTHROUGH_SHADERS`, `create_shader_module_trusted`) must remain explicitly `unsafe` or restricted to compiler-certified fixed kernels.
2. **Never Implement `etalon_hal::Api` for CUDA or DirectML:**
   - `etalon_hal::Api` (`etalon-hal/src/lib.rs:632-687`) requires `Surface`, `RenderPipeline`, `TextureView`, `Sampler`, and `BindGroup`. Do **not** create dummy graphics stubs for CUDA.
   - Instead, implement the compute-specialized `ComputeLane` executor inside `crates/etalon-agent` alongside `etalon-hal`, separating cold `prepare()` from hot `submit()`.
3. **Strict 48-Byte `NQuin` & Validated `P64` Boundary:**
   - `NQuin` is `[u64; 6]` with `#[repr(C, align(16))]` (`48 bytes` size, `48 bytes` array stride; `12 × u32` in WGSL). **Never pad `NQuin` to 64 bytes in storage.** Use the Warp-Transposed `uvec4` loader (`smem_quin[32][13]`) for coalesced GPU scans.
   - Keep `P64` file parsing, CRC-32C checksums, and database transaction lifecycles in `QualiaDB`. `Etalon` crates accept validated tensor views (address/handle, dtype, dimensions, strides, quantization scale layout, and ownership lease).
4. **Capability Truthfulness (`Detected` $\rightarrow$ `Compilable` $\rightarrow$ `Prepared` $\rightarrow$ `Verified`):**
   - Never report a hardware lane or feature as `available() == true` unless its kernel compilation and preparation probe actually succeed on the active device.
5. **No Subprocess Compilers or Unscoped Temp Files (`Rule 0-B`):**
   - Never spawn `dxc.exe` via `std::process::Command` or write files to `std::env::temp_dir()`. Any scratch directory must use `tempfile::TempDir` with RAII cleanup and a strict byte budget.
6. **File Size & Modularity Discipline (`Rule 0-B`):**
   - Every new implementation file in `crates/etalon-*` must stay **below 500 lines** and own a single responsibility. `mod.rs` files only route submodules and re-export public types.
7. **Zero-Heap Hot Paths (`Rule 0-A`):**
   - Tier-1 hot paths (per-token decode step, `ComputeLane::submit`, `ReusableComputeBundle` replay, `NQuin` predicate scan, audio callback, `UniformBelt` frame update) must allocate **0 bytes** on the heap (`assert_eq!(measured_allocs, 0)` in Etalon-owned code). Cold preparation (`prepare()`) may allocate bounded structures once.
8. **Per-Queue Completion Tracking (No Shared Monotonic Timeline Races):**
   - Do not share a single monotonic timeline counter across independent hardware queues (e.g., Compute Queue vs. Transfer Queue vs. CUDA Stream), as out-of-order queue execution can prematurely satisfy a wait. Track `(QueueId, u64)` completion pairs explicitly.
9. **Migrate Vertical Slices with Immediate Consumer & CPU Oracle Verification:**
   - Do not copy 75 files blindly and wait until the end to compile. Migrate one slice at a time, wire `QualiaDB` to consume it immediately, and run its CPU-oracle parity test before starting the next slice.
10. **Repo Sync & Code Style (`etalon/AGENTS.md`):**
    - Do not use emoji in source code or comments. Keep comments concise.
    - If you touch `AGENTS.md`, `.agents/`, or root license files, run `cargo xtask sync-repo-files --check`.
    - Never commit without explicit user permission.

---

## 3. Consolidated Architectural Resolution Matrix (Codex + Gemini Synthesis)

Where the initial planning docs (`01`–`04`) had open tensions, the two consultant reviews converged on the following binding resolutions:

| Topic | Initial Plan Assumption | Consolidated Binding Resolution |
| :--- | :--- | :--- |
| **Dependency Versioning** | Create `compat/wgpu` at `0.0.1` while `QualiaDB` requests `wgpu = "30"`, `naga = "30"`. | Cargo rejects a `0.0.1` package for a `"30"` requirement. In `QualiaDB/Cargo.toml`, replace `wgpu = "30"` and `naga = "30"` with direct workspace dependencies on `etalon` and `naga` (or provide a `compat/wgpu` shim stamped at version `30.0.0` that re-exports `etalon` and forwards all feature flags). Ensure only **one** `naga` and **one** `etalon` exist in `cargo tree -d`. |
| **CUDA vs. `etalon-hal`** | Ambiguous between `etalon-hal/src/cuda` vs. separate crate. | **Hybrid Executor in `etalon-agent`:** Keep `etalon-hal` for Vulkan/DX12/Metal/GLES/WebGPU. Implement CUDA (`cudarc`/PTX/`CUgraph`) and DirectML as `ComputeLane` executors in `etalon-agent`, with cold `prepare()` and hot `submit()`. |
| **BDA & Host `mmap`** | Raw `u64` pointers passed freely; assumed ReBAR makes host `mmap` equal to VRAM speed. | **Lease-Checked Views + Measured Placement:** Safe Rust uses `OwnedAllocation` $\rightarrow$ `BufferRegion` $\rightarrow$ `SubmissionLease` (preventing `munmap` before GPU queue completion). Use direct host-`mmap` import (`VK_EXT_external_memory_host` / `MTLBuffer(NoCopy)`) on UMA or single-pass streaming; use device-local VRAM for high-reuse weights (`reuse_count * bytes / link_bw` cost model). |
| **Cooperative Matrix / Tensor Cores** | Assumed `SM 6.8+ WaveMatrix` everywhere and `16x16x16` WMMA for `M=1` GEMV. | **Capability-Probed Schedules:** Unblock Vulkan `coopMultiplyAdd` (already fixed at `vulkan/adapter.rs:641`), add `SINT8`/`UINT8`/`BF16` and `K=32` (`CooperativeSize::ThirtyTwo`) to `naga` & `etalon-hal`, use **packed `DP4A` (`dot4I8Packed`)** for memory-bound decode ($M \in [1, 4]$), and use tiled `CoopMat` / `simdgroup_matrix` / `MMA` for $M \ge 8$. Gate DX12 LinAlg (`SM 6.10` preview) behind an experimental feature flag. |
| **Graph LSD Radix Sort** | Assumed `atomicAdd(&offsets[d], 1u)` in `graph_radix_scatter.wgsl` was a valid "unstable sort". | **Mandatory Bug Fix (Runbook 01):** An LSD radix sort **must** be stable within each digit pass, otherwise Pass $k+1$ scrambles the ordering established by Pass $k$ (turning `[1, 2]` into `[2, 1]`). Replace with a deterministic workgroup-histogram + device prefix-sum + intra-workgroup rank scatter. |
| **WASM Worker Topology** | Assumed `SharedArrayBuffer` is always available and `mappedAtCreation` is universally broken. | **Dual-Transport Worker Steward:** Dedicated Worker owns the decode `GPUDevice`. Support **both** COOP/COEP `SharedArrayBuffer` SPSC rings (`EtalonSharedBridge` with `Atomics.waitAsync`) **and** standard `postMessage` transferable `ArrayBuffer` fallback when cross-origin isolation is unavailable. Use `OffscreenCanvas` when worker-owned rendering is selected. |

---

## 4. Standard Verification Commands

Run these commands after every task before marking a milestone complete:

```powershell
# 1. Verify single-universe dependency graph (zero duplicate naga/etalon versions)
cargo metadata --locked --format-version 1 > $null
cargo tree -d

# 2. Native compile & lint check across core crates
cargo check -p etalon -p etalon-core -p etalon-hal -p etalon-types -p naga
cargo clippy --workspace --all-targets

# 3. WASM32 browser target compilation gate
cargo check -p etalon --target wasm32-unknown-unknown --no-default-features --features std,webgpu,wgsl

# 4. Fast unit test execution (no GPU required)
cargo nextest run -p naga -p etalon-core -p etalon-types

# 5. Targeted GPU test execution (when touching HAL / shaders)
cargo xtask test -E 'binary(etalon-gpu) & test(cooperative_matrix)'

# 6. Repository file synchronization check
cargo xtask sync-repo-files --check
```
