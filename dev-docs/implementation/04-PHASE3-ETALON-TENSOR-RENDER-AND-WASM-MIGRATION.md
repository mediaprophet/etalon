# Runbook 04: Phase 3 — Extracting `etalon-tensor`, `etalon-render`, and `etalon-wasm`

- **Prerequisite:** Complete [`03-PHASE2-ETALON-AGENT-FORGE-AND-NQUIN-MIGRATION.md`](./03-PHASE2-ETALON-AGENT-FORGE-AND-NQUIN-MIGRATION.md).
- **Target Crates:** `crates/etalon-tensor`, `crates/etalon-render`, `crates/etalon-wasm`
- **Objective:** Extract the sovereign tensor execution engine, the decoupled volumetric/HUD/signal-processing renderer, and the cross-origin/transferable WASM worker steward.

---

## Task 4.1: Create `crates/etalon-tensor` (Prepared Tensor Execution & Unified Forward Arena)

### Module Layout (`< 500` lines per file, `Rule 0-B`)

```text
c:\github\etalon\crates\etalon-tensor\
├── Cargo.toml
└── src\
    ├── lib.rs
    ├── view.rs            # Validated TensorView, QuantScheme (Q4_K_SOA, Q8_0, Q8_1, FP16, FP32)
    ├── kv_cache.rs        # KvCacheLayout (FP32, INT8, DictK, PagedKv BlockTable)
    ├── forward_arena.rs   # Unified Decode (B=1) + Prefill (B<=64) + Verify (B<=16) Arena!
    ├── attention.rs       # Online-softmax FlashAttention (with explicit all-masked row guard)
    ├── gpu_sieve_topk.rs  # In-shader SieveStateMask + GraphLogitBias + Block-Top-1/Top-K
    ├── cuda_mega.rs       # Optional #[cfg(feature = "cuda")] CapturedCudaGraph mega_pass
    └── dml_lane.rs        # Optional #[cfg(target_os = "windows")] DirectML executor (no DirectStorage stub claims)
```

### Mandatory Engineering Improvements During Migration

1. **Collapse the 3 Duplicated Arenas (`resident_decode.rs`, `prefill_arena.rs`, `verify_arena.rs`) into `forward_arena.rs`:**
   - As documented in [`verify_arena.rs:22`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/gguf_bridge/verify_arena.rs#L22), `verify_arena.rs` and `prefill_arena.rs` duplicate ~80% of `resident_decode.rs` and create ~420 `wgpu::BindGroup` objects per model.
   - In `forward_arena.rs`, parameterize the forward pass by `ForwardMode`:
     - `ForwardMode::Prefill { batch_len: u32 }` (populates KV cache only; skips output RMSNorm and logits GEMM).
     - `ForwardMode::DecodeSingle { pos: u32 }` (runs output RMSNorm + chunked logits GEMV + GPU Block-Top-1; reads back **4 bytes**).
     - `ForwardMode::SpeculativeVerify { start_pos: u32, batch_len: u32 }` (runs output RMSNorm + batched logits GEMM + in-shader per-row `argmax`; reads back **`batch_len * 4` bytes** (`<= 64 bytes`) instead of $9.72\text{ MiB}$!).

2. **Explicit All-Masked Guard in Online Softmax (`attention.rs`):**
   - In `fused_attention.wgsl` / `flash_attn_wgsl.rs`, ensure that if all keys in a tile or row are masked out (`m_new == -FLT_MAX`), `exp(m_old - m_new)` does **not** evaluate `exp(-inf - (-inf)) = exp(NaN)`. Guard with:

```wgsl
let safe_diff = select(m_old - m_new, 0.0, m_new <= -3.0e38);
let rescale = select(exp(safe_diff), 0.0, m_old <= -3.0e38);
```

3. **Keep GGUF/Safetensors File Parsing and Tokenization in `QualiaDB`:**
   - `gguf_sharder.rs`, `safetensor_loader.rs`, `domino_gbnf.rs`, and `qualia_hybrid.rs` policy remain in `QualiaDB`; `etalon-tensor` accepts `TensorView` slices, `KvCacheLayout`, and a compact `[u32; 16]` token-allow/bias mask.

---

## Task 4.2: Create `crates/etalon-render` (Decoupled Volumetric, HUD, Audio & EMF Substrate)

### Problem Statement (`Codex Review §2.2 P2`)

In [`c:\github\qualiaDB\crates\qualia-core-db\src\render\gpu\mod.rs:66-76, 430-565`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/render/gpu/mod.rs#L66-L76), `PortalGpu` directly imports database-specific types (`Camera`, `Telemetry`, `Tensor10D`, `VramLedger`, container opacity rules). Moving the folder as-is would force `etalon-render` to depend on `QualiaDB`.

### Step-by-Step Decoupling Instructions

1. **Define Pure POD Scene & Frame Contracts in `crates/etalon-render/src/scene_contract.rs`:**

```rust
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrameCameraPod {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub eye_pos: [f32; 4],
    pub viewport_wh_near_far: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ManifoldProjectionPod {
    pub coords_10d: [f32; 10],
    pub opacity: f32,
    pub flags: u32,
}
```

2. **Use `TextureUsages::TRANSIENT` Correctly (`etalon-types/src/texture.rs:191`):**
   - Apply `TextureUsages::TRANSIENT` **only** to intermediate per-frame depth/MSAA targets whose contents are discarded at the end of the render pass.
   - **Never** mark `temporal_resolve_gpu.rs` history textures or picking readback textures as `TRANSIENT`.
3. **Include the Reusable Signal/Physics Modules (`initial-doc.md:569-613`):**
   - Migrate [`wgsl_forge/audio/mel.rs`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/wgsl_forge/audio/mel.rs) (GPU Mel filterbank), [`wgsl_forge/physics/kinematics.rs`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/wgsl_forge/physics/kinematics.rs), and [`render/gpu/emf_pipeline.rs`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/render/gpu/emf_pipeline.rs) into clean submodules (`etalon-render::emf`, `etalon-wgsl-forge::audio`, `etalon-wgsl-forge::physics`) with CPU-oracle tests.

---

## Task 4.3: Create `crates/etalon-wasm` (Dual-Transport Worker & `OffscreenCanvas` Steward)

### Module Layout

```text
c:\github\etalon\crates\etalon-wasm\
├── Cargo.toml
└── src\
    ├── lib.rs
    ├── sab_ring.rs        # COOP/COEP SharedArrayBuffer SPSC ring + Triple-Buffered Manifold
    ├── msg_transport.rs   # Transferable ArrayBuffer fallback when SAB is unavailable
    └── worker_steward.rs  # Async map_async completion + OffscreenCanvas ownership negotiation
```

### Mandatory SPSC & Browser Safety Rules

1. **Never Block the Worker Event Loop:**
   - Inside the Web Worker owning the `GPUDevice`, use `wasm_bindgen_futures::spawn_local` + async `Buffer::map_async` for the 4-byte token readback. Never run a synchronous `for _ in 0..10_000` spin-poll loop.
2. **Dual-Transport Negotiation (`worker_steward.rs`):**
   - Detect `crossOriginIsolated` at startup:
     - When `true`: Instantiate `EtalonSharedBridge` (`sab_ring.rs`) over a 64-byte-aligned `SharedArrayBuffer` using `Ordering::Acquire` / `Ordering::Release` in Rust and `Atomics.load` / `Atomics.store` / `Atomics.waitAsync` in JS.
     - When `false`: Fall back automatically to `msg_transport.rs` using `Worker.postMessage(payload, [payload.buffer])` transferable `ArrayBuffer`s.
3. **Strict Backpressure & Generation Tracking:**
   - Never overwrite an unconsumed token slot in `token_ring` (`(write - read) < 256`).
   - Include `session_generation: AtomicU32` and `device_lost_flag: AtomicU32` in the control header so worker restarts or WebGPU `device.lost` events cleanly abort pending UI waits without deadlocking.

---

## Final End-to-End Acceptance Checklist

Before closing the migration:

1. [ ] `cargo tree -d` in both `c:\github\etalon` and `c:\github\qualiaDB` shows **zero** duplicate `naga`, `etalon`, or `wgpu` crates.
2. [ ] All 6 bugs in [`01-CRITICAL-BUG-FIXES-AND-TRUTHFULNESS-GATES.md`](./01-CRITICAL-BUG-FIXES-AND-TRUTHFULNESS-GATES.md) have passing regression tests.
3. [ ] `cargo check --workspace --all-targets` and `cargo clippy --workspace --all-targets` pass in both repositories.
4. [ ] `cargo check -p etalon -p etalon-wasm --target wasm32-unknown-unknown --no-default-features --features std,webgpu,wgsl` succeeds.
5. [ ] `cargo nextest run` passes all CPU-oracle parity tests for `etalon-nquin` (radix sort + warp scan), `etalon-wgsl-forge`, `etalon-tensor`, and `etalon-render`.
6. [ ] `cargo xtask sync-repo-files --check` passes cleanly.
