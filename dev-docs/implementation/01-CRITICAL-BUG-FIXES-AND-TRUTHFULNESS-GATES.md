# Runbook 01: Critical Bug Fixes & Truthfulness Gates

- **Prerequisite:** Read [`00-MASTER-IMPLEMENTATION-PLAYBOOK.md`](./00-MASTER-IMPLEMENTATION-PLAYBOOK.md)
- **Objective:** Resolve six source-verified correctness and truthfulness defects in `QualiaDB` and `Etalon` so that corrupted algorithms are **not** migrated into `Etalon`.
- **Execution Rule:** Fix each bug in an isolated, testable step and run its verification command before proceeding to the next bug.

---

## Bug 1.1: Fix Unstable Atomic Scatter in GPU LSD Radix Sort

### Root Cause
In [`c:\github\qualiaDB\crates\qualia-core-db\src\shaders\graph_radix_scatter.wgsl`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/shaders/graph_radix_scatter.wgsl#L27-L38) (lines 27–38):
```wgsl
let d = digit_at(i, params.shift);
let pos = atomicAdd(&offsets[d], 1u);
keys_out[pos * 2u] = keys_in[i * 2u];
```
**Why this is a fatal correctness bug:** Least-Significant-Digit (LSD) radix sort requires **every digit pass to be a stable permutation** of the previous pass. Because `atomicAdd(&offsets[d], 1u)` assigns output slots in hardware thread/workgroup arrival order rather than input index order `i`, Pass 1..7 (where high bytes of `[1, 2]` are both `0x00`) can schedule thread `i = 1` before thread `i = 0`, turning the already-sorted `[1, 2]` into `[2, 1]`! The host loop at [`query/graph_accel/gpu.rs:323-336`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/query/graph_accel/gpu.rs#L323-L336) detects that the output is unsorted and silently falls back to CPU after wasting 16 GPU dispatches and 8 PCIe readbacks.

### Exact Fix Instructions

1. **Replace `graph_radix_hist.wgsl` and `graph_radix_scatter.wgsl` with a Deterministic 3-Pass Stable Workgroup-Prefix Sort:**
   - **Pass A (`hist_wg`):** Each workgroup `wg_id = wid.x` of 256 threads counts its local 256-bin histogram in `var<workgroup> local_hist: array<atomic<u32>, 256>` and writes `wg_hist[d * num_wgs + wg_id] = count` (column-major by digit `d` so prefix sum is contiguous!).
   - **Pass B (`prefix_sum`):** A single workgroup of 256 threads scans `wg_hist[0 .. 256 * num_wgs]` exclusively on the GPU (zero CPU readbacks per digit!) to produce `wg_offsets[d * num_wgs + wg_id]`.
   - **Pass C (`stable_scatter`):** Within each workgroup of 256 threads, compute the exact stable rank of element `lid.x` among threads `< lid.x` in the same workgroup that share digit `d`.

2. **Replacement Shader for `graph_radix_scatter.wgsl`:**
```wgsl
// Target: c:\github\qualiaDB\crates\qualia-core-db\src\shaders\graph_radix_scatter.wgsl
// Stable intra-workgroup rank scatter for LSD radix sort.

struct Params {
    n: u32,
    shift: u32,
    num_wgs: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> keys_in: array<u32>;
@group(0) @binding(1) var<storage, read> idx_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> idx_out: array<u32>;
@group(0) @binding(4) var<storage, read> wg_offsets: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> wg_digits: array<u32, 256>;

fn digit_at(i: u32, shift: u32) -> u32 {
    let lo = keys_in[i * 2u];
    let hi = keys_in[i * 2u + 1u];
    if (shift < 32u) {
        return (lo >> shift) & 255u;
    }
    return (hi >> (shift - 32u)) & 255u;
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = gid.x;
    let t = lid.x;
    let valid = i < params.n;
    let d = select(0xFFFFFFFFu, digit_at(i, params.shift), valid);
    wg_digits[t] = d;
    workgroupBarrier();

    if (!valid) {
        return;
    }

    // Deterministic intra-workgroup rank: count threads u < t with identical digit d.
    var local_rank: u32 = 0u;
    for (var u: u32 = 0u; u < t; u = u + 1u) {
        local_rank = local_rank + select(0u, 1u, wg_digits[u] == d);
    }

    let base_offset = wg_offsets[d * params.num_wgs + wid.x];
    let pos = base_offset + local_rank;
    keys_out[pos * 2u] = keys_in[i * 2u];
    keys_out[pos * 2u + 1u] = keys_in[i * 2u + 1u];
    idx_out[pos] = idx_in[i];
}
```

3. **Update `c:\github\qualiaDB\crates\qualia-core-db\src\shaders\graph_radix_hist.wgsl`:**
```wgsl
// Target: c:\github\qualiaDB\crates\qualia-core-db\src\shaders\graph_radix_hist.wgsl
struct Params {
    n: u32,
    shift: u32,
    num_wgs: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> keys_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> wg_hist: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> local_hist: array<atomic<u32>, 256>;

fn digit_at(i: u32, shift: u32) -> u32 {
    let lo = keys_in[i * 2u];
    let hi = keys_in[i * 2u + 1u];
    if (shift < 32u) {
        return (lo >> shift) & 255u;
    }
    return (hi >> (shift - 32u)) & 255u;
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let t = lid.x;
    atomicStore(&local_hist[t], 0u);
    workgroupBarrier();

    let i = gid.x;
    if (i < params.n) {
        let d = digit_at(i, params.shift);
        atomicAdd(&local_hist[d], 1u);
    }
    workgroupBarrier();

    wg_hist[t * params.num_wgs + wid.x] = atomicLoad(&local_hist[t]);
}
```

4. **Verification Gate:**
   - Add a multi-workgroup adversarial test (`n = 1025`, reversed keys `(0..1025).rev()`, duplicate keys, high-bit keys `1u64 << 63 | i`) in `query/graph_accel/gpu.rs` asserting that `radix_sort_u64_indices_gpu(&keys)` returns `Some(indices)` Directly from the GPU (not CPU fallback!) and preserves stable relative order for equal keys.

---

## Bug 1.2: Fix False-Positive Metal Mega-Pass Capability Reporting

### Root Cause
In [`c:\github\qualiaDB\crates\qualia-core-db\src\inference\metal_lane.rs:110-119`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/inference/metal_lane.rs#L110-L119):
- `metal_mega_pass_available()` returns `true` whenever `#[cfg(all(target_os = "macos", feature = "metal-native"))]` is active, even though `try_metal_mega_pass()` at lines 97–105 unconditionally logs `"metal_mega_pass: not yet implemented on this build"` and returns `None`!
- Similarly, [`wgsl_forge/execute/wgpu.rs:1109-1130`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/wgsl_forge/execute/wgpu.rs#L1109-L1130) (`compile_msl`) returns an unimplemented error.

### Exact Fix Instructions
1. Edit [`c:\github\qualiaDB\crates\qualia-core-db\src\inference\metal_lane.rs`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/inference/metal_lane.rs#L108-L120):
   - Change `metal_mega_pass_available()` to return `false` until native `metal-rs` command-buffer execution is actually wired and passes parity tests.
   - Note that Apple Silicon GPU execution via `Etalon`'s Metal backend (`wgpu` WGSL $\rightarrow$ `naga::back::msl` with `simdgroup_matrix`) **is** implemented and active via `ResidentFullToken`; only the standalone `metal-rs` bypass in `metal_lane.rs` is a stub.
```rust
// Target: c:\github\qualiaDB\crates\qualia-core-db\src\inference\metal_lane.rs:110
pub fn metal_mega_pass_available() -> bool {
    // Truthfulness gate: return false until try_metal_mega_pass executes a verified pipeline.
    // Apple Silicon uses Etalon's Metal backend via ResidentFullToken (resident_decode.rs).
    false
}
```

---

## Bug 1.3: Remove WASM32 Synchronous 10,000-Iteration Spin-Poll in `map_read`

### Root Cause
In [`c:\github\qualiaDB\crates\qualia-core-db\src\query\graph_accel\gpu.rs:94-113`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/query/graph_accel/gpu.rs#L94-L113):
```rust
#[cfg(target_arch = "wasm32")]
{
    let mut mapped = false;
    for _ in 0..10_000 {
        let _ = device.poll(wgpu::PollType::Poll);
        match rx.try_recv() { ... }
    }
}
```
In a browser (`wasm32`), `Buffer::map_async` resolves via a JavaScript Promise on the microtask queue. A synchronous Rust `for _ in 0..10_000` loop never yields control to the browser event loop, so the callback can **never** fire during the loop! It simply burns 10,000 poll calls and returns `None`.

### Exact Fix Instructions
1. In synchronous functions (`radix_sort_u64_indices_gpu`, `sieve_eq_indices_gpu`), immediately return `None` on `#[cfg(target_arch = "wasm32")]` (allowing the caller to use the zero-latency synchronous CPU path without burning 10,000 wasted WebGPU polls).
2. Add an `async fn map_read_async(device: &wgpu::Device, staging: &wgpu::Buffer, size: u64) -> Option<Vec<u8>>` using `futures_channel::oneshot` (or `wasm_bindgen_futures`) for callers running inside the async Web Worker (`crates/etalon-wasm`).

---

## Bug 1.4: Fix DXC Subprocess Temp-File Race & Cache-Key Concatenation Bug

### Root Cause
1. **Hash Collision Vulnerability (`dxc_cache.rs:12-18`):**
   [`c:\github\qualiaDB\crates\qualia-core-db\src\wgsl_forge\emit\dxc_cache.rs:13-18`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/wgsl_forge/emit/dxc_cache.rs#L13-L18) computes:
   ```rust
   hasher.update(hlsl_source.as_bytes());
   hasher.update(entry_point.as_bytes());
   ```
   Without length delimiters, `("ab", "c")` and `("a", "bc")` hash identically, and compiler flags (`-T cs_6_0` vs `-T cs_6_8`, target env, ABI schema version) are omitted from the key.
2. **Unscoped Temp-File Race (`dxc.rs:34-78`):**
   [`c:\github\qualiaDB\crates\qualia-core-db\src\wgsl_forge\emit\dxc.rs:34-40`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/wgsl_forge/emit/dxc.rs#L34-L40) writes `qualia_dxc_in_{pid}_{entry_point}.hlsl` directly to `std::env::temp_dir()`. Two threads in the same process compiling the same `entry_point` overwrite each other's files and violate `Rule 0-B`.

### Exact Fix Instructions
1. **Fix `cache_key` in `dxc_cache.rs:12-18`:**
```rust
const DXC_CACHE_SCHEMA_V2: &[u8] = b"etalon-dxc-cache-v2";

fn cache_key(hlsl_source: &str, entry_point: &str, target_profile: &str, flags: &[&str]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DXC_CACHE_SCHEMA_V2);
    hasher.update(&(hlsl_source.len() as u64).to_le_bytes());
    hasher.update(hlsl_source.as_bytes());
    hasher.update(&(entry_point.len() as u64).to_le_bytes());
    hasher.update(entry_point.as_bytes());
    hasher.update(&(target_profile.len() as u64).to_le_bytes());
    hasher.update(target_profile.as_bytes());
    for flag in flags {
        hasher.update(&(flag.len() as u64).to_le_bytes());
        hasher.update(flag.as_bytes());
    }
    *hasher.finalize().as_bytes()
}
```
2. **Fix `dxc.rs:34-78`:**
   - Primary path: Use `naga` in-memory compilation (`naga::front::wgsl` $\rightarrow$ `naga::back::spv` / `naga::back::hlsl`).
   - If the external DXC fallback is invoked in cold tooling, allocate a thread-unique `tempfile::TempDir::with_prefix("etalon-dxc-")` that automatically cleans up on drop/panic (`Rule 0-B`).

---

## Bug 1.5: Unblock Vulkan Cooperative Matrix & Fix `M = 1` WMMA GEMV Waste

### Root Cause
1. In [`c:\github\qualiaDB\crates\qualia-core-db\src\wgsl_forge\emit\coopmat.rs:14-18`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/wgsl_forge/emit/coopmat.rs#L14-L18), `WGPU_COOPMAT_BLOCKED` is hardcoded to `true` because of wgpu 29.0.3 bug #9729 (`coopMultiplyAdd` returning all zeros due to missing `vulkanMemoryModelDeviceScope`). In `etalon` (`etalon-hal/src/vulkan/adapter.rs:628-645`), `.vulkan_memory_model_device_scope(needed)` is **already enabled**!
2. In [`c:\github\qualiaDB\crates\qualia-core-db\src\wgsl_forge\emit\cuda_c.rs:315-422`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/wgsl_forge/emit/cuda_c.rs#L315-L422), `Q4K_SOA_WMMA_GEMV_SRC` uses `16x16x16` `wmma::mma_sync` for `M = 1` by padding 15 rows of zeros (`cuda_c.rs:373-376`), wasting $93.75\%$ of Tensor Core compute.

### Exact Fix Instructions
1. In `wgsl_forge/emit/coopmat.rs:14-18`, set:
   ```rust
   pub const WGPU_API_VERSION: &str = "30.0.0-etalon";
   pub const WGPU_COOPMAT_BLOCKED: bool = false;
   ```
2. In `inference/cuda_lane` and `wgsl_forge/emit/cuda_c.rs`:
   - Route single-token decode (`M == 1..=4`) exclusively to `Q4K_SOA_GEMV_SRC` / `Q4K_SOA_DP4A_GEMV_SRC` (`__dp4a` / `dot4I8Packed` over `Q8_1` quantized activations).
   - Route batched execution (`M >= 8`, prefill and `verify_arena.rs` speculative verification) to `WMMA_GEMM_TILED_SRC` / Cooperative Matrix.

---

## Bug 1.6: Tighten Zero-Allocation Test Gate in `uniform_belt.rs`

### Root Cause
In [`c:\github\qualiaDB\crates\qualia-core-db\src\render\gpu\uniform_belt.rs:266-313`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/render/gpu/uniform_belt.rs#L266-L313), the test checks `allocs < 50` and silently passes if no GPU is found.

### Exact Fix Instructions
1. Separate the test into two explicit assertions:
   - **Pure Ring/Sub-allocation Logic Test (No GPU Driver Required):** Test `QualiaSlabAllocator::allocate` (`wgsl_forge/execute/memory.rs:152`) and `UniformBelt` offset advancing over 1,000 frames with `assert_eq!(measured_allocs, 0)` (strictly zero allocations, never skipped).
   - **Hardware Submission Test:** Measure `UniformBelt::write` + `recall` after 1 warmup frame; record driver-internal allocations separately from Etalon/application-owned allocations.
