# Runbook 03: Phase 2 — Extracting `etalon-agent`, `etalon-wgsl-forge`, and `etalon-nquin`

- **Prerequisite:** Complete [`01-CRITICAL-BUG-FIXES-AND-TRUTHFULNESS-GATES.md`](./01-CRITICAL-BUG-FIXES-AND-TRUTHFULNESS-GATES.md) and [`02-PHASE1-ETALON-HAL-AND-NAGA-SILICON-UNLOCKS.md`](./02-PHASE1-ETALON-HAL-AND-NAGA-SILICON-UNLOCKS.md).
- **Target Crates:** `crates/etalon-agent`, `crates/etalon-wgsl-forge`, `crates/etalon-nquin`
- **Objective:** Extract the hardware stewardship, multi-target shader forge, and 48-byte semantic graph acceleration crates with **zero upward dependencies back into `QualiaDB`** and immediate vertical-slice verification.

---

## Task 3.1: Create `crates/etalon-agent` (Silicon Steward & Safe Resource Lease Layer)

### Module Layout (`< 500` lines per file, `Rule 0-B`)

```text
c:\github\etalon\crates\etalon-agent\
├── Cargo.toml
└── src\
    ├── lib.rs             # Public API re-exports only
    ├── capability.rs      # CapabilityState (Detected -> Compilable -> Prepared -> Verified)
    ├── lease.rs           # OwnedAllocation -> BufferRegion -> PreparedPlan -> SubmissionLease
    ├── slab.rs            # Migrated QualiaSlabAllocator (256B aligned, lap-counter protected)
    ├── compute_lane.rs    # Cold prepare() / hot submit() trait + PerQueueCompletionToken
    ├── hal_lane.rs        # Etalon-HAL (Vulkan/DX12/Metal/WebGPU) ComputeLane implementation
    ├── cuda_lane.rs       # Optional #[cfg(feature = "cuda")] cudarc / CUgraph implementation
    └── interop.rs         # Optional External Memory & Timeline Semaphore bridge
```

### Mandatory Implementation Contracts

1. **Truthful Capability State Machine (`capability.rs`):**

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CapabilityState {
    Unsupported,
    Detected,   // Hardware extension/feature bit present
    Compilable, // Shader/PTX/MSL module compiled without error
    Prepared,   // Pipeline + resources bound and ready
    Verified,   // Passed 1-step micro-oracle self-check
}
```

- Never route production workloads to a lane unless its state is `CapabilityState::Verified`.

2. **Safe Resource Ownership & Lease Hierarchy (`lease.rs`):**
   - Implement the four-stage ownership chain from Codex Review §3 Q2:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AllocationId {
    pub device_id: u64,
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessRights {
    ReadOnly,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferRegion {
    pub allocation: AllocationId,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub alignment: u32,
    pub access: AccessRights,
}

impl BufferRegion {
    pub fn validate(&self, total_capacity: u64, min_align: u32) -> Result<(), &'static str> {
        if self.alignment < min_align || !self.byte_offset.is_multiple_of(min_align as u64) {
            return Err("BufferRegion alignment violation");
        }
        let end = self.byte_offset.checked_add(self.byte_len).ok_or("BufferRegion offset overflow")?;
        if end > total_capacity {
            return Err("BufferRegion exceeds allocation bounds");
        }
        Ok(())
    }
}
```

- When an external host `mmap` (`Arc<Mmap>`) is imported, wrap the `Arc<Mmap>` inside `OwnedAllocation` and retain an `Arc` clone inside every in-flight `SubmissionLease` until the specific hardware queue reaches the submission's `PerQueueCompletionToken(QueueId, u64)`. This guarantees the OS file mapping can **never** be unmapped or truncated while GPU reads are in flight.

3. **Source Files to Migrate from `QualiaDB` into `crates/etalon-agent`:**
   - [`qualiaDB/crates/qualia-core-db/src/wgsl_forge/execute/memory.rs`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/wgsl_forge/execute/memory.rs) $\rightarrow$ `crates/etalon-agent/src/slab.rs`
   - [`qualiaDB/crates/qualia-core-db/src/wgsl_forge/execute/wgpu.rs`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/wgsl_forge/execute/wgpu.rs) $\rightarrow$ `crates/etalon-agent/src/hal_lane.rs`
   - [`qualiaDB/crates/qualia-core-db/src/wgsl_forge/execute/cuda.rs`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/wgsl_forge/execute/cuda.rs) $\rightarrow$ `crates/etalon-agent/src/cuda_lane.rs`
   - [`qualiaDB/crates/qualia-core-db/src/gpu_context.rs`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/gpu_context.rs) & [`host_topology.rs`](file:///c:/github/qualiaDB/crates/qualia-core-db/src/host_topology.rs) $\rightarrow$ `crates/etalon-agent/src/capability.rs`

---

## Task 3.2: Create `crates/etalon-wgsl-forge` (Pure IR, Multi-Target Emitter & Certifier)

### Module Layout (`< 500` lines per file)

```text
c:\github\etalon\crates\etalon-wgsl-forge\
├── Cargo.toml
└── src\
    ├── lib.rs
    ├── ir.rs              # Pure operation/kernel IR (zero SDK link dependencies!)
    ├── schedule.rs        # Tile sizes, workgroup geometry, vector width
    ├── emit\
    │   ├── mod.rs         # TargetBackend dispatch
    │   ├── wgsl.rs        # Portable WGSL + subgroup + coopmat emission
    │   ├── msl.rs         # Metal Shading Language + simdgroup_matrix emission
    │   ├── hlsl_wave.rs   # HLSL SM 6.0-6.9 WaveOps emission (no subprocess!)
    │   ├── cuda_c.rs      # CUDA C / PTX emission (DP4A for M<=4, WMMA for M>=8)
    │   └── cache.rs       # Length-delimited blake3 artifact + device-keyed tuning cache
    ├── certifier.rs       # Naga validation + CPU oracle numerical drift check
    └── autotune.rs        # Pareto-frontier schedule tuner (warmed & clock-aware)
```

### Mandatory Decoupling Rules

1. **No Database or `P64` File Coupling:**
   - Keep `P64` file header parsing and CRC-32C validation (`qualiaDB/crates/qualia-core-db/src/q42/p64_weight/layout.rs`) inside `QualiaDB`.
   - `etalon-wgsl-forge` only defines `P64GpuWords64` (`[u32; 16]`, `#[repr(C, align(16))]`) and validated `TensorViewDesc` (dtype, dimensions, strides, quantization block size).
2. **Bump Schema Version Constants (`wgsl_forge/mod.rs:84-90`):**
   - Update `FORGE_SCHEMA_VERSION` and `FORGE_CACHE_MAGIC` so stale pre-fork cache entries are never loaded.

---

## Task 3.3: Create `crates/etalon-nquin` (48-Byte Semantic Scan & Stable GPU Radix Sort)

### Module Layout

```text
c:\github\etalon\crates\etalon-nquin\
├── Cargo.toml
└── src\
    ├── lib.rs
    ├── layout.rs          # NQuinWire ([u64; 6], repr(C, align(16)), 48B compile-time assert)
    ├── sieve.rs           # Warp-Transposed uvec4 equality & predicate sieve + CPU oracle
    ├── radix.rs           # Fixed 3-pass Stable GPU LSD Radix Sort (from Runbook 01 Bug 1.1)
    └── shaders\
        ├── nquin_warp_scan.wgsl
        ├── graph_radix_hist.wgsl
        ├── graph_radix_prefix.wgsl
        └── graph_radix_scatter.wgsl
```

### Mandatory ABI & Verification Gates

1. **Compile-Time Layout Assertions (`layout.rs`):**

```rust
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NQuinWire {
    pub subject: u64,
    pub predicate: u64,
    pub object: u64,
    pub context: u64,
    pub metadata: u64,
    pub parity: u64,
}

const _: () = {
    assert!(core::mem::size_of::<NQuinWire>() == 48);
    assert!(core::mem::align_of::<NQuinWire>() == 16);
};
```

2. **Immediate Consumer & Oracle Gate:**
   - Wire `qualia-core-db::query::graph_accel` to delegate to `etalon-nquin`.
   - Run `cargo nextest run -p etalon-nquin` covering:
     - Empty (`0`), singleton (`1`), prime (`1021`), workgroup-boundary (`255`, `256`, `257`, `1024`, `1025`), duplicate keys, reversed keys, and high-bit (`1u64 << 63`) `u64` keys.
     - Verify **zero** per-digit host histogram readbacks during the 8-pass GPU radix sort.
