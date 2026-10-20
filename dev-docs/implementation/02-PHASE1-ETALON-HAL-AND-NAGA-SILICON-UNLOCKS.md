# Runbook 02: Phase 1 — `etalon-hal`, `naga`, and Dependency Unlocks

- **Prerequisite:** Read [`00-MASTER-IMPLEMENTATION-PLAYBOOK.md`](./00-MASTER-IMPLEMENTATION-PLAYBOOK.md)
- **Target Crates:** `etalon-types`, `naga`, `etalon-naga-bridge`, `etalon-hal`, `etalon-core`, `etalon`, `compat/wgpu`
- **Objective:** Unify the Cargo dependency graph between `Etalon` and `QualiaDB`, unlock `16x16x32` INT8 & `BF16` Cooperative Matrices in `naga` and `etalon-hal`, expose safe lease-backed Buffer Device Address (BDA) and Host-Mmap import capabilities, and add `ReusableComputeBundle` + per-queue completion tracking.

---

## Task 2.1: Resolve the Cargo Dependency & Version Mismatch (`P1`)

### Problem Statement
- [`c:\github\etalon\Cargo.toml:80`](file:///c:/github/etalon/Cargo.toml#L80) defines `version = "0.0.1"` for `etalon` and `naga`.
- [`c:\github\qualiaDB\Cargo.toml:97, 127`](file:///c:/github/qualiaDB/Cargo.toml#L97) requests `naga = "30"` and `wgpu = "30"`.
- A `compat/wgpu` crate at version `0.0.1` **cannot** satisfy `wgpu = "30"` under Cargo's SemVer rules, and leaving `naga = "30"` in `QualiaDB` while `etalon` uses `naga = "0.0.1"` causes duplicate `naga` crate versions in the dependency graph.

### Step-by-Step Instructions

1. **Create `c:\github\etalon\compat\wgpu\Cargo.toml` and `src/lib.rs`:**
   - Stamp `compat/wgpu` at `version = "30.0.0"` so any downstream crate or transitive dependency requesting `wgpu = "30"` resolves cleanly via `[patch.crates-io]`, **or** point `QualiaDB` directly at `etalon` and `naga` path dependencies.
   - Forward all feature flags from `compat/wgpu` to `etalon`:
```toml
# Target: c:\github\etalon\compat\wgpu\Cargo.toml
[package]
name = "wgpu"
version = "30.0.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[features]
default = ["etalon/default"]
std = ["etalon/std"]
serde = ["etalon/serde"]
wgsl = ["etalon/wgsl"]
vulkan = ["etalon/vulkan"]
dx12 = ["etalon/dx12"]
metal = ["etalon/metal"]
gles = ["etalon/gles"]
webgpu = ["etalon/webgpu"]
webgl = ["etalon/webgl"]
spirv = ["etalon/spirv"]
glsl = ["etalon/glsl"]
noop = ["etalon/noop"]
static-dxc = ["etalon/static-dxc"]
counters = ["etalon/counters"]
fragile-send-sync-non-atomic-wasm = ["etalon/fragile-send-sync-non-atomic-wasm"]

[dependencies]
etalon = { workspace = true }
```
```rust
// Target: c:\github\etalon\compat\wgpu\src\lib.rs
#![no_std]
pub use etalon::*;
```

2. **Align `QualiaDB` Root `Cargo.toml` (`c:\github\qualiaDB\Cargo.toml:97, 127`):**
   - Point both `wgpu` (or `etalon`) and `naga` to the local `c:\github\etalon` workspace so there is **strictly one `naga` and one `etalon` instance**:
```toml
# In c:\github\qualiaDB\Cargo.toml [workspace.dependencies]:
naga = { version = "0.0.1", path = "../etalon/naga", features = ["wgsl-in", "spv-out", "msl-out", "hlsl-out"] }
wgpu = { version = "30.0.0", path = "../etalon/compat/wgpu" }
etalon = { version = "0.0.1", path = "../etalon/etalon" }
```

3. **Verification Gate:**
```powershell
cd c:\github\qualiaDB
cargo tree -d -p qualia-core-db
# Must show ZERO duplicate versions of `naga`, `etalon`, or `wgpu`.
```

---

## Task 2.2: Unlock `16x16x32` (`I8`/`U8`) and `BF16` Cooperative Matrices Across the Stack

### Problem Statement
- [`naga/src/ir/mod.rs:551-554`](file:///c:/github/etalon/naga/src/ir/mod.rs#L551-L554) only defines `CooperativeSize::Eight = 8` and `Sixteen = 16`.
- [`etalon-hal/src/vulkan/adapter.rs:3554-3570`](file:///c:/github/etalon/etalon-hal/src/vulkan/adapter.rs#L3554-L3570) only maps `FLOAT16`, `FLOAT32`, `SINT32`, `UINT32` and sizes `8 | 16`, silently discarding `SINT8` / `UINT8` (`16x16x32` INT8 Tensor Core tiles) and `BFLOAT16_KHR`.
- [`etalon-types/src/features.rs:1107-1125`](file:///c:/github/etalon/etalon-types/src/features.rs#L1107-L1125) has outdated doc comments claiming only `8x8 f32` is supported.

### Step-by-Step Instructions Across Stack Boundaries

1. **`etalon-types` (`c:\github\etalon\etalon-types\src\lib.rs` & `src\features.rs`):**
   - Locate `pub enum CooperativeScalarType` in `etalon-types` and add `BF16`, `I8`, and `U8`:
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CooperativeScalarType {
    F16,
    BF16,
    F32,
    I8,
    U8,
    I32,
    U32,
}
```
   - Update the doc comment on `Features::EXPERIMENTAL_COOPERATIVE_MATRIX` in `etalon-types/src/features.rs:1107-1125` to accurately state that supported tile dimensions (`8x8`, `16x16`, `16x16x32`) and scalar types (`f16`, `bf16`, `f32`, `i8`, `u8`, `i32`, `u32`) are enumerated per adapter via `Adapter::cooperative_matrix_properties()`.

2. **`naga` (`c:\github\etalon\naga\src\ir\mod.rs:551-554`):**
   - Add `ThirtyTwo = 32` to `CooperativeSize`:
```rust
pub enum CooperativeSize {
    Eight = 8,
    Sixteen = 16,
    ThirtyTwo = 32,
}
```
   - Update exhaustive matches on `CooperativeSize` in `naga/src/front/wgsl`, `naga/src/valid/`, `naga/src/back/spv/`, and `naga/src/back/msl/`. (In `naga/src/back/msl/`, `ThirtyTwo` remains unsupported by native `simdgroup_matrix` hardware and should return `Error::UnsupportedCooperativeMatrixConfiguration`).

3. **`etalon-hal` Vulkan Backend (`c:\github\etalon\etalon-hal\src\vulkan\adapter.rs:3554-3570`):**
   - Update `map_vk_component_type` and `map_vk_cooperative_size`:
```rust
fn map_vk_component_type(ty: vk::ComponentTypeKHR) -> Option<wgt::CooperativeScalarType> {
    match ty {
        vk::ComponentTypeKHR::FLOAT16 => Some(wgt::CooperativeScalarType::F16),
        vk::ComponentTypeKHR::BFLOAT16_KHR => Some(wgt::CooperativeScalarType::BF16),
        vk::ComponentTypeKHR::FLOAT32 => Some(wgt::CooperativeScalarType::F32),
        vk::ComponentTypeKHR::SINT8 => Some(wgt::CooperativeScalarType::I8),
        vk::ComponentTypeKHR::UINT8 => Some(wgt::CooperativeScalarType::U8),
        vk::ComponentTypeKHR::SINT32 => Some(wgt::CooperativeScalarType::I32),
        vk::ComponentTypeKHR::UINT32 => Some(wgt::CooperativeScalarType::U32),
        _ => None,
    }
}

fn map_vk_cooperative_size(size: u32) -> Option<u32> {
    match size {
        8 | 16 | 32 => Some(size),
        _ => None,
    }
}
```

4. **Verification Gate:**
```powershell
cargo nextest run -p naga -p etalon-types -p etalon-hal
cargo xtask test -E 'binary(etalon-gpu) & test(cooperative_matrix)'
```

---

## Task 2.3: Un-Gate Buffer Device Address (BDA) & Host-Mmap Import Behind Safe Lease Contracts

### Problem Statement
- In [`etalon-hal/src/vulkan/adapter.rs:965-969, 1448-1452, 3080-3086`](file:///c:/github/etalon/etalon-hal/src/vulkan/adapter.rs#L1448-L1452), `khr::buffer_device_address::NAME` is only enabled when `EXPERIMENTAL_RAY_QUERY` or `EXPERIMENTAL_RAY_TRACING_PIPELINES` is requested.
- Compute workloads on GPUs without ray-tracing enabled (or compute-only queues) cannot query `vkGetBufferDeviceAddress`.

### Step-by-Step Instructions Across Stack Boundaries

1. **`etalon-types/src/features.rs`:**
   - Add two explicit native-only features:
     - `EXPERIMENTAL_BUFFER_DEVICE_ADDRESS`
     - `EXPERIMENTAL_EXTERNAL_MEMORY_HOST`
2. **`etalon-hal/src/vulkan/adapter.rs`:**
   - At line 969, check `caps.supports_extension(khr::buffer_device_address::NAME) || caps.device_api_version >= vk::API_VERSION_1_2` and set `F::EXPERIMENTAL_BUFFER_DEVICE_ADDRESS` when `self.buffer_device_address.is_some_and(|b| b.buffer_device_address == vk::TRUE)`.
   - At line 1448, enable `khr::buffer_device_address::NAME` whenever `requested_features.intersects(F::EXPERIMENTAL_RAY_QUERY | F::EXPERIMENTAL_RAY_TRACING_PIPELINES | F::EXPERIMENTAL_BUFFER_DEVICE_ADDRESS)`.
   - At line 3080, set `vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR::default().buffer_device_address(true)` when `EXPERIMENTAL_BUFFER_DEVICE_ADDRESS` is requested.
3. **`etalon-core` Safe Boundary Enforcement:**
   - Do **not** expose `EXPERIMENTAL_BUFFER_DEVICE_ADDRESS` or `EXPERIMENTAL_EXTERNAL_MEMORY_HOST` over `etalon-core-remote` / `etalon-core-remote-types` (browser IPC).
   - Ensure any buffer created for BDA includes `vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS` only when `Features::EXPERIMENTAL_BUFFER_DEVICE_ADDRESS` is enabled on the device.

4. **Verification Gate:**
```powershell
cargo check -p etalon -p etalon-core -p etalon-hal -p etalon-types
cargo check -p etalon --target wasm32-unknown-unknown --no-default-features --features std,webgpu,wgsl
```
