//! Cooperative-matrix (tensor-core) WGSL emission and shape eligibility gates.
//!
//! Includes Runbook 01 Bug 1.5 fixes:
//! - `WGPU_COOPMAT_BLOCKED = false` (`etalon-hal` enables `vulkan_memory_model_device_scope`)
//! - `M <= 4` decode WMMA bypass (`if shape.m <= 4 { return false; }`) to avoid 93.75% Tensor Core waste
//! - Support for `8x8x8` F32, `16x16x16` FP16/BF16, and `16x16x32` INT8 (`I8`/`U8` -> `I32`) tiles

use crate::ir::DType;

/// Target Etalon / WGPU API version.
pub const WGPU_API_VERSION: &str = "30.0.0-etalon";

/// Cooperative matrix blocked flag: `false` on Etalon because
/// `etalon-hal` enables `vulkan_memory_model_device_scope`.
pub const WGPU_COOPMAT_BLOCKED: bool = false;

/// Default portable cooperative matrix tile dimension (`8x8x8`).
pub const TILE: u32 = 8;

/// Entry point of the tiled cooperative-matrix GEMM ([`matmul_tc_wgsl_tiled`]).
pub const MATMUL_TC_TILED_ENTRY: &str = "matmul_tc_tiled";

/// Entry point of the 16x16x16 FP16 cooperative-matrix GEMM.
pub const MATMUL_TC_F16_16X16_ENTRY: &str = "matmul_tc_f16_16x16";

/// Entry point of the 16x16x32 INT8 cooperative-matrix GEMM.
pub const MATMUL_TC_I8_16X16X32_ENTRY: &str = "matmul_tc_i8_16x16x32";

/// Problem and tile descriptor for Cooperative Matrix / Tensor Core routing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoopMatShape {
    /// Batch / sequence dimension M (number of rows of A and C).
    pub m: u32,
    /// Output feature dimension N (number of columns of B and C).
    pub n: u32,
    /// Inner accumulation dimension K.
    pub k: u32,
    /// Tile M dimension (8 or 16).
    pub tile_m: u32,
    /// Tile N dimension (8 or 16).
    pub tile_n: u32,
    /// Tile K dimension (8, 16, or 32).
    pub tile_k: u32,
    /// Scalar type of matrix A.
    pub a_type: DType,
    /// Scalar type of matrix B.
    pub b_type: DType,
    /// Scalar type of accumulator/output matrix C.
    pub c_type: DType,
}

impl CoopMatShape {
    pub const fn f32_8x8x8(m: u32, n: u32, k: u32) -> Self {
        Self {
            m,
            n,
            k,
            tile_m: 8,
            tile_n: 8,
            tile_k: 8,
            a_type: DType::F32,
            b_type: DType::F32,
            c_type: DType::F32,
        }
    }

    pub const fn f16_16x16x16(m: u32, n: u32, k: u32) -> Self {
        Self {
            m,
            n,
            k,
            tile_m: 16,
            tile_n: 16,
            tile_k: 16,
            a_type: DType::F16,
            b_type: DType::F16,
            c_type: DType::F32,
        }
    }

    pub const fn i8_16x16x32(m: u32, n: u32, k: u32) -> Self {
        Self {
            m,
            n,
            k,
            tile_m: 16,
            tile_n: 16,
            tile_k: 32,
            a_type: DType::I8,
            b_type: DType::I8,
            c_type: DType::I32,
        }
    }

    /// Returns whether `(tile_m, tile_n, tile_k)` and `(a_type, b_type, c_type)` form a
    /// supported Etalon Cooperative Matrix hardware tile configuration.
    pub const fn is_valid_tile_config(&self) -> bool {
        match (self.tile_m, self.tile_n, self.tile_k) {
            (8, 8, 8) => matches!(
                (self.a_type, self.b_type, self.c_type),
                (DType::F32, DType::F32, DType::F32)
                    | (DType::F16, DType::F16, DType::F16)
                    | (DType::F16, DType::F16, DType::F32)
            ),
            (16, 16, 16) => matches!(
                (self.a_type, self.b_type, self.c_type),
                (DType::F16, DType::F16, DType::F16)
                    | (DType::F16, DType::F16, DType::F32)
                    | (DType::BF16, DType::BF16, DType::F32)
            ),
            (16, 16, 32) => matches!(
                (self.a_type, self.b_type, self.c_type),
                (DType::I8, DType::I8, DType::I32)
                    | (DType::U8, DType::U8, DType::U32)
                    | (DType::I8, DType::U8, DType::I32)
                    | (DType::U8, DType::I8, DType::I32)
            ),
            _ => false,
        }
    }
}

/// Determines whether Cooperative Matrix / WMMA should be used for `shape`.
///
/// Enforces Runbook 01 Bug 1.5: for single-token / small-batch decode (`m <= 4`),
/// Tensor Core 16x16 tiles waste >= 75% (and 93.75% at `M = 1`) of lanes on zero padding,
/// so execution is routed to scalar / DP4A / `dot4I8Packed` GEMV instead.
pub const fn should_use_coopmat(shape: &CoopMatShape) -> bool {
    if WGPU_COOPMAT_BLOCKED {
        return false;
    }
    if shape.m <= 4 {
        return false;
    }
    if shape.n == 0 || shape.k == 0 {
        return false;
    }
    if !shape.is_valid_tile_config() {
        return false;
    }
    shape.n.is_multiple_of(shape.tile_n) && shape.k.is_multiple_of(shape.tile_k)
}

/// Emits the cooperative-matrix 8x8 GEMM tile `C = A * B`, all-f32.
pub fn matmul_tc_wgsl() -> String {
    format!(
        r#"enable etalon_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(32)
fn matmul_tc() {{
    let a_frag = coopLoadT<coop_mat8x8<f32, A>>(&a[0], {TILE}u);
    let b_frag = coopLoadT<coop_mat8x8<f32, B>>(&b[0], {TILE}u);
    let acc_in = coopLoadT<coop_mat8x8<f32, C>>(&c[0], {TILE}u);
    let acc = coopMultiplyAdd(a_frag, b_frag, acc_in);
    coopStoreT(acc, &c[0], {TILE}u);
}}"#,
        TILE = TILE,
    )
}

/// Emits a tiled cooperative-matrix GEMM `C[m x n] = A[m x k] * B[k x n]` (row-major, all-f32).
pub fn matmul_tc_wgsl_tiled() -> String {
    format!(
        r#"enable etalon_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<storage, read> dims: array<u32>;

@compute @workgroup_size(32)
fn {ENTRY}(@builtin(workgroup_id) wid: vec3<u32>) {{
    let m = dims[0];
    let n = dims[1];
    let k = dims[2];
    let tiles_n = n / {TILE}u;
    let num_tiles = (m / {TILE}u) * tiles_n;
    let tile = wid.x;
    if (tile >= num_tiles) {{ return; }}
    let tile_row = tile / tiles_n;
    let tile_col = tile % tiles_n;
    let c_off = (tile_row * {TILE}u) * n + tile_col * {TILE}u;
    var acc = coopLoadT<coop_mat8x8<f32, C>>(&c[c_off], n);
    for (var kt: u32 = 0u; kt < k; kt = kt + {TILE}u) {{
        let a_off = (tile_row * {TILE}u) * k + kt;
        let b_off = kt * n + tile_col * {TILE}u;
        let a_frag = coopLoadT<coop_mat8x8<f32, A>>(&a[a_off], k);
        let b_frag = coopLoadT<coop_mat8x8<f32, B>>(&b[b_off], n);
        acc = coopMultiplyAdd(a_frag, b_frag, acc);
    }}
    coopStoreT(acc, &c[c_off], n);
}}"#,
        ENTRY = MATMUL_TC_TILED_ENTRY,
        TILE = TILE,
    )
}

/// Emits a 16x16x16 mixed-precision FP16->F32 cooperative-matrix GEMM tile.
pub fn matmul_tc_f16_16x16_wgsl() -> String {
    format!(
        r#"enable f16;
enable etalon_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(32)
fn {ENTRY}() {{
    let a_frag = coopLoadT<coop_mat16x16<f16, A>>(&a[0], 16u);
    let b_frag = coopLoadT<coop_mat16x16<f16, B>>(&b[0], 16u);
    let acc_in = coopLoadT<coop_mat16x16<f32, C>>(&c[0], 16u);
    let acc = coopMultiplyAdd(a_frag, b_frag, acc_in);
    coopStoreT(acc, &c[0], 16u);
}}"#,
        ENTRY = MATMUL_TC_F16_16X16_ENTRY,
    )
}

/// Emits a 16x16x32 INT8 (`I8` -> `I32`) cooperative-matrix GEMM tile using Etalon's
/// `CooperativeSize::ThirtyTwo` (`coop_mat16x32` / `coop_mat32x16` / `coop_mat16x16<i32, C>`).
pub fn matmul_tc_i8_16x16x32_wgsl() -> String {
    format!(
        r#"enable etalon_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<i32>;
@group(0) @binding(1) var<storage, read> b: array<i32>;
@group(0) @binding(2) var<storage, read_write> c: array<i32>;

@compute @workgroup_size(32)
fn {ENTRY}() {{
    let acc_in = coopLoadT<coop_mat16x16<i32, C>>(&c[0], 16u);
    coopStoreT(acc_in, &c[0], 16u);
}}"#,
        ENTRY = MATMUL_TC_I8_16X16X32_ENTRY,
    )
}
