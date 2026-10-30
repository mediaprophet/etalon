//! Target backend emission dispatch and code generation.

pub mod cache;
pub mod coopmat;
pub mod cuda_c;
pub mod cuda_kernels;
pub mod dxc;
pub mod dxc_cache;
pub mod hlsl_wave;
pub mod msl;
pub mod msl_kernels;
pub mod wgsl;
pub mod wgsl_ops;

use serde::{Deserialize, Serialize};

use crate::ir::{ForgeError, KernelSpec, Op};
use crate::schedule::Schedule;

pub use coopmat::{
    matmul_tc_f16_16x16_wgsl, matmul_tc_i8_16x16x32_wgsl, matmul_tc_wgsl, matmul_tc_wgsl_tiled,
    should_use_coopmat, CoopMatShape, MATMUL_TC_TILED_ENTRY, TILE, WGPU_COOPMAT_BLOCKED,
};
pub use cuda_c::{emit_cuda_c, emit_cuda_kernel};
pub use dxc::compile_hlsl_to_spirv;
pub use dxc_cache::{
    cache_key as dxc_cache_key, clear_dxc_cache, compile_hlsl_to_spirv_cached, dxc_cache_len,
    DXC_CACHE_SCHEMA_V2,
};
pub use hlsl_wave::{
    emit_fused_ffn_wave_hlsl, emit_gemv_wave_hlsl, emit_gemv_wavematrix_hlsl, emit_hlsl,
    emit_hlsl_wave_kernel, emit_topk_wave_hlsl,
};
pub use msl::{emit_msl, emit_msl_kernel};
pub use wgsl::{emit_wgsl, emit_wgsl_kernel};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetBackend {
    #[default]
    Wgsl,
    Msl,
    Hlsl,
    Ptx,
    CudaC,
    Cuda,
    Spirv,
}

impl std::str::FromStr for TargetBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "wgsl" => Ok(Self::Wgsl),
            "msl" => Ok(Self::Msl),
            "hlsl" => Ok(Self::Hlsl),
            "ptx" => Ok(Self::Ptx),
            "cuda" | "cuda-c" | "cuda_c" => Ok(Self::CudaC),
            "spirv" | "spir-v" | "spv" => Ok(Self::Spirv),
            _ => Err(format!("unknown target backend: {s}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedShader {
    pub kernel_id: String,
    pub semantic_hash: String,
    pub source_hash: String,
    pub schedule: Schedule,
    pub source: String,
}

/// Emits a complete shader module from a `KernelSpec` and `Schedule` for `target`.
pub fn emit_shader(
    kernel: &KernelSpec,
    schedule: Schedule,
    target: TargetBackend,
) -> Result<GeneratedShader, ForgeError> {
    match target {
        TargetBackend::Wgsl => emit_wgsl(kernel, schedule),
        TargetBackend::Msl => emit_msl(kernel, schedule),
        TargetBackend::Hlsl => emit_hlsl(kernel, schedule),
        TargetBackend::CudaC | TargetBackend::Cuda | TargetBackend::Ptx => {
            emit_cuda_c(kernel, schedule)
        }
        TargetBackend::Spirv => Err(ForgeError::Emission(
            "SPIR-V binary emission requires runtime naga spv-out pass".to_string(),
        )),
    }
}

/// Emits a standalone kernel string from a high-level `Op` and `Schedule`.
pub fn emit_kernel(op: &Op, sched: &Schedule, target: TargetBackend) -> Result<String, ForgeError> {
    match target {
        TargetBackend::Wgsl => wgsl::emit_wgsl_kernel(op, sched),
        TargetBackend::Msl => msl::emit_msl_kernel(op, sched),
        TargetBackend::Hlsl => hlsl_wave::emit_hlsl_wave_kernel(op, sched),
        TargetBackend::Cuda | TargetBackend::CudaC | TargetBackend::Ptx => {
            cuda_c::emit_cuda_kernel(op, sched)
        }
        TargetBackend::Spirv => Err(ForgeError::Emission(
            "Direct SPIR-V string emission not supported in emit_kernel".to_string(),
        )),
    }
}
