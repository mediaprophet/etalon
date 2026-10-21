//! Pure IR, multi-target shader emission, and hardware certification for Etalon.

pub mod autotune;
pub mod cache;
pub mod certifier;
pub mod certify;
pub mod emit;
pub mod ir;
pub mod ir_builtins;
pub mod schedule;

#[cfg(test)]
mod tests;

pub use autotune::{
    generate_search_candidates, tune_with, CandidateEvaluation, CandidateFailure, CandidateResult,
    TuningConfig, TuningResult,
};
pub use cache::{
    compute_cache_key, AdapterIdentity, CertificationManifest, HardwareProfile, ManifestCache,
    TimingSource, TimingSummary, TuningManifest, ValidationLevel, CACHE_SCHEMA_V2,
};
pub use certify::{
    certify_kernel, compare_f32, gemv_cpu, matmul_cpu, validate_native, validate_wgsl,
    ComparisonReport, OracleTolerance, ValidationReport,
};
pub use emit::{
    compile_hlsl_to_spirv, compile_hlsl_to_spirv_cached, dxc_cache_key, emit_cuda_c,
    emit_cuda_kernel, emit_hlsl, emit_hlsl_wave_kernel, emit_kernel, emit_msl, emit_msl_kernel,
    emit_shader, emit_wgsl, emit_wgsl_kernel, matmul_tc_f16_16x16_wgsl, matmul_tc_i8_16x16x32_wgsl,
    matmul_tc_wgsl, matmul_tc_wgsl_tiled, should_use_coopmat, CoopMatShape, GeneratedShader,
    TargetBackend, DXC_CACHE_SCHEMA_V2, MATMUL_TC_TILED_ENTRY, TILE, WGPU_COOPMAT_BLOCKED,
};
pub use ir::{
    BufferAccess, BufferElement, BufferSpec, DType, ForgeError, Intrinsic, IntrinsicClass,
    KernelSpec, Op, P64GpuWords64, ScalarType, SharedLen, SharedMemorySpec, SubgroupReduceOp,
    TensorViewDesc,
};
pub use ir_builtins::BuiltinKernel;
pub use schedule::{AdapterConstraints, Schedule, ScheduleSpace, TileConfig, WorkgroupSize};

pub const FORGE_SCHEMA_VERSION: &str = "0.0.1-etalon";
pub const FORGE_SCHEMA_VERSION_NUM: u32 = 1;
pub const WGPU_API_VERSION: &str = "30.0.0-etalon";
pub const NAGA_API_VERSION: &str = "30.0.0-etalon";
pub const CUDARC_API_VERSION: &str = "0.12";
pub const FORGE_CACHE_MAGIC: [u8; 8] = *b"ETALFORG";
