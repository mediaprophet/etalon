//! Deterministic WGSL generation, validation, certification, and tuning.
//!
//! WGSL Forge treats shader semantics and hardware scheduling as separate typed
//! inputs. Tuning is therefore allowed to change work distribution without
//! changing the mathematical operation being certified.

pub mod attention;
pub mod audio;
pub mod backend;
// W10: the forge's calibration/adaptation pipeline (corpus→capture→learn→certify→package). Native-
// only (drives the real inference stack for capture + the PPL oracle). The third produce-and-certify
// entry point beside kernel certification and GGUF→p64 transcode.
pub mod cache;
pub mod compute_passes;
pub mod dispatch;
pub mod execute;
pub mod graph_ops;
pub mod ir;
pub mod manifest;
pub mod oracle;
pub mod physics;
pub mod profiling;
pub mod roofline;
pub mod runtime;
pub mod schedule;
pub mod tune;
pub mod validate;

#[cfg(test)]
/// `?::test_gpu_available`.
pub(crate) fn test_gpu_available() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| execute::WgpuComputeContext::new(64 * 1024).is_ok())
}

#[cfg(all(test, feature = "cuda"))]
/// `?::test_cuda_available`.
pub(crate) fn test_cuda_available() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| execute::CudaComputeContext::new(64 * 1024).is_ok())
}

pub use attention::{
    AttentionSchedule, AttentionShader, AttentionShaderParams, AttentionShape,
    causal_attention_oracle, synthesize_causal_attention,
};
pub use backend::resolve_execution_backend;
pub use cache::ManifestCache;
pub use dispatch::{
    ComputeCaps, GEMM_GPU_THRESHOLD, caps, coopmat_usable, fft_f32, gemm_cpu_f64, gemm_f32,
    gemm_f32_tc, gemm_f32_tc_coopmat, gemm_f32_tc_reduced, gemm_f64, gemm_f64_df64, gemv_cpu_f64,
    gemv_f32, gemv_f64, pairwise_sq_dist_cpu_f64, pairwise_sq_dist_f64,
};
pub use emit::{GeneratedShader, TargetBackend, decode_spirv_words, emit_shader, matmul_tc_wgsl};
pub use ir::{
    BufferAccess, BufferElement, BufferSpec, BuiltinKernel, KernelSpec, Op, P64GpuWords64,
    ScalarType, SharedLen, SharedMemorySpec,
};
pub use manifest::{
    AdapterIdentity, CertificationManifest, HardwareProfile, TimingSource, TimingSummary,
    TuningManifest, ValidationLevel,
};

pub use oracle::{
    AffineParams, ComparisonReport, FfnParams, FftParams, GpuEvaluation, OracleCase,
    OracleTolerance, TopKParams, candidate_evaluation, certify_builtin, compare_f32, dft_cpu,
    evaluate_builtin, evaluate_ffn, evaluate_fft, evaluate_matmul_tc, evaluate_p64, evaluate_topk,
    ffn_cpu, ffn_tensors, fft_inputs, matmul_cpu, p64_project_cpu, p64_records, topk_cpu,
    topk_inputs,
};
#[cfg(feature = "cuda")]
pub use oracle::{evaluate_affine_cuda, evaluate_ffn_cuda, evaluate_topk_cuda};
pub use roofline::{RooflineBound, RooflineEstimate, roofline_for};
pub use runtime::ForgeRuntime;
pub use schedule::{AdapterConstraints, Schedule, ScheduleSpace};
pub use tune::{
    CandidateEvaluation, CandidateFailure, CandidateResult, TuningConfig, TuningResult, tune_with,
};
pub use validate::{ValidationReport, validate_native, validate_wgsl};

pub mod etalon_bridge {
    use super::{BuiltinKernel, ForgeError, Schedule, TargetBackend};

    pub use crate::{
        BuiltinKernel as EtalonBuiltinKernel, FORGE_CACHE_MAGIC as ETALON_FORGE_CACHE_MAGIC,
        FORGE_SCHEMA_VERSION as ETALON_FORGE_SCHEMA_VERSION,
        GeneratedShader as EtalonGeneratedShader, KernelSpec as EtalonKernelSpec,
        Schedule as EtalonSchedule, TargetBackend as EtalonTargetBackend,
        ValidationReport as EtalonValidationReport, certify_kernel as certify_etalon_kernel,
        emit_shader as emit_etalon_shader,
    };

    // ── TargetBackend conversions ───────────────────────────────────────

    impl From<TargetBackend> for crate::TargetBackend {
        fn from(target: TargetBackend) -> Self {
            match target {
                TargetBackend::Wgsl => Self::Wgsl,
                TargetBackend::Msl => Self::Msl,
                TargetBackend::Hlsl => Self::Hlsl,
                TargetBackend::Ptx => Self::Ptx,
                TargetBackend::CudaC => Self::CudaC,
                TargetBackend::Spirv => Self::Spirv,
            }
        }
    }

    impl From<crate::TargetBackend> for TargetBackend {
        fn from(target: crate::TargetBackend) -> Self {
            match target {
                crate::TargetBackend::Wgsl => Self::Wgsl,
                crate::TargetBackend::Msl => Self::Msl,
                crate::TargetBackend::Hlsl => Self::Hlsl,
                crate::TargetBackend::Ptx => Self::Ptx,
                crate::TargetBackend::CudaC | crate::TargetBackend::Cuda => Self::CudaC,
                crate::TargetBackend::Spirv => Self::Spirv,
            }
        }
    }

    pub fn to_etalon_target(target: TargetBackend) -> crate::TargetBackend {
        target.into()
    }

    pub fn from_etalon_target(target: crate::TargetBackend) -> TargetBackend {
        target.into()
    }

    // ── BuiltinKernel conversions ───────────────────────────────────────

    impl From<BuiltinKernel> for crate::BuiltinKernel {
        fn from(kernel: BuiltinKernel) -> Self {
            match kernel {
                BuiltinKernel::AffineF32 => Self::AffineF32,
                BuiltinKernel::FusedFfn => Self::FusedFfn,
                BuiltinKernel::P64Project => Self::P64Project,
                BuiltinKernel::TopK => Self::TopK,
                BuiltinKernel::RayProbe => Self::RayProbe,
                BuiltinKernel::TernaryGemv => Self::TernaryGemv,
                BuiltinKernel::Gemm => Self::Gemm,
                BuiltinKernel::Gemv => Self::Gemv,
                BuiltinKernel::Fft => Self::Fft,
                BuiltinKernel::IqPowerF32 => Self::IqPowerF32,
                BuiltinKernel::IqPowerI16 => Self::IqPowerI16,
            }
        }
    }

    impl From<crate::BuiltinKernel> for BuiltinKernel {
        fn from(kernel: crate::BuiltinKernel) -> Self {
            match kernel {
                crate::BuiltinKernel::AffineF32 => Self::AffineF32,
                crate::BuiltinKernel::FusedFfn => Self::FusedFfn,
                crate::BuiltinKernel::P64Project => Self::P64Project,
                crate::BuiltinKernel::TopK => Self::TopK,
                crate::BuiltinKernel::RayProbe => Self::RayProbe,
                crate::BuiltinKernel::TernaryGemv => Self::TernaryGemv,
                crate::BuiltinKernel::Gemm => Self::Gemm,
                crate::BuiltinKernel::Gemv => Self::Gemv,
                crate::BuiltinKernel::Fft => Self::Fft,
                crate::BuiltinKernel::IqPowerF32 => Self::IqPowerF32,
                crate::BuiltinKernel::IqPowerI16 => Self::IqPowerI16,
            }
        }
    }

    pub fn to_etalon_builtin(kernel: BuiltinKernel) -> crate::BuiltinKernel {
        kernel.into()
    }

    pub fn from_etalon_builtin(kernel: crate::BuiltinKernel) -> BuiltinKernel {
        kernel.into()
    }

    // ── Schedule conversions ────────────────────────────────────────────

    impl From<Schedule> for crate::Schedule {
        fn from(schedule: Schedule) -> Self {
            crate::Schedule::from_1d(
                schedule.workgroup_size,
                schedule.items_per_invocation,
                schedule.vector_width,
            )
        }
    }

    impl From<crate::Schedule> for Schedule {
        fn from(schedule: crate::Schedule) -> Self {
            Schedule {
                workgroup_size: schedule.effective_workgroup_size(),
                items_per_invocation: schedule.items_per_invocation,
                vector_width: schedule.vector_width,
            }
        }
    }

    pub fn to_etalon_schedule(schedule: Schedule) -> crate::Schedule {
        schedule.into()
    }

    pub fn from_etalon_schedule(schedule: crate::Schedule) -> Schedule {
        schedule.into()
    }

    // ── GeneratedShader conversions ─────────────────────────────────────

    impl From<crate::GeneratedShader> for super::GeneratedShader {
        fn from(s: crate::GeneratedShader) -> Self {
            Self {
                kernel_id: s.kernel_id,
                semantic_hash: s.semantic_hash,
                source_hash: s.source_hash,
                schedule: s.schedule.into(),
                source: s.source,
            }
        }
    }

    impl From<super::GeneratedShader> for crate::GeneratedShader {
        fn from(s: super::GeneratedShader) -> Self {
            Self {
                kernel_id: s.kernel_id,
                semantic_hash: s.semantic_hash,
                source_hash: s.source_hash,
                schedule: s.schedule.into(),
                source: s.source,
            }
        }
    }

    /// Compatibility wrapper that discards the report from [`generate_with_validation_via_etalon`].
    /// Successful emission alone does not establish device or numerical correctness.
    pub fn generate_and_certify_via_etalon(
        builtin: BuiltinKernel,
        schedule: Schedule,
        target: TargetBackend,
    ) -> Result<crate::GeneratedShader, ForgeError> {
        generate_with_validation_via_etalon(builtin, schedule, target).map(|(shader, _)| shader)
    }

    /// Emits a shader and retains Etalon's WGSL validation report unchanged.
    ///
    /// Non-WGSL targets return `None`: no native compiler validation is run here.
    /// WGSL returns `Some(report)`, whose `naga_validated` is false when Etalon's
    /// `naga-validate` feature is disabled. IR validation does not prove device
    /// compatibility or numerical correctness; safe shader creation still validates.
    pub fn generate_with_validation_via_etalon(
        builtin: BuiltinKernel,
        schedule: Schedule,
        target: TargetBackend,
    ) -> Result<(EtalonGeneratedShader, Option<EtalonValidationReport>), ForgeError> {
        let etalon_builtin: crate::BuiltinKernel = builtin.into();
        let etalon_schedule: crate::Schedule = schedule.into();
        let etalon_target: crate::TargetBackend = target.into();

        let spec = etalon_builtin.spec();
        let generated =
            crate::emit_shader(&spec, etalon_schedule, etalon_target).map_err(ForgeError::from)?;

        let validation = if matches!(target, TargetBackend::Wgsl) {
            Some(crate::certify_kernel(&generated.source).map_err(ForgeError::from)?)
        } else {
            None
        };

        Ok((generated, validation))
    }
}

pub use etalon_bridge::{
    ETALON_FORGE_CACHE_MAGIC, ETALON_FORGE_SCHEMA_VERSION, EtalonBuiltinKernel,
    EtalonGeneratedShader, EtalonKernelSpec, EtalonSchedule, EtalonTargetBackend,
    EtalonValidationReport, certify_etalon_kernel, emit_etalon_shader,
    generate_and_certify_via_etalon, generate_with_validation_via_etalon,
};

impl From<crate::ForgeError> for ForgeError {
    fn from(value: crate::ForgeError) -> Self {
        match value {
            crate::ForgeError::UnknownKernel(msg) => Self::UnknownKernel(msg),
            crate::ForgeError::InvalidKernel(msg) => Self::InvalidKernel(msg),
            crate::ForgeError::InvalidSchedule(msg) => Self::InvalidSchedule(msg),
            crate::ForgeError::Emission(msg) => Self::Emission(msg),
            crate::ForgeError::WgslParse(msg) => Self::WgslParse(msg),
            crate::ForgeError::WgslValidation(msg) => Self::WgslValidation(msg),
            crate::ForgeError::Validation(msg) => Self::WgslValidation(msg),
            crate::ForgeError::GpuUnavailable(msg) => Self::GpuUnavailable(msg),
            crate::ForgeError::GpuValidation(msg) => Self::GpuValidation(msg),
            crate::ForgeError::DeviceLost(msg) => Self::DeviceLost(msg),
            crate::ForgeError::OracleMismatch(msg) => Self::OracleMismatch(msg),
            crate::ForgeError::Serialization(msg) => Self::Serialization(msg),
            crate::ForgeError::Io(msg) => Self::Io(msg),
        }
    }
}

impl From<ForgeError> for crate::ForgeError {
    fn from(value: ForgeError) -> Self {
        match value {
            ForgeError::UnknownKernel(msg) => Self::UnknownKernel(msg),
            ForgeError::InvalidKernel(msg) => Self::InvalidKernel(msg),
            ForgeError::InvalidSchedule(msg) => Self::InvalidSchedule(msg),
            ForgeError::Emission(msg) => Self::Emission(msg),
            ForgeError::WgslParse(msg) => Self::WgslParse(msg),
            ForgeError::WgslValidation(msg) => Self::WgslValidation(msg),
            ForgeError::GpuUnavailable(msg) => Self::GpuUnavailable(msg),
            ForgeError::GpuValidation(msg) => Self::GpuValidation(msg),
            ForgeError::DeviceLost(msg) => Self::DeviceLost(msg),
            ForgeError::OracleMismatch(msg) => Self::OracleMismatch(msg),
            ForgeError::Serialization(msg) => Self::Serialization(msg),
            ForgeError::Io(msg) => Self::Io(msg),
        }
    }
}

pub const FORGE_SCHEMA_VERSION: u32 = 2;
pub const WGPU_API_VERSION: &str = "30.0.0";
pub const NAGA_API_VERSION: &str = "30.0.0";
/// `cudarc` crate API version the cross-backend (CUDA) oracle is built against.
/// Folded into the tuning/certification cache key (plan §8) so reuse is
/// invalidated when the CUDA toolchain surface changes.
pub const CUDARC_API_VERSION: &str = "0.19";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeError {
    UnknownKernel(String),
    InvalidKernel(String),
    InvalidSchedule(String),
    Emission(String),
    WgslParse(String),
    WgslValidation(String),
    GpuUnavailable(String),
    GpuValidation(String),
    /// The compute device was lost (driver reset, removal, or a fatal poll
    /// failure). Unified across backends so callers handle one variant rather
    /// than backend-specific panics (plan §7).
    DeviceLost(String),
    OracleMismatch(String),
    Serialization(String),
    Io(String),
}

impl core::fmt::Display for ForgeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (kind, message) = match self {
            Self::UnknownKernel(message) => ("unknown kernel", message),
            Self::InvalidKernel(message) => ("invalid kernel", message),
            Self::InvalidSchedule(message) => ("invalid schedule", message),
            Self::Emission(message) => ("WGSL emission", message),
            Self::WgslParse(message) => ("WGSL parse", message),
            Self::WgslValidation(message) => ("WGSL validation", message),
            Self::GpuUnavailable(message) => ("GPU unavailable", message),
            Self::GpuValidation(message) => ("GPU validation", message),
            Self::DeviceLost(message) => ("device lost", message),
            Self::OracleMismatch(message) => ("oracle mismatch", message),
            Self::Serialization(message) => ("serialization", message),
            Self::Io(message) => ("I/O", message),
        };
        write!(f, "{kind}: {message}")
    }
}

impl std::error::Error for ForgeError {}

impl From<std::io::Error> for ForgeError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<serde_json::Error> for ForgeError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

/// Generates a WGSL builtin.
pub fn generate_builtin(
    builtin: BuiltinKernel,
    schedule: Schedule,
    target: TargetBackend,
) -> Result<GeneratedShader, ForgeError> {
    let kernel = builtin.spec();
    schedule.validate(&kernel, &AdapterConstraints::portable())?;
    emit_shader(&kernel, schedule, target)
}

pub mod emit;

pub mod device_options;

#[cfg(not(target_arch = "wasm32"))]
pub mod substrate_gemm;

#[cfg(test)]
mod allocation_counter;
