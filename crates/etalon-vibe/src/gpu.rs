//! `EtalonGpuComputeProvider` — Zero-heap GPU/tensor compute kernels (`affine_f32`,
//! `gemv4_f32`, `topk2_f32`, `rmsnorm_f32`, `rope_f32`).

use vibescript_abi::{
    BorrowedSliceView, CallerOutputSlice, CanonicalIri, ContractVersionAxes, DeterminismGuarantee,
    EffectSet, FuncDescriptor, NQuin48, ParamSpec, PrimitiveKind, ProviderDescriptor,
    ResourceEnvelope, ReversibilityKind, SemVer64, StableFuncId, SubNamespaceDescriptor,
    TargetProfileMask, VibeNamespaceProvider, VibeTrap,
};

use crate::helpers::{
    check_arity, check_len_exact, check_len_range, make_func_descriptor, ontology_triad,
    prepare_output, read_f32_finite, read_f32_vec4,
};

pub const GPU_PROVIDER_IRI: &str = "https://qualiadb.org/ext/etalon/gpu";
pub const GPU_SHORT_ALIAS: &str = "gpu";

static GPU_ONTOLOGY: [NQuin48; 3] = ontology_triad(
    GPU_PROVIDER_IRI,
    "https://qualiadb.org/ontology/etalon/gpu#FiniteTensorInputs",
    "https://qualiadb.org/ontology/etalon/gpu#BoundedFloatEnvelope",
    "https://qualiadb.org/ontology/etalon/gpu#ZeroHeapSimdInvariance",
);

const PARAMS_AFFINE: [ParamSpec; 2] = [
    ParamSpec::slice("input_vec", PrimitiveKind::F32, 4, 4),
    ParamSpec::slice("scale_bias", PrimitiveKind::F32, 2, 2),
];
const PARAMS_GEMV4: [ParamSpec; 2] = [
    ParamSpec::slice("mat4x4", PrimitiveKind::F32, 16, 16),
    ParamSpec::slice("vec4", PrimitiveKind::F32, 4, 4),
];
const PARAMS_TOPK2: [ParamSpec; 1] = [ParamSpec::slice("scores", PrimitiveKind::F32, 4, 16)];
const PARAMS_RMSNORM: [ParamSpec; 3] = [
    ParamSpec::slice("input_vec", PrimitiveKind::F32, 4, 4),
    ParamSpec::slice("weight_vec", PrimitiveKind::F32, 4, 4),
    ParamSpec::scalar("eps", PrimitiveKind::F32),
];
const PARAMS_ROPE: [ParamSpec; 2] = [
    ParamSpec::slice("vec4", PrimitiveKind::F32, 4, 4),
    ParamSpec::slice("theta_pos", PrimitiveKind::F32, 2, 2),
];

const COMPUTE_FUNCS: [FuncDescriptor; 5] = [
    make_func_descriptor(
        "affine_f32",
        "https://qualiadb.org/ext/etalon/gpu/compute/affine_f32",
        StableFuncId::new(0, 0),
        &PARAMS_AFFINE,
        PrimitiveKind::F32,
        4,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(32, 16),
        "(F32[4],F32[2])->F32[4]",
    ),
    make_func_descriptor(
        "gemv4_f32",
        "https://qualiadb.org/ext/etalon/gpu/compute/gemv4_f32",
        StableFuncId::new(0, 1),
        &PARAMS_GEMV4,
        PrimitiveKind::F32,
        4,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(64, 16),
        "(F32[16],F32[4])->F32[4]",
    ),
    make_func_descriptor(
        "topk2_f32",
        "https://qualiadb.org/ext/etalon/gpu/compute/topk2_f32",
        StableFuncId::new(0, 2),
        &PARAMS_TOPK2,
        PrimitiveKind::F32,
        2,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(64, 8),
        "(F32[4..16])->F32[2]",
    ),
    make_func_descriptor(
        "rmsnorm_f32",
        "https://qualiadb.org/ext/etalon/gpu/compute/rmsnorm_f32",
        StableFuncId::new(0, 3),
        &PARAMS_RMSNORM,
        PrimitiveKind::F32,
        4,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(64, 16),
        "(F32[4],F32[4],F32[1])->F32[4]",
    ),
    make_func_descriptor(
        "rope_f32",
        "https://qualiadb.org/ext/etalon/gpu/compute/rope_f32",
        StableFuncId::new(0, 4),
        &PARAMS_ROPE,
        PrimitiveKind::F32,
        4,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(64, 16),
        "(F32[4],F32[2])->F32[4]",
    ),
];

const GPU_SUB_NAMESPACES: [SubNamespaceDescriptor; 1] = [SubNamespaceDescriptor {
    name: "compute",
    sub_id: 0,
    functions: &COMPUTE_FUNCS,
}];

const GPU_DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    canonical_iri: CanonicalIri::new(GPU_PROVIDER_IRI),
    short_alias: GPU_SHORT_ALIAS,
    versions: ContractVersionAxes::v0_3_default(
        SemVer64::release(0, 3, 0),
        SemVer64::release(0, 3, 0),
    ),
    supported_targets: TargetProfileMask(
        TargetProfileMask::NATIVE_DESKTOP.0
            | TargetProfileMask::WASM_SANDBOX.0
            | TargetProfileMask::GPU_COMPUTE.0,
    ),
    sub_namespaces: &GPU_SUB_NAMESPACES,
    max_ontology_quins: 8,
};

/// Reference oracle for `(0, 0) affine_f32`: `y[i] = input[i] * scale + bias`.
#[inline]
pub fn oracle_affine_f32(input: [f32; 4], scale: f32, bias: f32) -> Result<[f32; 4], VibeTrap> {
    if !scale.is_finite() || !bias.is_finite() || input.iter().any(|x| !x.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    let out = [
        input[0] * scale + bias,
        input[1] * scale + bias,
        input[2] * scale + bias,
        input[3] * scale + bias,
    ];
    if out.iter().any(|y| !y.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok(out)
}

/// Reference oracle for `(0, 1) gemv4_f32`: row-major 4x4 matrix-vector multiply.
#[inline]
pub fn oracle_gemv4_f32(mat4x4: &[f32; 16], vec4: &[f32; 4]) -> Result<[f32; 4], VibeTrap> {
    if mat4x4.iter().any(|x| !x.is_finite()) || vec4.iter().any(|x| !x.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    let mut out = [0.0f32; 4];
    for r in 0..4 {
        let row = r * 4;
        let acc = (mat4x4[row] as f64) * (vec4[0] as f64)
            + (mat4x4[row + 1] as f64) * (vec4[1] as f64)
            + (mat4x4[row + 2] as f64) * (vec4[2] as f64)
            + (mat4x4[row + 3] as f64) * (vec4[3] as f64);
        let y = acc as f32;
        if !y.is_finite() {
            return Err(VibeTrap::NumericalDomainError);
        }
        out[r] = y;
    }
    Ok(out)
}

/// Reference oracle for `(0, 2) topk2_f32`: returns `[largest, second_largest]`.
#[inline]
pub fn oracle_topk2_f32(scores: &[f32]) -> Result<[f32; 2], VibeTrap> {
    if scores.len() < 4 || scores.len() > 16 {
        return Err(VibeTrap::SliceLengthMismatch);
    }
    if scores.iter().any(|x| !x.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    let mut best0 = f32::NEG_INFINITY;
    let mut best1 = f32::NEG_INFINITY;
    for &s in scores {
        if s >= best0 {
            best1 = best0;
            best0 = s;
        } else if s > best1 {
            best1 = s;
        }
    }
    Ok([best0, best1])
}

/// Reference oracle for `(0, 3) rmsnorm_f32`: `y[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]`.
#[inline]
pub fn oracle_rmsnorm_f32(
    input: [f32; 4],
    weight: [f32; 4],
    eps: f32,
) -> Result<[f32; 4], VibeTrap> {
    if !eps.is_finite()
        || eps <= 0.0
        || input.iter().any(|x| !x.is_finite())
        || weight.iter().any(|w| !w.is_finite())
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    let sum_sq = (input[0] as f64) * (input[0] as f64)
        + (input[1] as f64) * (input[1] as f64)
        + (input[2] as f64) * (input[2] as f64)
        + (input[3] as f64) * (input[3] as f64);
    let mean_sq = sum_sq * 0.25 + (eps as f64);
    let inv_rms = 1.0 / mean_sq.sqrt();
    let mut out = [0.0f32; 4];
    for i in 0..4 {
        let y = ((input[i] as f64) * inv_rms * (weight[i] as f64)) as f32;
        if !y.is_finite() {
            return Err(VibeTrap::NumericalDomainError);
        }
        out[i] = y;
    }
    Ok(out)
}

/// Reference oracle for `(0, 4) rope_f32`: 2D rotary position embedding on pairs `(0,1)` and `(2,3)`.
#[inline]
pub fn oracle_rope_f32(vec4: [f32; 4], base_theta: f32, pos: f32) -> Result<[f32; 4], VibeTrap> {
    if !base_theta.is_finite()
        || base_theta <= 0.0
        || !pos.is_finite()
        || vec4.iter().any(|x| !x.is_finite())
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    let a0 = pos as f64;
    let a1 = (pos as f64) / (base_theta as f64).sqrt();
    let (s0, c0) = a0.sin_cos();
    let (s1, c1) = a1.sin_cos();
    let x0 = vec4[0] as f64;
    let x1 = vec4[1] as f64;
    let x2 = vec4[2] as f64;
    let x3 = vec4[3] as f64;
    let out = [
        (x0 * c0 - x1 * s0) as f32,
        (x0 * s0 + x1 * c0) as f32,
        (x2 * c1 - x3 * s1) as f32,
        (x2 * s1 + x3 * c1) as f32,
    ];
    if out.iter().any(|y| !y.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok(out)
}

/// Zero-heap Etalon GPU/Tensor compute provider (`https://qualiadb.org/ext/etalon/gpu`).
#[derive(Debug, Clone, Copy, Default)]
pub struct EtalonGpuComputeProvider;

pub static GPU_PROVIDER: EtalonGpuComputeProvider = EtalonGpuComputeProvider;

impl VibeNamespaceProvider for EtalonGpuComputeProvider {
    #[inline]
    fn descriptor(&self) -> &'static ProviderDescriptor {
        &GPU_DESCRIPTOR
    }

    #[inline]
    fn static_ontology_nquins(&self) -> &'static [NQuin48] {
        &GPU_ONTOLOGY
    }

    fn call(
        &self,
        stable_id: StableFuncId,
        inputs: &[BorrowedSliceView<'_>],
        output: &mut CallerOutputSlice<'_>,
    ) -> Result<u64, VibeTrap> {
        match (stable_id.sub_namespace_id, stable_id.func_id) {
            (0, 0) => {
                check_arity(inputs, 2)?;
                let x = read_f32_vec4(&inputs[0])?;
                check_len_exact(&inputs[1], PrimitiveKind::F32, 2)?;
                let scale = read_f32_finite(&inputs[1], 0)?;
                let bias = read_f32_finite(&inputs[1], 1)?;
                prepare_output(output, PrimitiveKind::F32, 4)?;
                let y = oracle_affine_f32(x, scale, bias)?;
                for v in y {
                    output.push_f32_le(v)?;
                }
                Ok(8)
            }
            (0, 1) => {
                check_arity(inputs, 2)?;
                check_len_exact(&inputs[0], PrimitiveKind::F32, 16)?;
                let mut mat = [0.0f32; 16];
                for (i, slot) in mat.iter_mut().enumerate() {
                    *slot = read_f32_finite(&inputs[0], i)?;
                }
                let vec4 = read_f32_vec4(&inputs[1])?;
                prepare_output(output, PrimitiveKind::F32, 4)?;
                let y = oracle_gemv4_f32(&mat, &vec4)?;
                for v in y {
                    output.push_f32_le(v)?;
                }
                Ok(20)
            }
            (0, 2) => {
                check_arity(inputs, 1)?;
                let n = check_len_range(&inputs[0], PrimitiveKind::F32, 4, 16)?;
                let mut buf = [0.0f32; 16];
                for i in 0..n {
                    buf[i] = read_f32_finite(&inputs[0], i)?;
                }
                prepare_output(output, PrimitiveKind::F32, 2)?;
                let top2 = oracle_topk2_f32(&buf[..n])?;
                output.push_f32_le(top2[0])?;
                output.push_f32_le(top2[1])?;
                Ok(n as u64 + 4)
            }
            (0, 3) => {
                check_arity(inputs, 3)?;
                let x = read_f32_vec4(&inputs[0])?;
                let w = read_f32_vec4(&inputs[1])?;
                check_len_exact(&inputs[2], PrimitiveKind::F32, 1)?;
                let eps = read_f32_finite(&inputs[2], 0)?;
                prepare_output(output, PrimitiveKind::F32, 4)?;
                let y = oracle_rmsnorm_f32(x, w, eps)?;
                for v in y {
                    output.push_f32_le(v)?;
                }
                Ok(16)
            }
            (0, 4) => {
                check_arity(inputs, 2)?;
                let x = read_f32_vec4(&inputs[0])?;
                check_len_exact(&inputs[1], PrimitiveKind::F32, 2)?;
                let theta = read_f32_finite(&inputs[1], 0)?;
                let pos = read_f32_finite(&inputs[1], 1)?;
                prepare_output(output, PrimitiveKind::F32, 4)?;
                let y = oracle_rope_f32(x, theta, pos)?;
                for v in y {
                    output.push_f32_le(v)?;
                }
                Ok(18)
            }
            _ => Err(VibeTrap::FunctionNotFound),
        }
    }
}
