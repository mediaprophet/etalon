//! `EtalonModelProvider` — Zero-heap P64 weight container alignment, MoE top-2 router
//! softmax, and bounded residency admission checks backed by `etalon-model` and `etalon-resources`.

use etalon_model::p64::P64_CACHE_LINE_BYTES;
use etalon_resources::AdmissionLimits;
use vibescript_abi::{
    BorrowedSliceView, CallerOutputSlice, CanonicalIri, ContractVersionAxes, DeterminismGuarantee,
    EffectSet, FuncDescriptor, NQuin48, ParamSpec, PrimitiveKind, ProviderDescriptor,
    ResourceEnvelope, ReversibilityKind, SemVer64, StableFuncId, SubNamespaceDescriptor,
    TargetProfileMask, VibeNamespaceProvider, VibeTrap,
};

use crate::helpers::{
    check_arity, check_len_exact, check_len_range, make_func_descriptor, ontology_triad,
    prepare_output, read_f32_finite,
};

pub const MODEL_PROVIDER_IRI: &str = "https://qualiadb.org/ext/etalon/model";
pub const MODEL_SHORT_ALIAS: &str = "p64";

static MODEL_ONTOLOGY: [NQuin48; 3] = ontology_triad(
    MODEL_PROVIDER_IRI,
    "https://qualiadb.org/ontology/etalon/model#CacheLineAndPageAlignment",
    "https://qualiadb.org/ontology/etalon/model#NormalizedExpertRoutingWeights",
    "https://qualiadb.org/ontology/etalon/model#BoundedResidencyBudgetInvariance",
);

const PARAMS_P64_ALIGN: [ParamSpec; 1] = [ParamSpec::slice("offset_align", PrimitiveKind::U64, 2, 2)];
const PARAMS_MOE_TOP2: [ParamSpec; 1] = [ParamSpec::slice("RouterLogits", PrimitiveKind::F32, 4, 16)];
const PARAMS_RESIDENCY: [ParamSpec; 1] = [ParamSpec::slice("used_req_cap", PrimitiveKind::U64, 3, 3)];

const RUNTIME_FUNCS: [FuncDescriptor; 3] = [
    make_func_descriptor(
        "p64_align_offset",
        "https://qualiadb.org/ext/etalon/model/runtime/p64_align_offset",
        StableFuncId::new(0, 0),
        &PARAMS_P64_ALIGN,
        PrimitiveKind::U64,
        2,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicExact,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(24, 16),
        "(U64[2])->U64[2]",
    ),
    make_func_descriptor(
        "moe_top2_softmax",
        "https://qualiadb.org/ext/etalon/model/runtime/moe_top2_softmax",
        StableFuncId::new(0, 1),
        &PARAMS_MOE_TOP2,
        PrimitiveKind::F32,
        4,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(64, 16),
        "(F32[4..16])->F32[4]",
    ),
    make_func_descriptor(
        "residency_admit_check",
        "https://qualiadb.org/ext/etalon/model/runtime/residency_admit_check",
        StableFuncId::new(0, 2),
        &PARAMS_RESIDENCY,
        PrimitiveKind::U64,
        2,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicExact,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(24, 16),
        "(U64[3])->U64[2]",
    ),
];

const MODEL_SUB_NAMESPACES: [SubNamespaceDescriptor; 1] = [SubNamespaceDescriptor {
    name: "runtime",
    sub_id: 0,
    functions: &RUNTIME_FUNCS,
}];

const MODEL_DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    canonical_iri: CanonicalIri::new(MODEL_PROVIDER_IRI),
    short_alias: MODEL_SHORT_ALIAS,
    versions: ContractVersionAxes::v0_3_default(
        SemVer64::release(0, 3, 0),
        SemVer64::release(0, 3, 0),
    ),
    supported_targets: TargetProfileMask(
        TargetProfileMask::ALL_CPU.0 | TargetProfileMask::GPU_COMPUTE.0,
    ),
    sub_namespaces: &MODEL_SUB_NAMESPACES,
    max_ontology_quins: 8,
};

/// Reference oracle for `(0, 0) p64_align_offset`: requires power-of-two `align >= 64`.
#[inline]
pub fn oracle_p64_align_offset(offset: u64, align: u64) -> Result<[u64; 2], VibeTrap> {
    if align < (P64_CACHE_LINE_BYTES as u64) || !align.is_power_of_two() {
        return Err(VibeTrap::NumericalDomainError);
    }
    let mask = align - 1;
    let aligned = offset
        .checked_add(mask)
        .ok_or(VibeTrap::NumericalDomainError)?
        & !mask;
    let padding = aligned - offset;
    Ok([aligned, padding])
}

/// Reference oracle for `(0, 1) moe_top2_softmax`: `[idx0_f32, weight0, idx1_f32, weight1]`.
#[inline]
pub fn oracle_moe_top2_softmax(logits: &[f32]) -> Result<[f32; 4], VibeTrap> {
    if logits.len() < 4 || logits.len() > 16 {
        return Err(VibeTrap::SliceLengthMismatch);
    }
    if logits.iter().any(|x| !x.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    let mut idx0 = 0usize;
    let mut val0 = f32::NEG_INFINITY;
    let mut idx1 = 0usize;
    let mut val1 = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > val0 {
            idx1 = idx0;
            val1 = val0;
            idx0 = i;
            val0 = v;
        } else if v > val1 {
            idx1 = i;
            val1 = v;
        }
    }
    let e0 = 1.0f64;
    let e1 = ((val1 as f64) - (val0 as f64)).exp();
    let denom = e0 + e1;
    let w0 = (e0 / denom) as f32;
    let w1 = (e1 / denom) as f32;
    Ok([idx0 as f32, w0, idx1 as f32, w1])
}

/// Reference oracle for `(0, 2) residency_admit_check` using `etalon_resources::AdmissionLimits`.
#[inline]
pub fn oracle_residency_admit_check(used: u64, req: u64, cap: u64) -> Result<[u64; 2], VibeTrap> {
    if cap == 0 || used > cap {
        return Err(VibeTrap::NumericalDomainError);
    }
    let limits = AdmissionLimits {
        global_capacity: cap,
        global_recorded: used,
        partition_capacity: cap,
        partition_recorded: used,
    };
    let avail = limits.remaining_capacity();
    if req <= avail {
        Ok([1, avail - req])
    } else {
        Ok([0, avail])
    }
}

/// Zero-heap Etalon P64 model & residency provider (`https://qualiadb.org/ext/etalon/model`).
#[derive(Debug, Clone, Copy, Default)]
pub struct EtalonModelProvider;

pub static MODEL_PROVIDER: EtalonModelProvider = EtalonModelProvider;

impl VibeNamespaceProvider for EtalonModelProvider {
    #[inline]
    fn descriptor(&self) -> &'static ProviderDescriptor {
        &MODEL_DESCRIPTOR
    }

    #[inline]
    fn static_ontology_nquins(&self) -> &'static [NQuin48] {
        &MODEL_ONTOLOGY
    }

    fn call(
        &self,
        stable_id: StableFuncId,
        inputs: &[BorrowedSliceView<'_>],
        output: &mut CallerOutputSlice<'_>,
    ) -> Result<u64, VibeTrap> {
        match (stable_id.sub_namespace_id, stable_id.func_id) {
            (0, 0) => {
                check_arity(inputs, 1)?;
                check_len_exact(&inputs[0], PrimitiveKind::U64, 2)?;
                let offset = inputs[0].read_u64_le(0)?;
                let align = inputs[0].read_u64_le(1)?;
                prepare_output(output, PrimitiveKind::U64, 2)?;
                let res = oracle_p64_align_offset(offset, align)?;
                output.push_u64_le(res[0])?;
                output.push_u64_le(res[1])?;
                Ok(6)
            }
            (0, 1) => {
                check_arity(inputs, 1)?;
                let n = check_len_range(&inputs[0], PrimitiveKind::F32, 4, 16)?;
                let mut logits = [0.0f32; 16];
                for i in 0..n {
                    logits[i] = read_f32_finite(&inputs[0], i)?;
                }
                prepare_output(output, PrimitiveKind::F32, 4)?;
                let res = oracle_moe_top2_softmax(&logits[..n])?;
                for v in res {
                    output.push_f32_le(v)?;
                }
                Ok((n as u64) + 6)
            }
            (0, 2) => {
                check_arity(inputs, 1)?;
                check_len_exact(&inputs[0], PrimitiveKind::U64, 3)?;
                let used = inputs[0].read_u64_le(0)?;
                let req = inputs[0].read_u64_le(1)?;
                let cap = inputs[0].read_u64_le(2)?;
                prepare_output(output, PrimitiveKind::U64, 2)?;
                let res = oracle_residency_admit_check(used, req, cap)?;
                output.push_u64_le(res[0])?;
                output.push_u64_le(res[1])?;
                Ok(6)
            }
            _ => Err(VibeTrap::FunctionNotFound),
        }
    }
}
