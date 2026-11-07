//! `EtalonSignalProvider` — Zero-heap DSP and vision kernels backed by `etalon-signal`
//! (`signal_rms_peak`, `cfar_cell_average`, `biquad_df1_step`, `region_mean_u8`).

use etalon_signal::{df1_step_f32, mean_u8_region, rms_f32, ImageRegion};
use vibescript_abi::{
    BorrowedSliceView, CallerOutputSlice, CanonicalIri, ContractVersionAxes, DeterminismGuarantee,
    EffectSet, FuncDescriptor, NQuin48, ParamSpec, PrimitiveKind, ProviderDescriptor,
    ResourceEnvelope, ReversibilityKind, SemVer64, StableFuncId, SubNamespaceDescriptor,
    TargetProfileMask, VibeNamespaceProvider, VibeTrap,
};

use crate::helpers::{
    check_arity, check_len_exact, check_len_range, make_func_descriptor, ontology_triad,
    prepare_output, read_f32_finite, read_f32_vec3,
};

pub const SIGNAL_PROVIDER_IRI: &str = "https://qualiadb.org/ext/etalon/signal";
pub const SIGNAL_SHORT_ALIAS: &str = "signal";

static SIGNAL_ONTOLOGY: [NQuin48; 3] = ontology_triad(
    SIGNAL_PROVIDER_IRI,
    "https://qualiadb.org/ontology/etalon/signal#FiniteSampleBuffer",
    "https://qualiadb.org/ontology/etalon/signal#ParsevalBoundedEnergy",
    "https://qualiadb.org/ontology/etalon/signal#UncalibratedTransformInvariance",
);

const PARAMS_RMS_PEAK: [ParamSpec; 1] = [ParamSpec::slice("samples", PrimitiveKind::F32, 2, 64)];
const PARAMS_CFAR: [ParamSpec; 2] = [
    ParamSpec::slice("cells", PrimitiveKind::F32, 4, 32),
    ParamSpec::scalar("alpha", PrimitiveKind::F32),
];
const PARAMS_BIQUAD: [ParamSpec; 3] = [
    ParamSpec::slice("x_n", PrimitiveKind::F32, 3, 3),
    ParamSpec::slice("y_prev", PrimitiveKind::F32, 2, 2),
    ParamSpec::slice("coeffs_b0b1b2a1a2", PrimitiveKind::F32, 5, 5),
];
const PARAMS_REGION_U8: [ParamSpec; 1] = [ParamSpec::slice("pixels", PrimitiveKind::U8, 4, 64)];

const DSP_FUNCS: [FuncDescriptor; 3] = [
    make_func_descriptor(
        "signal_rms_peak",
        "https://qualiadb.org/ext/etalon/signal/dsp/signal_rms_peak",
        StableFuncId::new(0, 0),
        &PARAMS_RMS_PEAK,
        PrimitiveKind::F32,
        2,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(128, 8),
        "(F32[2..64])->F32[2]",
    ),
    make_func_descriptor(
        "cfar_cell_average",
        "https://qualiadb.org/ext/etalon/signal/dsp/cfar_cell_average",
        StableFuncId::new(0, 1),
        &PARAMS_CFAR,
        PrimitiveKind::F32,
        2,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(64, 8),
        "(F32[4..32],F32[1])->F32[2]",
    ),
    make_func_descriptor(
        "biquad_df1_step",
        "https://qualiadb.org/ext/etalon/signal/dsp/biquad_df1_step",
        StableFuncId::new(0, 2),
        &PARAMS_BIQUAD,
        PrimitiveKind::F32,
        1,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(32, 4),
        "(F32[3],F32[2],F32[5])->F32[1]",
    ),
];

const VISION_FUNCS: [FuncDescriptor; 1] = [make_func_descriptor(
    "region_mean_u8",
    "https://qualiadb.org/ext/etalon/signal/vision/region_mean_u8",
    StableFuncId::new(1, 0),
    &PARAMS_REGION_U8,
    PrimitiveKind::F32,
    2,
    EffectSet::PURE_COMPUTE,
    DeterminismGuarantee::DeterministicTolerance,
    ReversibilityKind::PureNoState,
    ResourceEnvelope::hot_zero_heap(64, 8),
    "(U8[4..64])->F32[2]",
)];

const SIGNAL_SUB_NAMESPACES: [SubNamespaceDescriptor; 2] = [
    SubNamespaceDescriptor {
        name: "dsp",
        sub_id: 0,
        functions: &DSP_FUNCS,
    },
    SubNamespaceDescriptor {
        name: "vision",
        sub_id: 1,
        functions: &VISION_FUNCS,
    },
];

const SIGNAL_DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    canonical_iri: CanonicalIri::new(SIGNAL_PROVIDER_IRI),
    short_alias: SIGNAL_SHORT_ALIAS,
    versions: ContractVersionAxes::v0_3_default(
        SemVer64::release(0, 3, 0),
        SemVer64::release(0, 3, 0),
    ),
    supported_targets: TargetProfileMask::ALL_CPU,
    sub_namespaces: &SIGNAL_SUB_NAMESPACES,
    max_ontology_quins: 8,
};

/// Reference oracle for `(0, 0) signal_rms_peak` calling `etalon_signal::rms_f32`.
#[inline]
pub fn oracle_signal_rms_peak(samples: &[f32]) -> Result<[f32; 2], VibeTrap> {
    if samples.len() < 2 || samples.len() > 64 {
        return Err(VibeTrap::SliceLengthMismatch);
    }
    if samples.iter().any(|x| !x.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    let rms = rms_f32(samples);
    let mut peak = 0.0f32;
    for &s in samples {
        let a = s.abs();
        if a > peak {
            peak = a;
        }
    }
    if !rms.is_finite() || !peak.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok([rms, peak])
}

/// Reference oracle for `(0, 1) cfar_cell_average`: `[noise_mean, threshold = alpha * noise_mean]`.
#[inline]
pub fn oracle_cfar_cell_average(cells: &[f32], alpha: f32) -> Result<[f32; 2], VibeTrap> {
    if cells.len() < 4 || cells.len() > 32 {
        return Err(VibeTrap::SliceLengthMismatch);
    }
    if !alpha.is_finite() || alpha <= 0.0 || cells.iter().any(|c| !c.is_finite() || *c < 0.0) {
        return Err(VibeTrap::NumericalDomainError);
    }
    let mut sum = 0.0f64;
    for &c in cells {
        sum += c as f64;
    }
    let noise_mean = (sum / (cells.len() as f64)) as f32;
    let threshold = alpha * noise_mean;
    if !noise_mean.is_finite() || !threshold.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok([noise_mean, threshold])
}

/// Reference oracle for `(0, 2) biquad_df1_step` calling `etalon_signal::df1_step_f32`.
#[inline]
pub fn oracle_biquad_df1_step(
    x_n: [f32; 3],
    y_prev: [f32; 2],
    coeffs: [f32; 5],
) -> Result<f32, VibeTrap> {
    if x_n.iter().any(|x| !x.is_finite())
        || y_prev.iter().any(|y| !y.is_finite())
        || coeffs.iter().any(|c| !c.is_finite())
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    let mut x1 = x_n[1];
    let mut x2 = x_n[2];
    let mut y1 = y_prev[0];
    let mut y2 = y_prev[1];
    let y = df1_step_f32(
        coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], &mut x1, &mut x2, &mut y1, &mut y2,
        x_n[0],
    );
    if !y.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok(y)
}

/// Reference oracle for `(1, 0) region_mean_u8` calling `etalon_signal::mean_u8_region`.
#[inline]
pub fn oracle_region_mean_u8(pixels: &[u8]) -> Result<[f32; 2], VibeTrap> {
    if pixels.len() < 4 || pixels.len() > 64 {
        return Err(VibeTrap::SliceLengthMismatch);
    }
    let n = pixels.len();
    let mean = mean_u8_region(
        pixels,
        n,
        1,
        n,
        ImageRegion {
            x: 0,
            y: 0,
            width: n,
            height: 1,
        },
    )
    .map_err(|_| VibeTrap::NumericalDomainError)? as f32;
    let mut min_px = u8::MAX;
    let mut max_px = u8::MIN;
    for &px in pixels {
        if px < min_px {
            min_px = px;
        }
        if px > max_px {
            max_px = px;
        }
    }
    let span = (max_px - min_px) as f32;
    Ok([mean, span])
}

/// Zero-heap Etalon DSP & Vision provider (`https://qualiadb.org/ext/etalon/signal`).
#[derive(Debug, Clone, Copy, Default)]
pub struct EtalonSignalProvider;

pub static SIGNAL_PROVIDER: EtalonSignalProvider = EtalonSignalProvider;

impl VibeNamespaceProvider for EtalonSignalProvider {
    #[inline]
    fn descriptor(&self) -> &'static ProviderDescriptor {
        &SIGNAL_DESCRIPTOR
    }

    #[inline]
    fn static_ontology_nquins(&self) -> &'static [NQuin48] {
        &SIGNAL_ONTOLOGY
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
                let n = check_len_range(&inputs[0], PrimitiveKind::F32, 2, 64)?;
                let mut samples = [0.0f32; 64];
                for i in 0..n {
                    samples[i] = read_f32_finite(&inputs[0], i)?;
                }
                prepare_output(output, PrimitiveKind::F32, 2)?;
                let res = oracle_signal_rms_peak(&samples[..n])?;
                output.push_f32_le(res[0])?;
                output.push_f32_le(res[1])?;
                Ok((n as u64) + 4)
            }
            (0, 1) => {
                check_arity(inputs, 2)?;
                let n = check_len_range(&inputs[0], PrimitiveKind::F32, 4, 32)?;
                let mut cells = [0.0f32; 32];
                for i in 0..n {
                    cells[i] = read_f32_finite(&inputs[0], i)?;
                }
                check_len_exact(&inputs[1], PrimitiveKind::F32, 1)?;
                let alpha = read_f32_finite(&inputs[1], 0)?;
                prepare_output(output, PrimitiveKind::F32, 2)?;
                let res = oracle_cfar_cell_average(&cells[..n], alpha)?;
                output.push_f32_le(res[0])?;
                output.push_f32_le(res[1])?;
                Ok((n as u64) + 4)
            }
            (0, 2) => {
                check_arity(inputs, 3)?;
                let x_n = read_f32_vec3(&inputs[0])?;
                check_len_exact(&inputs[1], PrimitiveKind::F32, 2)?;
                let y_prev = [
                    read_f32_finite(&inputs[1], 0)?,
                    read_f32_finite(&inputs[1], 1)?,
                ];
                check_len_exact(&inputs[2], PrimitiveKind::F32, 5)?;
                let coeffs = [
                    read_f32_finite(&inputs[2], 0)?,
                    read_f32_finite(&inputs[2], 1)?,
                    read_f32_finite(&inputs[2], 2)?,
                    read_f32_finite(&inputs[2], 3)?,
                    read_f32_finite(&inputs[2], 4)?,
                ];
                prepare_output(output, PrimitiveKind::F32, 1)?;
                let y = oracle_biquad_df1_step(x_n, y_prev, coeffs)?;
                output.push_f32_le(y)?;
                Ok(10)
            }
            (1, 0) => {
                check_arity(inputs, 1)?;
                let n = check_len_range(&inputs[0], PrimitiveKind::U8, 4, 64)?;
                let mut px = [0u8; 64];
                for i in 0..n {
                    px[i] = inputs[0].read_u8(i)?;
                }
                prepare_output(output, PrimitiveKind::F32, 2)?;
                let res = oracle_region_mean_u8(&px[..n])?;
                output.push_f32_le(res[0])?;
                output.push_f32_le(res[1])?;
                Ok((n as u64) + 4)
            }
            _ => Err(VibeTrap::FunctionNotFound),
        }
    }
}
