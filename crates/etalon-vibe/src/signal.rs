//! `EtalonSignalProvider` — Zero-heap DSP and vision kernels backed by `etalon-signal`
//! (`signal_rms_peak`, `cfar_cell_average`, `biquad_df1_step`, `region_mean_u8`).

use etalon_signal::{
    df1_step_f32,
    hoa::{evaluate_hoa_basis_f64, spherical_to_unit_cartesian_f64, HoaNormalization, HoaOrder},
    mean_u8_region, rms_f32, woodworth_itd_seconds, DistanceModel, ImageRegion,
    DEFAULT_HEAD_RADIUS_M, DEFAULT_SPEED_OF_SOUND_M_S,
};
use vibescript_abi::{
    BorrowedSliceView, CallerOutputSlice, CanonicalIri, ContractVersionAxes, DeterminismGuarantee,
    EffectSet, FuncDescriptor, NQuin48, ParamSpec, PrimitiveKind, ProviderDescriptor,
    ResourceEnvelope, ReversibilityKind, SemVer64, StableFuncId, SubNamespaceDescriptor,
    TargetProfileMask, VibeNamespaceProvider, VibeTrap,
};

use crate::helpers::{
    check_arity, check_len_exact, check_len_range, make_func_descriptor, ontology_triad,
    prepare_output, read_f32_finite, read_f32_vec3, read_f32_vec4, read_f64_vec3,
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
const PARAMS_QAC_CONTAINER: [ParamSpec; 1] =
    [ParamSpec::slice("header_meta", PrimitiveKind::U32, 4, 4)];
const PARAMS_QAC_BINAURAL: [ParamSpec; 2] = [
    ParamSpec::slice("pose_deg_dist", PrimitiveKind::F32, 3, 3),
    ParamSpec::scalar("gain_linear", PrimitiveKind::F32),
];
const PARAMS_QAC_DIRECTIVITY: [ParamSpec; 2] = [
    ParamSpec::slice("angles_rad", PrimitiveKind::F32, 2, 2),
    ParamSpec::slice("weights_l0_l1", PrimitiveKind::F32, 4, 4),
];
const PARAMS_QAC_SONIFY: [ParamSpec; 1] =
    [ParamSpec::slice("emf_sample", PrimitiveKind::F64, 3, 3)];

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

const QAC_FUNCS: [FuncDescriptor; 4] = [
    make_func_descriptor(
        "decode_container_summary",
        "https://qualiadb.org/ext/etalon/signal/qac/decode_container_summary",
        StableFuncId::new(2, 0),
        &PARAMS_QAC_CONTAINER,
        PrimitiveKind::U64,
        2,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicExact,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(16, 16),
        "(U32[4])->U64[2]",
    ),
    make_func_descriptor(
        "render_binaural_frame",
        "https://qualiadb.org/ext/etalon/signal/qac/render_binaural_frame",
        StableFuncId::new(2, 1),
        &PARAMS_QAC_BINAURAL,
        PrimitiveKind::F32,
        3,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(48, 12),
        "(F32[3],F32[1])->F32[3]",
    ),
    make_func_descriptor(
        "evaluate_directivity_sh",
        "https://qualiadb.org/ext/etalon/signal/qac/evaluate_directivity_sh",
        StableFuncId::new(2, 2),
        &PARAMS_QAC_DIRECTIVITY,
        PrimitiveKind::F32,
        2,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(40, 8),
        "(F32[2],F32[4])->F32[2]",
    ),
    make_func_descriptor(
        "sonify_emf_envelope",
        "https://qualiadb.org/ext/etalon/signal/qac/sonify_emf_envelope",
        StableFuncId::new(2, 3),
        &PARAMS_QAC_SONIFY,
        PrimitiveKind::F64,
        2,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(32, 16),
        "(F64[3])->F64[2]",
    ),
];

const SIGNAL_SUB_NAMESPACES: [SubNamespaceDescriptor; 3] = [
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
    SubNamespaceDescriptor {
        name: "qac",
        sub_id: 2,
        functions: &QAC_FUNCS,
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

/// Reference oracle for `(2, 0) decode_container_summary`: `[duration_micros, active_mode_count]`.
#[inline]
pub fn oracle_decode_container_summary(header_meta: [u32; 4]) -> Result<[u64; 2], VibeTrap> {
    let [entry_count, sample_rate_hz, frame_samples, modes_mask] = header_meta;
    if entry_count == 0
        || sample_rate_hz == 0
        || frame_samples == 0
        || modes_mask == 0
        || modes_mask > 0x0F
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    let total_samples = (entry_count as u128) * (frame_samples as u128);
    let duration_micros_u128 = (total_samples * 1_000_000u128) / (sample_rate_hz as u128);
    let duration_micros =
        u64::try_from(duration_micros_u128).map_err(|_| VibeTrap::NumericalDomainError)?;
    let active_mode_count = modes_mask.count_ones() as u64;
    Ok([duration_micros, active_mode_count])
}

/// Reference oracle for `(2, 1) render_binaural_frame`: `[left_gain, right_gain, itd_sec]`.
#[inline]
pub fn oracle_render_binaural_frame(
    pose_deg_dist: [f32; 3],
    gain_linear: f32,
) -> Result<[f32; 3], VibeTrap> {
    if pose_deg_dist.iter().any(|x| !x.is_finite())
        || !gain_linear.is_finite()
        || gain_linear <= 0.0
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    let [azimuth_deg, elevation_deg, distance_m] = pose_deg_dist;
    if !(-180.0..=180.0).contains(&azimuth_deg)
        || !(-90.0..=90.0).contains(&elevation_deg)
        || distance_m <= 0.05
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    let az_rad = (azimuth_deg as f64).to_radians();
    let el_rad = (elevation_deg as f64).to_radians();
    let itd_base = woodworth_itd_seconds(
        az_rad as f32,
        DEFAULT_HEAD_RADIUS_M,
        DEFAULT_SPEED_OF_SOUND_M_S,
    )
    .map_err(|_| VibeTrap::NumericalDomainError)? as f64;
    let itd_sec = (itd_base * el_rad.cos()) as f32;
    let pan = (az_rad.sin() * el_rad.cos()).clamp(-1.0, 1.0);
    let pan_angle = (pan + 1.0) * core::f64::consts::FRAC_PI_4;
    let dist_gain = DistanceModel::Inverse {
        ref_distance: 1.0,
        max_distance: 10_000.0,
        rolloff_factor: 1.0,
    }
    .calculate_gain(distance_m) as f64;
    let net_gain = (gain_linear as f64) * dist_gain;
    let left_gain = (net_gain * pan_angle.cos()) as f32;
    let right_gain = (net_gain * pan_angle.sin()) as f32;
    if !left_gain.is_finite() || !right_gain.is_finite() || !itd_sec.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok([left_gain, right_gain, itd_sec])
}

/// Reference oracle for `(2, 2) evaluate_directivity_sh`: `[directivity_gain, l0_energy_ratio]`.
#[inline]
pub fn oracle_evaluate_directivity_sh(
    angles_rad: [f32; 2],
    weights_l0_l1: [f32; 4],
) -> Result<[f32; 2], VibeTrap> {
    if angles_rad.iter().any(|x| !x.is_finite()) || weights_l0_l1.iter().any(|w| !w.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    if angles_rad[1].abs() > core::f32::consts::FRAC_PI_2 + 1e-4 {
        return Err(VibeTrap::NumericalDomainError);
    }
    let el_clamped =
        (angles_rad[1] as f64).clamp(-core::f64::consts::FRAC_PI_2, core::f64::consts::FRAC_PI_2);
    let dir = spherical_to_unit_cartesian_f64(angles_rad[0] as f64, el_clamped)
        .map_err(|_| VibeTrap::NumericalDomainError)?;
    let mut basis = [0.0f64; 4];
    evaluate_hoa_basis_f64(HoaOrder::Order1, HoaNormalization::Sn3d, dir, &mut basis)
        .map_err(|_| VibeTrap::NumericalDomainError)?;
    let directivity_gain = ((weights_l0_l1[0] as f64) * basis[0]
        + (weights_l0_l1[1] as f64) * basis[1]
        + (weights_l0_l1[2] as f64) * basis[2]
        + (weights_l0_l1[3] as f64) * basis[3]) as f32;
    let w0 = weights_l0_l1[0] as f64;
    let w1 = weights_l0_l1[1] as f64;
    let w2 = weights_l0_l1[2] as f64;
    let w3 = weights_l0_l1[3] as f64;
    let e0 = w0 * w0;
    let total = e0 + (w1 * w1) + (w2 * w2) + (w3 * w3);
    let l0_energy_ratio = if total > 0.0 {
        (e0 / total) as f32
    } else {
        0.0
    };
    if !directivity_gain.is_finite() || !l0_energy_ratio.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok([directivity_gain, l0_energy_ratio])
}

/// Reference oracle for `(2, 3) sonify_emf_envelope`: `[carrier_hz, amplitude_norm]`.
#[inline]
pub fn oracle_sonify_emf_envelope(emf_sample: [f64; 3]) -> Result<[f64; 2], VibeTrap> {
    if emf_sample.iter().any(|x| !x.is_finite()) {
        return Err(VibeTrap::NumericalDomainError);
    }
    let [wavelength_nm, e_mag_v_per_m, spectral_sigma] = emf_sample;
    if !(380.0..=780.0).contains(&wavelength_nm)
        || e_mag_v_per_m < 0.0
        || !(0.0..=1.0).contains(&spectral_sigma)
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    // Matches QualiaDB `sigma_to_center_frequency_hz`: linear 400..700 nm -> 1760..110 Hz fold.
    let t = ((wavelength_nm - 400.0) / 300.0).clamp(0.0, 1.0);
    let carrier_hz = (1760.0 * (1.0 - t) + 110.0 * t).clamp(55.0, 8_000.0);
    let amp_field = e_mag_v_per_m / (1.0 + e_mag_v_per_m);
    let amplitude_norm = (amp_field * (0.5 + 0.5 * spectral_sigma)).clamp(0.0, 1.0);
    if !carrier_hz.is_finite() || !amplitude_norm.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok([carrier_hz, amplitude_norm])
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
            (2, 0) => {
                check_arity(inputs, 1)?;
                check_len_exact(&inputs[0], PrimitiveKind::U32, 4)?;
                let meta = [
                    inputs[0].read_u32_le(0)?,
                    inputs[0].read_u32_le(1)?,
                    inputs[0].read_u32_le(2)?,
                    inputs[0].read_u32_le(3)?,
                ];
                prepare_output(output, PrimitiveKind::U64, 2)?;
                let res = oracle_decode_container_summary(meta)?;
                output.push_u64_le(res[0])?;
                output.push_u64_le(res[1])?;
                Ok(6)
            }
            (2, 1) => {
                check_arity(inputs, 2)?;
                let pose = read_f32_vec3(&inputs[0])?;
                check_len_exact(&inputs[1], PrimitiveKind::F32, 1)?;
                let gain = read_f32_finite(&inputs[1], 0)?;
                prepare_output(output, PrimitiveKind::F32, 3)?;
                let res = oracle_render_binaural_frame(pose, gain)?;
                output.push_f32_le(res[0])?;
                output.push_f32_le(res[1])?;
                output.push_f32_le(res[2])?;
                Ok(12)
            }
            (2, 2) => {
                check_arity(inputs, 2)?;
                check_len_exact(&inputs[0], PrimitiveKind::F32, 2)?;
                let angles = [
                    read_f32_finite(&inputs[0], 0)?,
                    read_f32_finite(&inputs[0], 1)?,
                ];
                let weights = read_f32_vec4(&inputs[1])?;
                prepare_output(output, PrimitiveKind::F32, 2)?;
                let res = oracle_evaluate_directivity_sh(angles, weights)?;
                output.push_f32_le(res[0])?;
                output.push_f32_le(res[1])?;
                Ok(12)
            }
            (2, 3) => {
                check_arity(inputs, 1)?;
                let sample = read_f64_vec3(&inputs[0])?;
                prepare_output(output, PrimitiveKind::F64, 2)?;
                let res = oracle_sonify_emf_envelope(sample)?;
                output.push_f64_le(res[0])?;
                output.push_f64_le(res[1])?;
                Ok(8)
            }
            _ => Err(VibeTrap::FunctionNotFound),
        }
    }
}
