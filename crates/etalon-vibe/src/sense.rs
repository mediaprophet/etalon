//! `EtalonSenseProvider` — Zero-heap sensor telemetry, monotonic clock drift,
//! spectral bin frequency, and RSSI log-distance kernels backed by `etalon-sense-core`.

use etalon_sense_core::time::ClockStamp;
use vibescript_abi::{
    BorrowedSliceView, CallerOutputSlice, CanonicalIri, ContractVersionAxes, DeterminismGuarantee,
    EffectSet, FuncDescriptor, NQuin48, ParamSpec, PrimitiveKind, ProviderDescriptor,
    ResourceEnvelope, ReversibilityKind, SemVer64, StableFuncId, SubNamespaceDescriptor,
    TargetProfileMask, VibeNamespaceProvider, VibeTrap,
};

use crate::helpers::{
    check_arity, check_len_exact, make_func_descriptor, ontology_triad, prepare_output,
    read_f64_finite, read_f64_vec3,
};

pub const SENSE_PROVIDER_IRI: &str = "https://qualiadb.org/ext/etalon/sense";
pub const SENSE_SHORT_ALIAS: &str = "sense";

static SENSE_ONTOLOGY: [NQuin48; 3] = ontology_triad(
    SENSE_PROVIDER_IRI,
    "https://qualiadb.org/ontology/etalon/sense#ValidatedMonotonicClockAndBand",
    "https://qualiadb.org/ontology/etalon/sense#BoundedUncertaintyInterval",
    "https://qualiadb.org/ontology/etalon/sense#ExplicitProvenanceInvariance",
);

const PARAMS_CLOCK_DRIFT: [ParamSpec; 2] = [
    ParamSpec::slice("stamps_ns", PrimitiveKind::U64, 2, 2),
    ParamSpec::scalar("max_uncertainty_ns", PrimitiveKind::U64),
];
const PARAMS_SPECTRUM_BIN: [ParamSpec; 2] = [
    ParamSpec::slice("center_hz_bw_hz", PrimitiveKind::F64, 2, 2),
    ParamSpec::slice("bin_idx_total", PrimitiveKind::U32, 2, 2),
];
const PARAMS_RSSI_DIST: [ParamSpec; 1] =
    [ParamSpec::slice("rssi_dbm_tx_dbm_n", PrimitiveKind::F64, 3, 3)];

const TELEMETRY_FUNCS: [FuncDescriptor; 3] = [
    make_func_descriptor(
        "clock_stamp_drift_ns",
        "https://qualiadb.org/ext/etalon/sense/telemetry/clock_stamp_drift_ns",
        StableFuncId::new(0, 0),
        &PARAMS_CLOCK_DRIFT,
        PrimitiveKind::U64,
        2,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicExact,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(32, 16),
        "(U64[2],U64[1])->U64[2]",
    ),
    make_func_descriptor(
        "spectrum_bin_hz",
        "https://qualiadb.org/ext/etalon/sense/telemetry/spectrum_bin_hz",
        StableFuncId::new(0, 1),
        &PARAMS_SPECTRUM_BIN,
        PrimitiveKind::F64,
        1,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(32, 8),
        "(F64[2],U32[2])->F64[1]",
    ),
    make_func_descriptor(
        "rssi_log_distance_m",
        "https://qualiadb.org/ext/etalon/sense/telemetry/rssi_log_distance_m",
        StableFuncId::new(0, 2),
        &PARAMS_RSSI_DIST,
        PrimitiveKind::F64,
        1,
        EffectSet::PURE_COMPUTE,
        DeterminismGuarantee::DeterministicTolerance,
        ReversibilityKind::PureNoState,
        ResourceEnvelope::hot_zero_heap(48, 8),
        "(F64[3])->F64[1]",
    ),
];

const SENSE_SUB_NAMESPACES: [SubNamespaceDescriptor; 1] = [SubNamespaceDescriptor {
    name: "telemetry",
    sub_id: 0,
    functions: &TELEMETRY_FUNCS,
}];

const SENSE_DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    canonical_iri: CanonicalIri::new(SENSE_PROVIDER_IRI),
    short_alias: SENSE_SHORT_ALIAS,
    versions: ContractVersionAxes::v0_3_default(
        SemVer64::release(0, 3, 0),
        SemVer64::release(0, 3, 0),
    ),
    supported_targets: TargetProfileMask::ALL_CPU,
    sub_namespaces: &SENSE_SUB_NAMESPACES,
    max_ontology_quins: 8,
};

/// Reference oracle for `(0, 0) clock_stamp_drift_ns` using `etalon_sense_core::time::ClockStamp`.
#[inline]
pub fn oracle_clock_stamp_drift_ns(
    t0_ns: u64,
    t1_ns: u64,
    max_uncertainty_ns: u64,
) -> Result<[u64; 2], VibeTrap> {
    let s0 = ClockStamp::new(1, 0, t0_ns, max_uncertainty_ns)
        .map_err(|_| VibeTrap::NumericalDomainError)?;
    let s1 = ClockStamp::new(1, 0, t1_ns, max_uncertainty_ns)
        .map_err(|_| VibeTrap::NumericalDomainError)?;
    let abs_delta = s0.ticks_ns().abs_diff(s1.ticks_ns());
    let within_bound = u64::from(abs_delta <= max_uncertainty_ns);
    Ok([abs_delta, within_bound])
}

/// Reference oracle for `(0, 1) spectrum_bin_hz`.
#[inline]
pub fn oracle_spectrum_bin_hz(
    center_hz: f64,
    bw_hz: f64,
    bin_idx: u32,
    total_bins: u32,
) -> Result<f64, VibeTrap> {
    if !center_hz.is_finite()
        || !bw_hz.is_finite()
        || center_hz <= 0.0
        || bw_hz <= 0.0
        || total_bins == 0
        || bin_idx >= total_bins
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    let start_hz = center_hz - 0.5 * bw_hz;
    if start_hz < 0.0 {
        return Err(VibeTrap::NumericalDomainError);
    }
    let df = bw_hz / (total_bins as f64);
    let bin_center = start_hz + ((bin_idx as f64) + 0.5) * df;
    if !bin_center.is_finite() {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok(bin_center)
}

/// Reference oracle for `(0, 2) rssi_log_distance_m`: `10^((tx_dbm - rssi_dbm) / (10 * n))`.
#[inline]
pub fn oracle_rssi_log_distance_m(
    rssi_dbm: f64,
    tx_power_1m_dbm: f64,
    path_loss_exp: f64,
) -> Result<f64, VibeTrap> {
    if !rssi_dbm.is_finite()
        || !tx_power_1m_dbm.is_finite()
        || !path_loss_exp.is_finite()
        || path_loss_exp <= 0.0
    {
        return Err(VibeTrap::NumericalDomainError);
    }
    let exponent = (tx_power_1m_dbm - rssi_dbm) / (10.0 * path_loss_exp);
    let dist_m = 10.0f64.powf(exponent);
    if !dist_m.is_finite() || dist_m <= 0.0 {
        return Err(VibeTrap::NumericalDomainError);
    }
    Ok(dist_m)
}

/// Zero-heap Etalon sensor telemetry provider (`https://qualiadb.org/ext/etalon/sense`).
#[derive(Debug, Clone, Copy, Default)]
pub struct EtalonSenseProvider;

pub static SENSE_PROVIDER: EtalonSenseProvider = EtalonSenseProvider;

impl VibeNamespaceProvider for EtalonSenseProvider {
    #[inline]
    fn descriptor(&self) -> &'static ProviderDescriptor {
        &SENSE_DESCRIPTOR
    }

    #[inline]
    fn static_ontology_nquins(&self) -> &'static [NQuin48] {
        &SENSE_ONTOLOGY
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
                check_len_exact(&inputs[0], PrimitiveKind::U64, 2)?;
                check_len_exact(&inputs[1], PrimitiveKind::U64, 1)?;
                let t0 = inputs[0].read_u64_le(0)?;
                let t1 = inputs[0].read_u64_le(1)?;
                let max_unc = inputs[1].read_u64_le(0)?;
                prepare_output(output, PrimitiveKind::U64, 2)?;
                let res = oracle_clock_stamp_drift_ns(t0, t1, max_unc)?;
                output.push_u64_le(res[0])?;
                output.push_u64_le(res[1])?;
                Ok(6)
            }
            (0, 1) => {
                check_arity(inputs, 2)?;
                check_len_exact(&inputs[0], PrimitiveKind::F64, 2)?;
                check_len_exact(&inputs[1], PrimitiveKind::U32, 2)?;
                let center_hz = read_f64_finite(&inputs[0], 0)?;
                let bw_hz = read_f64_finite(&inputs[0], 1)?;
                let bin_idx = inputs[1].read_u32_le(0)?;
                let total_bins = inputs[1].read_u32_le(1)?;
                prepare_output(output, PrimitiveKind::F64, 1)?;
                let hz = oracle_spectrum_bin_hz(center_hz, bw_hz, bin_idx, total_bins)?;
                output.push_f64_le(hz)?;
                Ok(8)
            }
            (0, 2) => {
                check_arity(inputs, 1)?;
                let p = read_f64_vec3(&inputs[0])?;
                prepare_output(output, PrimitiveKind::F64, 1)?;
                let dist = oracle_rssi_log_distance_m(p[0], p[1], p[2])?;
                output.push_f64_le(dist)?;
                Ok(10)
            }
            _ => Err(VibeTrap::FunctionNotFound),
        }
    }
}
