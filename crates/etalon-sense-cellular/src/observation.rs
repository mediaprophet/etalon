//! Caller-mapped conversion from validated cellular host telemetry into the
//! shared observation envelope. No cellular IDs or units are inferred here.

use etalon_sense_core::{
    evidence::EvidenceOrigin,
    observation::{HostPolicyDecision, MeasurementUncertainty, ObservationEnvelope, ObservationKind, QuantityUnitIds},
    policy::{PolicyOutcome, PolicyRequest},
    quality::{CalibrationRecord, CalibrationState, QualityRecord, SourceDescriptor},
    ContractError,
};

use crate::{CellularTelemetry, OrdinalBars, RadioGeneration, TelemetryTimestamp};

pub const CELLULAR_METRIC_CAPACITY: usize = 5;

/// Stable field labels; `SignalBars` remains ordinal, never a physical unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellularMetric {
    RsrpDbm,
    RsrqDb,
    SinrDb,
    RssiDbm,
    SignalBars,
}

/// Host-supplied mapping for one telemetry stream. `source` is the host's
/// mapping of the provider's source identifier; it is not derived from it.
/// `stream_generation` describes the host stream lifecycle, not 4G/5G radio
/// generation. Semantic IDs are opaque and deliberately supplied per field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellularObservationMapping {
    pub source: SourceDescriptor,
    pub stream_id: u64,
    pub stream_generation: Option<u32>,
    /// Index order: RSRP, RSRQ, SINR, RSSI, ordinal signal bars.
    pub semantics: [QuantityUnitIds; CELLULAR_METRIC_CAPACITY],
    /// This is stored as a host declaration, not revalidated by the adapter.
    pub policy: Option<(PolicyRequest, PolicyOutcome)>,
}

/// One host telemetry field with its source envelope. `provider_source_id`,
/// radio generation, and gap count remain raw host assertions. `value` is the
/// supplied numeric field; semantic meaning comes only from the mapping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellularObservation {
    pub envelope: ObservationEnvelope,
    pub metric: CellularMetric,
    pub value: f64,
    pub provider_source_id: u64,
    pub radio_generation: RadioGeneration,
    pub gap_before: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptError {
    OutputTooSmall { required: usize },
    InvalidMapping(ContractError),
}

/// Convert all present fields from one validated host record. The operation
/// writes only after checking capacity and clears output on any error. Caller
/// mapping must supply semantic IDs; bars should use an ordinal descriptor or
/// leave quantity/unit unknown. Unknown OS time remains `None`.
pub fn adapt_cellular(
    telemetry: CellularTelemetry,
    mapping: CellularObservationMapping,
    output: &mut [Option<CellularObservation>],
) -> Result<usize, AdaptError> {
    output.fill(None);
    let values = [
        telemetry.rsrp_dbm().map(|v| (CellularMetric::RsrpDbm, v)),
        telemetry.rsrq_db().map(|v| (CellularMetric::RsrqDb, v)),
        telemetry.sinr_db().map(|v| (CellularMetric::SinrDb, v)),
        telemetry.rssi_dbm().map(|v| (CellularMetric::RssiDbm, v)),
        telemetry.bars().map(|bars| (CellularMetric::SignalBars, bars.level as f64)),
    ];
    let required = values.iter().filter(|value| value.is_some()).count();
    if output.len() < required {
        return Err(AdaptError::OutputTooSmall { required });
    }
    if mapping.stream_id == 0 || mapping.stream_generation == Some(0) {
        return Err(AdaptError::InvalidMapping(ContractError::InvalidIdentifier));
    }
    let timestamp = match telemetry.timestamp() {
        TelemetryTimestamp::Unknown => None,
        TelemetryTimestamp::Clock(stamp) => Some(stamp),
    };
    let policy = mapping.policy.map_or(HostPolicyDecision::NotEvaluated, |(request, outcome)| {
        HostPolicyDecision::Evaluated { request, outcome }
    });
    let calibration = CalibrationRecord::new(CalibrationState::Missing)
        .map_err(AdaptError::InvalidMapping)?;

    let mut out_index = 0;
    for (mapping_index, value) in values.into_iter().enumerate() {
        if let Some((metric, value)) = value {
            let envelope = ObservationEnvelope::new(
                ObservationKind::HostDerived,
                mapping.source,
                Some(mapping.stream_id),
                mapping.stream_generation,
                Some(telemetry.sequence()),
                timestamp,
                mapping.semantics[mapping_index],
                MeasurementUncertainty::Unknown,
                QualityRecord::Unknown,
                calibration,
                EvidenceOrigin::Unknown,
                None,
                policy,
            ).map_err(|error| {
                output.fill(None);
                AdaptError::InvalidMapping(error)
            })?;
            output[out_index] = Some(CellularObservation {
                envelope,
                metric,
                value,
                provider_source_id: telemetry.source_id(),
                radio_generation: telemetry.generation(),
                gap_before: telemetry.gap_before(),
            });
            out_index += 1;
        }
    }
    Ok(out_index)
}

#[cfg(test)]
#[path = "observation_tests.rs"]
mod tests;
