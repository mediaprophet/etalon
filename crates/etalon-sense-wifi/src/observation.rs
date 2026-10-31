//! Caller-mapped conversion from cached BSS rows into shared observations.
//! BSSID bytes are carried as raw radio metadata and never become source or
//! identity identifiers. Cache timestamp and age remain unknown.

use etalon_sense_core::{
    evidence::EvidenceOrigin,
    observation::{HostPolicyDecision, MeasurementUncertainty, ObservationEnvelope, ObservationKind, QuantityUnitIds},
    policy::{PolicyOutcome, PolicyRequest},
    quality::{CalibrationRecord, CalibrationState, QualityRecord, SourceDescriptor},
    radio::{RadioIdentifierObservation, RadioObservationTime},
    ContractError,
};

use crate::{BssObservation, BssReadSummary, TimestampStatus};

pub const WIFI_METRIC_CAPACITY: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WifiMetric {
    RssiDbm,
    LinkQuality,
    CenterFrequencyKhz,
}

/// Host mapping for this cached query. `source` is independent of BSSID bytes;
/// semantic IDs are caller-supplied because this adapter does not define an
/// ontology or convert between reported fields and other units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WifiObservationMapping {
    /// Host-declared collecting source. This is not a BSSID or interface MAC.
    pub source: SourceDescriptor,
    /// Index order: RSSI dBm, driver link-quality ordinal, center frequency kHz.
    pub semantics: [QuantityUnitIds; WIFI_METRIC_CAPACITY],
    /// Stored as a host declaration; it is not a fresh authorization check.
    pub policy: Option<(PolicyRequest, PolicyOutcome)>,
}

/// One field from one cached BSS row. `bssid` is preserved exactly as a raw
/// link-layer address. It is not promoted to a person, source, or device DID.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WifiObservation {
    pub envelope: ObservationEnvelope,
    pub metric: WifiMetric,
    pub value: f64,
    pub bssid: [u8; 6],
    /// Typed peer BSSID metadata; it never replaces the host source descriptor.
    pub radio_identifier_observation: RadioIdentifierObservation,
    pub cache_timestamp: TimestampStatus,
    pub cache_age_known: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptError {
    OutputTooSmall { required: usize },
    InvalidMapping(ContractError),
    InvalidSummary,
}

/// Convert a cached BSS result into three typed field observations per row.
/// Output is cleared on failure. No timestamp, cache age, stream order, source
/// ID, or physical-unit mapping is inferred from the row or BSSID.
pub fn adapt_cached_bss(
    rows: &[BssObservation],
    summary: BssReadSummary,
    mapping: WifiObservationMapping,
    output: &mut [Option<WifiObservation>],
) -> Result<usize, AdaptError> {
    output.fill(None);
    if summary.written != rows.len()
        || summary.timestamp != TimestampStatus::Unknown
        || summary.cache_age_known
    {
        return Err(AdaptError::InvalidSummary);
    }
    let required = rows.len().checked_mul(WIFI_METRIC_CAPACITY)
        .ok_or(AdaptError::OutputTooSmall { required: usize::MAX })?;
    if output.len() < required {
        return Err(AdaptError::OutputTooSmall { required });
    }
    let policy = mapping.policy.map_or(HostPolicyDecision::NotEvaluated, |(request, outcome)| {
        HostPolicyDecision::Evaluated { request, outcome }
    });
    let calibration = CalibrationRecord::new(CalibrationState::Missing)
        .map_err(AdaptError::InvalidMapping)?;

    let mut out_index = 0;
    for row in rows {
        let values = [
            (WifiMetric::RssiDbm, row.rssi_dbm as f64),
            (WifiMetric::LinkQuality, row.link_quality as f64),
            (WifiMetric::CenterFrequencyKhz, row.center_frequency_khz as f64),
        ];
        for (metric_index, (metric, value)) in values.into_iter().enumerate() {
            let envelope = match ObservationEnvelope::new(
                ObservationKind::CachedQuery,
                mapping.source,
                None,
                None,
                None,
                None,
                mapping.semantics[metric_index],
                MeasurementUncertainty::Unknown,
                QualityRecord::Unknown,
                calibration,
                EvidenceOrigin::Unknown,
                None,
                policy,
            ) {
                Ok(envelope) => envelope,
                Err(error) => {
                    output.fill(None);
                    return Err(AdaptError::InvalidMapping(error));
                }
            };
            output[out_index] = Some(WifiObservation {
                envelope,
                metric,
                value,
                bssid: row.bssid,
                radio_identifier_observation: RadioIdentifierObservation {
                    identifier: row.radio_identifier(),
                    observed_at: RadioObservationTime::Unknown,
                },
                cache_timestamp: summary.timestamp,
                cache_age_known: summary.cache_age_known,
            });
            out_index += 1;
        }
    }
    Ok(out_index)
}

#[cfg(test)]
#[path = "observation_tests.rs"]
mod tests;
