//! Adapter for an unverified native OS classification.
use etalon_sense_core::{
    evidence::EvidenceOrigin,
    observation::{
        HostPolicyDecision, MeasurementUncertainty, ObservationEnvelope, ObservationKind,
        QuantityUnitIds,
    },
    quality::{CalibrationRecord, CalibrationState, QualityRecord, SourceDescriptor},
};

use crate::{AdaptError, OsPresenceReading};

/// Source and semantic identifiers are declarations supplied by the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OsPresenceMapping {
    pub source: SourceDescriptor,
    pub semantics: QuantityUnitIds,
}

/// The OS code and time remain raw and are not identity or presence proof.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MappedOsPresence {
    pub envelope: ObservationEnvelope,
    pub presence_code: i32,
    pub windows_utc_ticks_100ns: i64,
}

/// Preserve an OS classification without inventing calibration, a clock
/// conversion, semantic identity, confidence, evidence or authorization.
pub fn adapt_os_presence(
    reading: OsPresenceReading,
    mapping: OsPresenceMapping,
) -> Result<MappedOsPresence, AdaptError> {
    if reading.source_id == 0 || reading.source_id != mapping.source.source_id() {
        return Err(AdaptError::SourceMismatch);
    }
    let calibration =
        CalibrationRecord::new(CalibrationState::Missing).map_err(AdaptError::Contract)?;
    let envelope = ObservationEnvelope::new(
        ObservationKind::CachedQuery,
        mapping.source,
        None,
        None,
        None,
        None,
        mapping.semantics,
        MeasurementUncertainty::Unknown,
        QualityRecord::Unknown,
        calibration,
        EvidenceOrigin::Unknown,
        None,
        HostPolicyDecision::NotEvaluated,
    )
    .map_err(AdaptError::Contract)?;
    Ok(MappedOsPresence {
        envelope,
        presence_code: reading.presence_code,
        windows_utc_ticks_100ns: reading.windows_utc_ticks_100ns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use etalon_sense_core::quality::SourceKind;

    fn mapping() -> OsPresenceMapping {
        OsPresenceMapping {
            source: SourceDescriptor::new(1, SourceKind::Live).unwrap(),
            semantics: QuantityUnitIds::new(None, None).unwrap(),
        }
    }

    #[test]
    fn unknown_os_code_and_time_remain_raw() {
        let reading = OsPresenceReading {
            source_id: 1,
            presence_code: -42,
            windows_utc_ticks_100ns: 123,
        };
        let mapped = adapt_os_presence(reading, mapping()).unwrap();
        assert_eq!(mapped.presence_code, -42);
        assert_eq!(mapped.windows_utc_ticks_100ns, 123);
        assert_eq!(mapped.envelope.timestamp(), None);
        assert_eq!(mapped.envelope.evidence(), EvidenceOrigin::Unknown);
        assert_eq!(mapped.envelope.policy(), HostPolicyDecision::NotEvaluated);
    }

    #[test]
    fn rejects_a_different_source() {
        let reading = OsPresenceReading {
            source_id: 2,
            presence_code: 0,
            windows_utc_ticks_100ns: 0,
        };
        assert_eq!(
            adapt_os_presence(reading, mapping()),
            Err(AdaptError::SourceMismatch)
        );
    }
}
