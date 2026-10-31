use super::*;
use etalon_sense_core::quality::SourceKind;
use etalon_sense_core::radio::{RadioAddress48, RadioIdentifier, RadioObservationTime};

fn mapping() -> WifiObservationMapping {
    let unknown = QuantityUnitIds::new(None, None).unwrap();
    WifiObservationMapping {
        source: SourceDescriptor::new(500, SourceKind::Live).unwrap(),
        semantics: [unknown; WIFI_METRIC_CAPACITY],
        policy: None,
    }
}

fn row() -> BssObservation {
    BssObservation {
        bssid: [0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff],
        rssi_dbm: -53,
        link_quality: 72,
        center_frequency_khz: 5_180_000,
    }
}

fn summary(written: usize) -> BssReadSummary {
    BssReadSummary { written, timestamp: TimestampStatus::Unknown, cache_age_known: false }
}

#[test]
fn cached_row_yields_three_unmapped_unknown_time_observations() {
    let rows = [row()];
    let mut output = [None; WIFI_METRIC_CAPACITY];
    let count = adapt_cached_bss(&rows, summary(1), mapping(), &mut output).unwrap();
    assert_eq!(count, 3);
    let rssi = output[0].unwrap();
    assert_eq!(rssi.metric, WifiMetric::RssiDbm);
    assert_eq!(rssi.value, -53.0);
    assert_eq!(rssi.bssid, row().bssid);
    assert_eq!(
        rssi.radio_identifier_observation.identifier,
        RadioIdentifier::Bssid(RadioAddress48::from_octets(row().bssid)),
    );
    assert_eq!(rssi.radio_identifier_observation.observed_at, RadioObservationTime::Unknown);
    assert_eq!(rssi.envelope.source().source_id(), 500);
    assert_ne!(
        rssi.radio_identifier_observation.identifier,
        RadioIdentifier::Mac(RadioAddress48::from_octets(row().bssid)),
    );
    assert_ne!(rssi.envelope.source().source_id(), u64::from_le_bytes([0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0, 0]));
    assert_eq!(rssi.envelope.timestamp(), None);
    assert_eq!(rssi.envelope.kind(), ObservationKind::CachedQuery);
    assert_eq!(rssi.envelope.quantity_unit().quantity_id(), None);
    assert_eq!(rssi.envelope.evidence(), EvidenceOrigin::Unknown);
    assert!(!rssi.cache_age_known);
    assert_eq!(output[1].unwrap().metric, WifiMetric::LinkQuality);
    assert_eq!(output[1].unwrap().value, 72.0);
    assert_eq!(output[2].unwrap().metric, WifiMetric::CenterFrequencyKhz);
    assert_eq!(output[2].unwrap().value, 5_180_000.0);
}

#[test]
fn short_output_is_cleared_and_summary_count_must_match() {
    let rows = [row()];
    let mut output = [Some(WifiObservation {
        envelope: ObservationEnvelope::new(
            ObservationKind::CachedQuery,
            mapping().source,
            None, None, None, None,
            QuantityUnitIds::new(None, None).unwrap(),
            MeasurementUncertainty::Unknown,
            QualityRecord::Unknown,
            CalibrationRecord::new(CalibrationState::Missing).unwrap(),
            EvidenceOrigin::Unknown,
            None,
            HostPolicyDecision::NotEvaluated,
        ).unwrap(),
        metric: WifiMetric::RssiDbm,
        value: 0.0,
        bssid: [0; 6],
        radio_identifier_observation: RadioIdentifierObservation {
            identifier: RadioIdentifier::Bssid(RadioAddress48::from_octets([0; 6])),
            observed_at: RadioObservationTime::Unknown,
        },
        cache_timestamp: TimestampStatus::Unknown,
        cache_age_known: false,
    }); 2];
    assert_eq!(
        adapt_cached_bss(&rows, summary(1), mapping(), &mut output),
        Err(AdaptError::OutputTooSmall { required: 3 })
    );
    assert_eq!(output, [None; 2]);
    let mut enough = [None; 3];
    assert_eq!(
        adapt_cached_bss(&rows, summary(0), mapping(), &mut enough),
        Err(AdaptError::InvalidSummary)
    );
    assert_eq!(enough, [None; 3]);
}
