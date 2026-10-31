use super::*;
use crate::{
    validate_batch, HostCellularTelemetry, OrdinalBars, RadioGeneration, TelemetryTimestamp,
};
use etalon_sense_core::quality::SourceKind;

fn validated() -> CellularTelemetry {
    let input = [HostCellularTelemetry {
        source_id: 700,
        generation: RadioGeneration::HostDeclared5G,
        rsrp_dbm: Some(-91.5),
        rsrq_db: None,
        sinr_db: Some(7.0),
        rssi_dbm: None,
        bars: Some(OrdinalBars {
            level: 3,
            levels: 5,
        }),
        sequence: 12,
        gap_before: Some(2),
        timestamp: TelemetryTimestamp::Unknown,
    }];
    let mut output = [None];
    validate_batch(&input, &mut output).unwrap();
    output[0].unwrap()
}

fn mapping() -> CellularObservationMapping {
    let unknown = QuantityUnitIds::new(None, None).unwrap();
    CellularObservationMapping {
        source: SourceDescriptor::new(900, SourceKind::Recorded).unwrap(),
        stream_id: 3,
        stream_generation: Some(8),
        semantics: [unknown; CELLULAR_METRIC_CAPACITY],
        policy: None,
    }
}

#[test]
fn adapts_only_present_fields_and_preserves_unknown_metadata() {
    let mut output = [None; CELLULAR_METRIC_CAPACITY];
    let count = adapt_cellular(validated(), mapping(), &mut output).unwrap();
    assert_eq!(count, 3);
    let first = output[0].unwrap();
    assert_eq!(first.metric, CellularMetric::RsrpDbm);
    assert_eq!(first.value, -91.5);
    assert_eq!(first.provider_source_id, 700);
    assert_eq!(first.envelope.source().source_id(), 900);
    assert_eq!(first.envelope.timestamp(), None);
    assert_eq!(first.envelope.sequence(), Some(12));
    assert_eq!(first.envelope.quantity_unit().quantity_id(), None);
    assert_eq!(first.envelope.evidence(), EvidenceOrigin::Unknown);
    assert_eq!(first.gap_before, Some(2));
    assert_eq!(output[2].unwrap().metric, CellularMetric::SignalBars);
    assert_eq!(output[2].unwrap().value, 3.0);
}

#[test]
fn too_small_output_is_cleared_without_partial_observations() {
    let mut output = [Some(adapt_placeholder()); 2];
    assert_eq!(
        adapt_cellular(validated(), mapping(), &mut output),
        Err(AdaptError::OutputTooSmall { required: 3 })
    );
    assert_eq!(output, [None; 2]);
}

fn adapt_placeholder() -> CellularObservation {
    let source = SourceDescriptor::new(1, SourceKind::Synthetic).unwrap();
    let envelope = ObservationEnvelope::new(
        ObservationKind::HostDerived,
        source,
        Some(1),
        None,
        None,
        None,
        QuantityUnitIds::new(None, None).unwrap(),
        MeasurementUncertainty::Unknown,
        QualityRecord::Unknown,
        CalibrationRecord::new(CalibrationState::Missing).unwrap(),
        EvidenceOrigin::Unknown,
        None,
        HostPolicyDecision::NotEvaluated,
    )
    .unwrap();
    CellularObservation {
        envelope,
        metric: CellularMetric::RssiDbm,
        value: 0.0,
        provider_source_id: 1,
        radio_generation: RadioGeneration::Unknown,
        gap_before: None,
    }
}
