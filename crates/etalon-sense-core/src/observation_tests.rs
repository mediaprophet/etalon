use super::*;
use crate::{
    evidence::EvidenceOrigin,
    policy::{EvidenceCommitment, PolicyId},
    quality::{CalibrationState, QualityRecord, SourceKind},
};

fn source(id: u64, kind: SourceKind) -> SourceDescriptor {
    SourceDescriptor::new(id, kind).unwrap()
}

fn calibration() -> CalibrationRecord {
    CalibrationRecord::new(CalibrationState::Missing).unwrap()
}

#[test]
fn directly_constructed_invalid_uncertainty_is_rejected() {
    for uncertainty in [
        MeasurementUncertainty::RelativeBound { fraction: f64::NAN },
        MeasurementUncertainty::RelativeBound { fraction: -1.0 },
        MeasurementUncertainty::AbsoluteBound {
            magnitude: f64::INFINITY,
            unit_id: None,
        },
        MeasurementUncertainty::Interval {
            lower: 2.0,
            upper: 1.0,
            unit_id: None,
        },
    ] {
        let result = ObservationEnvelope::new(
            ObservationKind::HostDerived,
            source(5, SourceKind::Synthetic),
            None,
            None,
            None,
            None,
            QuantityUnitIds::new(None, None).unwrap(),
            uncertainty,
            QualityRecord::Unknown,
            calibration(),
            EvidenceOrigin::Unknown,
            None,
            HostPolicyDecision::NotEvaluated,
        );
        assert_eq!(result, Err(ContractError::InvalidQuality));
    }
}

fn policy_request() -> PolicyRequest {
    let id = |n| PolicyId::new(n).unwrap();
    PolicyRequest::new(
        None,
        None,
        id(1),
        id(2),
        id(3),
        id(4),
        id(5),
        id(6),
        EvidenceCommitment::new([7; 32]).unwrap(),
    )
}

#[test]
fn live_observation_preserves_host_declarations_and_clock_domain() {
    let source = source(41, SourceKind::Live);
    let evidence = EvidenceOrigin::asserted(source, 9).unwrap();
    let time = ClockStamp::new(2, 3, 100, 1).unwrap();
    let ids = QuantityUnitIds::new(Some(77), Some(88)).unwrap();
    let uncertainty = MeasurementUncertainty::absolute_bound(0.25, Some(88)).unwrap();
    let outcome = PolicyOutcome::Permit(crate::policy::DecisionCommitment::new([9; 32]).unwrap());

    let envelope = ObservationEnvelope::new(
        ObservationKind::LiveFrame,
        source,
        Some(12),
        Some(4),
        Some(0),
        Some(time),
        ids,
        uncertainty,
        QualityRecord::Unknown,
        calibration(),
        evidence,
        None,
        HostPolicyDecision::Evaluated {
            request: policy_request(),
            outcome,
        },
    )
    .unwrap();

    assert_eq!(envelope.timestamp(), Some(time));
    assert_eq!(envelope.sequence(), Some(0));
    assert_eq!(
        envelope.source().trust_state(),
        crate::quality::TrustState::Unverified
    );
    assert_eq!(envelope.evidence(), evidence);
    assert!(matches!(
        envelope.policy(),
        HostPolicyDecision::Evaluated { .. }
    ));
}

#[test]
fn cached_query_can_keep_time_stream_and_provenance_unknown() {
    let envelope = ObservationEnvelope::new(
        ObservationKind::CachedQuery,
        source(5, SourceKind::Recorded),
        None,
        None,
        None,
        None,
        QuantityUnitIds::new(None, Some(99)).unwrap(),
        MeasurementUncertainty::Unknown,
        QualityRecord::Unknown,
        calibration(),
        EvidenceOrigin::Unknown,
        None,
        HostPolicyDecision::NotEvaluated,
    )
    .unwrap();

    assert_eq!(envelope.timestamp(), None);
    assert_eq!(envelope.stream_id(), None);
    assert_eq!(envelope.quantity_unit().quantity_id(), None);
    assert_eq!(envelope.quantity_unit().unit_id().unwrap().get(), 99);
    assert_eq!(envelope.evidence(), EvidenceOrigin::Unknown);
}

#[test]
fn stream_order_requires_a_nonzero_stream_identity() {
    let result = ObservationEnvelope::new(
        ObservationKind::HostDerived,
        source(5, SourceKind::Synthetic),
        None,
        Some(1),
        Some(2),
        None,
        QuantityUnitIds::new(None, None).unwrap(),
        MeasurementUncertainty::Unknown,
        QualityRecord::Unknown,
        calibration(),
        EvidenceOrigin::Unknown,
        None,
        HostPolicyDecision::NotEvaluated,
    );

    assert_eq!(result.unwrap_err(), ContractError::InvalidSource);
}

#[test]
fn evidence_for_a_different_source_is_rejected() {
    let result = ObservationEnvelope::new(
        ObservationKind::LiveFrame,
        source(5, SourceKind::Live),
        None,
        None,
        None,
        None,
        QuantityUnitIds::new(None, None).unwrap(),
        MeasurementUncertainty::Unknown,
        QualityRecord::Unknown,
        calibration(),
        EvidenceOrigin::asserted(source(6, SourceKind::Live), 9).unwrap(),
        None,
        HostPolicyDecision::NotEvaluated,
    );

    assert_eq!(result.unwrap_err(), ContractError::InvalidSource);
}

#[test]
fn direct_frame_kind_must_match_declared_source_kind() {
    let ids = QuantityUnitIds::new(None, None).unwrap();
    let build = |kind, source_kind| {
        ObservationEnvelope::new(
            kind,
            source(5, source_kind),
            None,
            None,
            None,
            None,
            ids,
            MeasurementUncertainty::Unknown,
            QualityRecord::Unknown,
            calibration(),
            EvidenceOrigin::Unknown,
            None,
            HostPolicyDecision::NotEvaluated,
        )
    };

    assert_eq!(
        build(ObservationKind::LiveFrame, SourceKind::Recorded),
        Err(ContractError::InvalidSource)
    );
    assert_eq!(
        build(ObservationKind::RecordedFrame, SourceKind::Synthetic),
        Err(ContractError::InvalidSource)
    );
    assert_eq!(
        build(ObservationKind::SyntheticFrame, SourceKind::Live),
        Err(ContractError::InvalidSource)
    );
    assert!(build(ObservationKind::CachedQuery, SourceKind::Live).is_ok());
    assert!(build(ObservationKind::HostDerived, SourceKind::Recorded).is_ok());
}

#[test]
fn absolute_and_interval_uncertainty_require_the_declared_unit() {
    let build = |uncertainty, unit_id| {
        ObservationEnvelope::new(
            ObservationKind::LiveFrame,
            source(5, SourceKind::Live),
            None,
            None,
            None,
            None,
            QuantityUnitIds::new(Some(7), unit_id).unwrap(),
            uncertainty,
            QualityRecord::Unknown,
            calibration(),
            EvidenceOrigin::Unknown,
            None,
            HostPolicyDecision::NotEvaluated,
        )
    };
    let absolute = MeasurementUncertainty::absolute_bound(0.1, Some(8)).unwrap();
    let interval = MeasurementUncertainty::interval(-0.1, 0.1, Some(8)).unwrap();

    assert!(build(absolute, Some(8)).is_ok());
    assert_eq!(build(absolute, None), Err(ContractError::IncompatibleUnit));
    assert_eq!(
        build(absolute, Some(9)),
        Err(ContractError::IncompatibleUnit)
    );
    assert_eq!(
        build(interval, Some(9)),
        Err(ContractError::IncompatibleUnit)
    );
    assert!(build(MeasurementUncertainty::relative_bound(0.2).unwrap(), None).is_ok());
    assert!(build(MeasurementUncertainty::Unknown, None).is_ok());
}

#[test]
fn uncertainty_rejects_nonfinite_negative_and_reversed_values() {
    assert_eq!(
        MeasurementUncertainty::absolute_bound(f64::NAN, None).unwrap_err(),
        ContractError::InvalidQuality
    );
    assert_eq!(
        MeasurementUncertainty::relative_bound(-0.1).unwrap_err(),
        ContractError::InvalidQuality
    );
    assert_eq!(
        MeasurementUncertainty::interval(2.0, 1.0, None).unwrap_err(),
        ContractError::InvalidQuality
    );
    assert_eq!(
        MeasurementUncertainty::absolute_bound(1.0, Some(0)).unwrap_err(),
        ContractError::InvalidIdentifier
    );
}
