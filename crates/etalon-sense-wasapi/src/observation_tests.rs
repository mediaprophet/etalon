use super::*;
use etalon_sense_core::{
    adapter::{AdapterId, ExtendedCaptureSession, ExtendedDiscoveredSource},
    extended_capability::ExtendedCapability,
    policy::{DecisionCommitment, EvidenceCommitment, PolicyId},
    quantity::SampleEncoding,
};

fn session() -> ExtendedCaptureSession {
    let capability = ExtendedCapability::new(
        40, 50, Some(60), SampleEncoding::I16, [480, 2, 0, 0], None,
    ).unwrap();
    let source = ExtendedDiscoveredSource::new(70, capability).unwrap();
    ExtendedCaptureSession::new(AdapterId::new(80).unwrap(), source, 90, 2).unwrap()
}

#[test]
fn envelope_keeps_exact_host_semantics_policy_and_unknown_sample_time() {
    let id = |value| PolicyId::new(value).unwrap();
    let request = PolicyRequest::new(
        None, None, id(1), id(2), id(3), id(4), id(5), id(6),
        EvidenceCommitment::new([7; 32]).unwrap(),
    );
    let mapping = WasapiObservationMapping {
        source_kind: SourceKind::Live,
        policy: Some((request, PolicyOutcome::Permit(DecisionCommitment::new([8; 32]).unwrap()))),
    };
    let envelope = make_envelope(session(), mapping).unwrap();
    assert_eq!(envelope.source().source_id(), 70);
    assert_eq!(envelope.stream_id().unwrap().get(), 90);
    assert_eq!(envelope.generation().unwrap().get(), 2);
    assert_eq!(envelope.sequence(), None);
    assert_eq!(envelope.timestamp(), None);
    assert_eq!(envelope.time_anchor(), etalon_sense_core::observation::ObservationTimeAnchor::Unknown);
    assert_eq!(envelope.quantity_unit().quantity_id().unwrap().get(), 50);
    assert_eq!(envelope.quantity_unit().unit_id().unwrap().get(), 60);
    assert_eq!(envelope.evidence(), EvidenceOrigin::Unknown);
    assert!(matches!(envelope.policy(), HostPolicyDecision::Evaluated { request: seen, .. } if seen == request));
}

#[test]
fn live_capture_rejects_nonlive_mapping_before_acquisition() {
    let mapping = WasapiObservationMapping { source_kind: SourceKind::Recorded, policy: None };
    assert_eq!(make_envelope(session(), mapping), Err(ObservationError::Contract(ContractError::InvalidSource)));
}
