use etalon_sense_core::{
    evidence::EvidenceOrigin,
    extended_capability::ExtendedCapability,
    observation::{HostPolicyDecision, ObservationTimeAnchor},
    quality::SourceKind,
    quantity::{PhysicalQuantity, SampleEncoding, Unit},
    time::ClockStamp,
};

use crate::{
    CameraModality, FrameDeclaration, FrameFormat, FrameMetadata, FramePixelFormat,
    FramePlane, FrameTimestamp, NegotiatedFrame, RadiometricStatus,
};

use super::{
    HostPolicyClaim, HostPolicyOutcome, NativeTimestamp, ObservationEnvelope,
    ObservationError, ObservationQuantityMapping,
};

fn setup() -> (FrameDeclaration, NegotiatedFrame) {
    let capability = ExtendedCapability::new(
        30, 40, Some(50), SampleEncoding::U8, [2, 1, 0, 0], None,
    ).unwrap();
    let declaration = FrameDeclaration::new_with_pixel_format(
        10, CameraModality::Visible, capability, FramePixelFormat::Gray8,
        2, 1, 2, RadiometricStatus::NotApplicable,
    ).unwrap();
    let mut planes = [FramePlane::default(); 4];
    planes[0] = FramePlane::new(0, 2, 2, 1);
    let format = FrameFormat::new_with_pixel_format(
        60, 2, 1, 2, FramePixelFormat::Gray8, planes, 1,
    ).unwrap();
    let negotiated = NegotiatedFrame {
        source_id: 10,
        modality: CameraModality::Visible,
        format,
        radiometric: RadiometricStatus::NotApplicable,
    };
    (declaration, negotiated)
}

fn quantity() -> ObservationQuantityMapping {
    ObservationQuantityMapping::new(
        40, Some(50), Some(PhysicalQuantity::Dimensionless), Some(Unit::Unitless),
    ).unwrap()
}

#[test]
fn envelope_preserves_host_frame_and_semantic_declarations() {
    let (declaration, negotiated) = setup();
    let raw = NativeTimestamp::new(-4, 70, 80).unwrap();
    let policy = HostPolicyClaim::new(1, 2, 3, HostPolicyOutcome::Unknown).unwrap();
    let envelope = ObservationEnvelope::new(
        declaration,
        negotiated,
        FrameMetadata {
            format_id: 60,
            sequence: 9,
            gap_before: Some(2),
            timestamp: FrameTimestamp::Unknown,
        },
        90,
        4,
        quantity(),
        Some(raw),
        Some(ClockStamp::new(5, 6, 100, 2).unwrap()),
        Some(policy),
    ).unwrap();

    assert_eq!(envelope.source_id(), 10);
    assert_eq!(envelope.stream_id(), 90);
    assert_eq!(envelope.generation(), 4);
    assert_eq!(envelope.modality(), CameraModality::Visible);
    assert_eq!(envelope.quantity(), quantity());
    assert_eq!(envelope.pixel_format(), FramePixelFormat::Gray8);
    assert_eq!(envelope.planes()[0], FramePlane::new(0, 2, 2, 1));
    assert_eq!(envelope.sequence(), 9);
    assert_eq!(envelope.gap_before(), Some(2));
    assert_eq!(envelope.raw_timestamp(), Some(raw));
    assert_eq!(envelope.mapped_clock().unwrap().clock_id(), 5);
    assert_eq!(envelope.policy_claim(), Some(policy));
}

#[test]
fn envelope_rejects_source_and_quantity_binding_mismatches() {
    let (declaration, mut negotiated) = setup();
    negotiated.source_id = 11;
    let metadata = FrameMetadata { format_id: 60, sequence: 0, gap_before: None, timestamp: FrameTimestamp::Unknown };
    assert_eq!(ObservationEnvelope::new(declaration, negotiated, metadata, 90, 1,
        quantity(), None, None, None),
        Err(ObservationError::SourceMismatch));

    let (_, negotiated) = setup();
    let wrong_quantity = ObservationQuantityMapping::new(41, Some(50), None, None).unwrap();
    assert_eq!(ObservationEnvelope::new(declaration, negotiated, metadata, 90, 1,
        wrong_quantity, None, None, None), Err(ObservationError::QuantityMismatch));
}

#[test]
fn raw_and_mapped_timestamp_associations_are_not_silently_replaced() {
    let (declaration, negotiated) = setup();
    let raw_metadata = FrameMetadata {
        format_id: 60,
        sequence: 1,
        gap_before: None,
        timestamp: FrameTimestamp::Raw { ticks: 12, clock_domain_id: 7, unit_id: 8 },
    };
    let other_raw = NativeTimestamp::new(13, 7, 8).unwrap();
    assert_eq!(ObservationEnvelope::new(declaration, negotiated, raw_metadata, 90, 1,
        quantity(), Some(other_raw), None, None), Err(ObservationError::InvalidTimestamp));

    let mapped = ClockStamp::new(5, 6, 100, 2).unwrap();
    let mapped_metadata = FrameMetadata {
        format_id: 60,
        sequence: 2,
        gap_before: None,
        timestamp: FrameTimestamp::Mapped(mapped),
    };
    let different = ClockStamp::new(5, 6, 101, 2).unwrap();
    assert_eq!(ObservationEnvelope::new(declaration, negotiated, mapped_metadata, 90, 1,
        quantity(), None, Some(different), None), Err(ObservationError::InvalidTimestamp));
}

#[test]
fn quantity_unit_semantics_are_checked_when_typed_mapping_is_given() {
    assert_eq!(ObservationQuantityMapping::new(
        40, Some(50), Some(PhysicalQuantity::Temperature), Some(Unit::Pascal),
    ), Err(ObservationError::UnitMismatch));
}

#[test]
fn conversion_populates_core_envelope_without_forging_trust_or_time_anchor() {
    let (declaration, negotiated) = setup();
    let mapped = ClockStamp::new(12, 2, 500, 3).unwrap();
    let camera = ObservationEnvelope::new(
        declaration,
        negotiated,
        FrameMetadata {
            format_id: 60,
            sequence: 8,
            gap_before: Some(1),
            timestamp: FrameTimestamp::Mapped(mapped),
        },
        90,
        4,
        quantity(),
        None,
        Some(mapped),
        None,
    ).unwrap();
    let core = camera.to_core_observation(SourceKind::Live, ObservationTimeAnchor::Unknown).unwrap();
    assert_eq!(core.source().source_id(), 10);
    assert_eq!(core.source().kind(), SourceKind::Live);
    assert_eq!(core.source().trust_state(), etalon_sense_core::quality::TrustState::Unverified);
    assert_eq!(core.stream_id().unwrap().get(), 90);
    assert_eq!(core.generation().unwrap().get(), 4);
    assert_eq!(core.sequence(), Some(8));
    assert_eq!(core.timestamp(), Some(mapped));
    assert_eq!(core.time_anchor(), ObservationTimeAnchor::Unknown);
    assert_eq!(core.evidence(), EvidenceOrigin::Unknown);
    assert_eq!(core.policy(), HostPolicyDecision::NotEvaluated);

    let anchored = camera.to_core_observation(
        SourceKind::Recorded, ObservationTimeAnchor::ReadCompletion,
    ).unwrap();
    assert_eq!(anchored.time_anchor(), ObservationTimeAnchor::ReadCompletion);
}

#[test]
fn conversion_rejects_generation_that_does_not_fit_core_contract() {
    let (declaration, negotiated) = setup();
    let camera = ObservationEnvelope::new(
        declaration,
        negotiated,
        FrameMetadata { format_id: 60, sequence: 0, gap_before: None, timestamp: FrameTimestamp::Unknown },
        90,
        u64::from(u32::MAX) + 1,
        quantity(),
        None,
        None,
        None,
    ).unwrap();
    assert_eq!(camera.to_core_observation(SourceKind::Recorded, ObservationTimeAnchor::Unknown),
        Err(ObservationError::GenerationOutOfRange));
}
