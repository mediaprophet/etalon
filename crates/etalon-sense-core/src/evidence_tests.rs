use super::{
    ArtifactDigest, AuthorityKeyRef, DigestAlgorithm, EvidenceOrigin, EvidenceRecord,
    ProcessingEvidence,
};
use crate::{
    quality::{
        CalibrationArtifactId, CalibrationRecord, CalibrationState, SourceDescriptor, SourceKind,
        TrustState,
    },
    time::ClockStamp,
    ContractError,
};

fn digest(byte: u8) -> ArtifactDigest {
    ArtifactDigest::new(DigestAlgorithm::Sha256, [byte; 32])
}

fn source() -> SourceDescriptor {
    SourceDescriptor::new(9, SourceKind::Live).unwrap()
}

#[test]
fn unknown_evidence_remains_unknown_without_synthesized_values() {
    let record = EvidenceRecord::unknown();
    assert_eq!(record.origin(), EvidenceOrigin::Unknown);
    assert_eq!(record.stream_id(), None);
    assert_eq!(record.sequence(), None);
    assert_eq!(record.timestamp(), None);
    assert_eq!(record.payload_digest(), None);
    assert_eq!(record.processing(), None);
    assert_eq!(record.calibration().state(), CalibrationState::Missing);
}

#[test]
fn caller_assertion_is_not_source_verification() {
    let origin = EvidenceOrigin::asserted(source(), 15).unwrap();
    let EvidenceOrigin::Asserted {
        source: declared_source,
        assertion_id,
    } = origin
    else {
        panic!("expected assertion");
    };
    assert_eq!(declared_source.source_id(), 9);
    assert_eq!(declared_source.trust_state(), TrustState::Unverified);
    assert_eq!(assertion_id, 15);
    assert_eq!(
        EvidenceOrigin::asserted(source(), 0),
        Err(ContractError::InvalidIdentifier)
    );
}

#[test]
fn presented_signature_is_not_a_verified_origin() {
    let authority = AuthorityKeyRef::new(4, 5).unwrap();
    let origin = EvidenceOrigin::signature_presented(source(), authority, digest(1), digest(2));
    let EvidenceOrigin::SignaturePresented {
        source: declared_source,
        authority: declared_authority,
        signed_digest,
        signature_digest,
    } = origin
    else {
        panic!("signature presentation must not become verification");
    };
    assert_eq!(declared_source.trust_state(), TrustState::Unverified);
    assert_eq!(declared_authority.issuer_id(), 4);
    assert_eq!(declared_authority.key_id(), 5);
    assert_eq!(signed_digest, digest(1));
    assert_eq!(signature_digest, digest(2));
    assert_eq!(
        AuthorityKeyRef::new(0, 5),
        Err(ContractError::InvalidIdentifier)
    );
    assert_eq!(
        AuthorityKeyRef::new(4, 0),
        Err(ContractError::InvalidIdentifier)
    );
}

#[test]
fn receipt_preserves_stream_sequence_clock_and_artifact_references() {
    let stamp = ClockStamp::new(3, 2, 100, 5).unwrap();
    let artifact = CalibrationArtifactId::new([8; 32]).unwrap();
    let calibration = CalibrationRecord::new(CalibrationState::Calibrated {
        artifact_id: artifact,
    })
    .unwrap();
    let processing = ProcessingEvidence::new(11, Some(digest(3)), Some(12)).unwrap();
    let record = EvidenceRecord::new(
        EvidenceOrigin::asserted(source(), 15).unwrap(),
        Some(17),
        Some(0),
        Some(stamp),
        Some(digest(9)),
        calibration,
        Some(processing),
    )
    .unwrap();

    assert_eq!(record.stream_id(), Some(17));
    assert_eq!(record.sequence(), Some(0));
    assert_eq!(record.timestamp(), Some(stamp));
    assert_eq!(record.payload_digest(), Some(digest(9)));
    assert_eq!(record.calibration().state(), calibration.state());
    assert_eq!(record.processing(), Some(processing));
}

#[test]
fn absent_processing_facts_stay_absent_and_zero_ids_are_rejected() {
    let processing = ProcessingEvidence::new(22, None, None).unwrap();
    assert_eq!(processing.configuration_digest(), None);
    assert_eq!(processing.provider_id(), None);
    assert_eq!(
        ProcessingEvidence::new(0, None, None),
        Err(ContractError::InvalidIdentifier)
    );
    assert_eq!(
        ProcessingEvidence::new(22, None, Some(0)),
        Err(ContractError::InvalidIdentifier)
    );

    assert_eq!(
        EvidenceRecord::new(
            EvidenceOrigin::Unknown,
            Some(0),
            None,
            None,
            None,
            CalibrationRecord::new(CalibrationState::Uncalibrated).unwrap(),
            None,
        ),
        Err(ContractError::InvalidIdentifier)
    );
}

#[test]
fn digest_algorithm_is_explicit_and_digest_is_only_a_reference() {
    let sha = digest(7);
    let blake = ArtifactDigest::new(DigestAlgorithm::Blake3, [7; 32]);
    assert_eq!(sha.algorithm(), DigestAlgorithm::Sha256);
    assert_eq!(sha.bytes(), [7; 32]);
    assert_ne!(sha, blake);
}
