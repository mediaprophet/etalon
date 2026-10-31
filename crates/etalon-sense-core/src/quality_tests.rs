use super::{
    CalibrationArtifactId, CalibrationRecord, CalibrationState, QualityMetrics, QualityRecord,
    SourceDescriptor, SourceKind, TrustState,
};
use crate::ContractError;

#[test]
fn source_declarations_never_mint_trust() {
    for kind in [
        SourceKind::Synthetic,
        SourceKind::Recorded,
        SourceKind::Live,
    ] {
        let source = SourceDescriptor::new(1, kind).unwrap();
        assert_eq!(source.kind(), kind);
        assert_eq!(source.trust_state(), TrustState::Unverified);
    }
    assert_eq!(
        SourceDescriptor::new(0, SourceKind::Live),
        Err(ContractError::InvalidIdentifier)
    );
}

#[test]
fn calibration_state_preserves_missing_and_nonverified_states() {
    for state in [
        CalibrationState::Missing,
        CalibrationState::Expired,
        CalibrationState::OutOfRange,
        CalibrationState::Uncalibrated,
    ] {
        assert_eq!(CalibrationRecord::new(state).unwrap().state(), state);
    }
}

#[test]
fn calibration_artifact_identifier_must_be_nonzero() {
    assert_eq!(
        CalibrationArtifactId::new([0; 32]),
        Err(ContractError::InvalidCalibration)
    );
    let id = CalibrationArtifactId::new([1; 32]).unwrap();
    let state = CalibrationState::Calibrated { artifact_id: id };
    assert_eq!(CalibrationRecord::new(state).unwrap().state(), state);
}

#[test]
fn unknown_quality_is_explicit_and_metrics_have_no_implicit_default() {
    assert_eq!(QualityRecord::unknown(), QualityRecord::Unknown);
    assert!(matches!(
        QualityRecord::known(QualityMetrics::new(0.8, 0.2).unwrap()),
        QualityRecord::Known(_)
    ));
}

#[test]
fn quality_rejects_nonfinite_and_out_of_range_values() {
    for (confidence, uncertainty) in [
        (f32::NAN, 0.1),
        (f32::INFINITY, 0.1),
        (0.5, f32::NEG_INFINITY),
        (-0.01, 0.1),
        (1.01, 0.1),
        (0.5, -0.01),
        (0.5, 1.01),
    ] {
        assert_eq!(
            QualityMetrics::new(confidence, uncertainty),
            Err(ContractError::InvalidQuality)
        );
    }
    assert!(QualityMetrics::new(0.0, 1.0).is_ok());
}
