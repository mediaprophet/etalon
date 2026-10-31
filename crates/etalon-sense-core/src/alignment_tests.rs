use super::*;
use crate::{
    evidence::EvidenceOrigin,
    observation::{MeasurementUncertainty, QuantityUnitIds},
    quality::{CalibrationRecord, CalibrationState, QualityRecord, SourceDescriptor, SourceKind},
    synchronization::ClockAnchor,
    time::TimeInterval,
};

fn envelope(
    source_id: u64,
    timestamp: Option<ClockStamp>,
    quantity_id: u64,
) -> ObservationEnvelope {
    let envelope = ObservationEnvelope::new(
        crate::observation::ObservationKind::LiveFrame,
        SourceDescriptor::new(source_id, SourceKind::Live).unwrap(),
        Some(10),
        Some(1),
        Some(5),
        timestamp,
        QuantityUnitIds::new(Some(quantity_id), Some(quantity_id + 100)).unwrap(),
        MeasurementUncertainty::Unknown,
        QualityRecord::Unknown,
        CalibrationRecord::new(CalibrationState::Missing).unwrap(),
        EvidenceOrigin::Unknown,
        None,
        crate::observation::HostPolicyDecision::NotEvaluated,
    )
    .unwrap();
    if timestamp.is_some() {
        envelope
            .with_time_anchor(ObservationTimeAnchor::FirstSample)
            .unwrap()
    } else {
        envelope
    }
}

fn timed(observation: ObservationEnvelope) -> TimedObservation {
    TimedObservation::new(observation, Some(TimestampAnchorId::new(1).unwrap()))
}

fn config(
    missing: MissingObservationPolicy,
    unknown_time: UnknownTimestampPolicy,
    max_skew: u64,
) -> AlignmentConfig {
    AlignmentConfig::new(
        ClockDomain::new(9, 2).unwrap(),
        max_skew,
        missing,
        unknown_time,
    )
}

fn synchronization() -> ClockSynchronization {
    let source_a = ClockStamp::new(3, 1, 100, 0).unwrap();
    let source_b = ClockStamp::new(3, 1, 200, 0).unwrap();
    let target_a = ClockStamp::new(9, 2, 1_000, 0).unwrap();
    let target_b = ClockStamp::new(9, 2, 1_100, 0).unwrap();
    ClockSynchronization::new(
        55,
        ClockAnchor::new(source_a, target_a),
        ClockAnchor::new(source_b, target_b),
        TimeInterval::new(110, 190).unwrap(),
        0,
        0,
        10,
    )
    .unwrap()
}

#[test]
fn target_domain_observations_align_in_expected_slot_order() {
    let first = envelope(1, Some(ClockStamp::new(9, 2, 100, 1).unwrap()), 21);
    let second = envelope(2, Some(ClockStamp::new(9, 2, 105, 1).unwrap()), 22);
    let expected = [
        ObservationKey::from_observation(first),
        ObservationKey::from_observation(second),
    ];
    let mut output = [None; 2];

    let report = align_observations(
        &[timed(second), timed(first)],
        &expected,
        &[],
        config(
            MissingObservationPolicy::Reject,
            UnknownTimestampPolicy::Reject,
            10,
        ),
        &mut output,
    )
    .unwrap();

    assert_eq!(report.written, 2);
    assert_eq!(report.conservative_skew_ns, Some(7));
    assert_eq!(
        (
            report.present,
            report.aligned,
            report.unaligned,
            report.missing
        ),
        (2, 2, 0, 0)
    );
    assert_eq!(output[0].unwrap().observation(), Some(first));
    assert_eq!(output[0].unwrap().target_timestamp(), first.timestamp());
}

#[test]
fn alignment_key_rejects_different_observation_route() {
    let item = envelope(1, Some(ClockStamp::new(9, 2, 100, 0).unwrap()), 10);
    let expected = [ObservationKey::new(
        crate::observation::ObservationKind::RecordedFrame,
        SourceKind::Recorded,
        1,
        Some(10),
        Some(1),
        Some(5),
        Some(10),
        Some(110),
    )
    .unwrap()];
    let mut output = [None; 1];
    assert_eq!(
        align_observations(
            &[timed(item)],
            &expected,
            &[],
            config(
                MissingObservationPolicy::Reject,
                UnknownTimestampPolicy::Reject,
                0
            ),
            &mut output,
        ),
        Err(AlignmentError::UnexpectedObservation),
    );
    assert_eq!(output, [None]);
}

#[test]
fn alignment_key_rejects_different_source_origin() {
    let item = envelope(1, Some(ClockStamp::new(9, 2, 100, 0).unwrap()), 10);
    let expected = [ObservationKey::new(
        crate::observation::ObservationKind::LiveFrame,
        SourceKind::Recorded,
        1,
        Some(10),
        Some(1),
        Some(5),
        Some(10),
        Some(110),
    )
    .unwrap()];
    let mut output = [None; 1];
    assert_eq!(
        align_observations(
            &[timed(item)],
            &expected,
            &[],
            config(
                MissingObservationPolicy::Reject,
                UnknownTimestampPolicy::Reject,
                0
            ),
            &mut output,
        ),
        Err(AlignmentError::UnexpectedObservation),
    );
    assert_eq!(output, [None]);
}

#[test]
fn non_target_domain_requires_exact_mapping_and_preserves_original_stamp() {
    let original = ClockStamp::new(3, 1, 150, 1).unwrap();
    let item = envelope(1, Some(original), 20);
    let expected = [ObservationKey::from_observation(item)];
    let sync = synchronization();
    let mut output = [None; 1];

    let result = align_observations(
        &[timed(item)],
        &expected,
        &[sync],
        config(
            MissingObservationPolicy::Reject,
            UnknownTimestampPolicy::Reject,
            10,
        ),
        &mut output,
    )
    .unwrap();
    let aligned = output[0].unwrap();
    assert!(result.conservative_skew_ns.is_some());
    assert_eq!(aligned.observation().unwrap().timestamp(), Some(original));
    assert_eq!(aligned.target_timestamp().unwrap().clock_id(), 9);
    assert_eq!(aligned.target_timestamp().unwrap().ticks_ns(), 1_050);
    assert_eq!(aligned.synchronization_id(), Some(55));

    assert_eq!(
        align_observations(
            &[timed(item)],
            &expected,
            &[],
            config(
                MissingObservationPolicy::Reject,
                UnknownTimestampPolicy::Reject,
                10
            ),
            &mut output,
        ),
        Err(AlignmentError::MissingClockSynchronization),
    );
    assert_eq!(output, [None]);
}

#[test]
fn unknown_times_and_missing_members_follow_explicit_policy() {
    let known = envelope(1, Some(ClockStamp::new(9, 2, 10, 0).unwrap()), 10);
    let unknown = envelope(2, None, 20);
    let absent = envelope(3, None, 30);
    let expected = [
        ObservationKey::from_observation(known),
        ObservationKey::from_observation(unknown),
        ObservationKey::from_observation(absent),
    ];
    let mut output = [None; 3];

    align_observations(
        &[timed(known), timed(unknown)],
        &expected,
        &[],
        config(
            MissingObservationPolicy::PreserveMissing,
            UnknownTimestampPolicy::PreserveUnaligned,
            0,
        ),
        &mut output,
    )
    .unwrap();
    assert_eq!(
        (
            output[0].unwrap().observation().is_some(),
            output[1].unwrap().observation().is_some(),
            output[2].unwrap().observation().is_some()
        ),
        (true, true, false)
    );
    assert_eq!(output[0].unwrap().target_timestamp(), known.timestamp());
    assert_eq!(output[1].unwrap().observation(), Some(unknown));
    assert_eq!(output[1].unwrap().target_timestamp(), None);
    assert_eq!(output[2].unwrap().observation(), None);

    assert_eq!(
        align_observations(
            &[timed(known), timed(unknown)],
            &expected,
            &[],
            config(
                MissingObservationPolicy::Reject,
                UnknownTimestampPolicy::Reject,
                0
            ),
            &mut output,
        ),
        Err(AlignmentError::UnknownTimestamp),
    );
    assert_eq!(output, [None; 3]);
}

#[test]
fn uncertainty_clamps_at_clock_zero_and_upper_overflow_is_rejected() {
    let stamp = ClockStamp::new(9, 2, 2, 5).unwrap();
    assert_eq!(stamp.interval().lower_ns(), 0);
    assert_eq!(stamp.interval().upper_ns(), 7);
    let item = envelope(1, Some(stamp), 9);
    let expected = [ObservationKey::from_observation(item)];
    let mut output = [None; 1];
    let report = align_observations(
        &[timed(item)],
        &expected,
        &[],
        config(
            MissingObservationPolicy::Reject,
            UnknownTimestampPolicy::Reject,
            7,
        ),
        &mut output,
    )
    .unwrap();
    assert_eq!(report.conservative_skew_ns, Some(7));
    assert_eq!(
        ClockStamp::new(9, 2, u64::MAX, 1),
        Err(ContractError::InvalidTime),
    );
}

#[test]
fn duplicate_and_skew_errors_clear_all_output() {
    let a = envelope(1, Some(ClockStamp::new(9, 2, 10, 0).unwrap()), 10);
    let b = envelope(2, Some(ClockStamp::new(9, 2, 30, 0).unwrap()), 20);
    let expected = [
        ObservationKey::from_observation(a),
        ObservationKey::from_observation(b),
    ];
    let mut output = [None; 2];
    assert_eq!(
        align_observations(
            &[timed(a), timed(a)],
            &expected,
            &[],
            config(
                MissingObservationPolicy::PreserveMissing,
                UnknownTimestampPolicy::Reject,
                100
            ),
            &mut output,
        ),
        Err(AlignmentError::DuplicateObservation),
    );
    assert_eq!(output, [None; 2]);
    assert_eq!(
        align_observations(
            &[timed(a), timed(b)],
            &expected,
            &[],
            config(
                MissingObservationPolicy::Reject,
                UnknownTimestampPolicy::Reject,
                19
            ),
            &mut output,
        ),
        Err(AlignmentError::SkewExceeded {
            measured_ns: 20,
            maximum_ns: 19
        }),
    );
    assert_eq!(output, [None; 2]);
    assert_eq!(
        align_observations(
            &[],
            &[expected[0], expected[0]],
            &[],
            config(
                MissingObservationPolicy::PreserveMissing,
                UnknownTimestampPolicy::Reject,
                1
            ),
            &mut output,
        ),
        Err(AlignmentError::DuplicateExpectedKey),
    );
    assert_eq!(output, [None; 2]);
}

#[test]
fn member_ceiling_is_enforced_without_partial_output() {
    let sample = envelope(1, None, 11);
    let key = ObservationKey::from_observation(sample);
    let expected = [key; MAX_ALIGNMENT_MEMBERS + 1];
    let mut output = [None; MAX_ALIGNMENT_MEMBERS + 1];
    assert_eq!(
        align_observations(
            &[],
            &expected,
            &[],
            config(
                MissingObservationPolicy::PreserveMissing,
                UnknownTimestampPolicy::PreserveUnaligned,
                0
            ),
            &mut output,
        ),
        Err(AlignmentError::TooManyMembers {
            maximum: MAX_ALIGNMENT_MEMBERS
        }),
    );
    assert!(output.iter().all(Option::is_none));
}

#[test]
fn incompatible_event_anchors_are_never_joined() {
    let a = envelope(1, Some(ClockStamp::new(9, 2, 10, 0).unwrap()), 10);
    let b = envelope(2, Some(ClockStamp::new(9, 2, 10, 0).unwrap()), 20)
        .with_time_anchor(ObservationTimeAnchor::ReadCompletion)
        .unwrap();
    let expected = [
        ObservationKey::from_observation(a),
        ObservationKey::from_observation(b),
    ];
    let items = [
        TimedObservation::new(a, Some(TimestampAnchorId::new(1).unwrap())),
        TimedObservation::new(b, Some(TimestampAnchorId::new(1).unwrap())),
    ];
    let mut output = [None; 2];
    assert_eq!(
        align_observations(
            &items,
            &expected,
            &[],
            config(
                MissingObservationPolicy::Reject,
                UnknownTimestampPolicy::Reject,
                0
            ),
            &mut output,
        ),
        Err(AlignmentError::TimestampAnchorMismatch),
    );
    assert_eq!(output, [None; 2]);

    let unknown_envelope = a.with_time_anchor(ObservationTimeAnchor::Unknown).unwrap();
    let unknown_anchor = [TimedObservation::new(
        unknown_envelope,
        Some(TimestampAnchorId::new(1).unwrap()),
    )];
    let one_expected = [expected[0]];
    let mut one_output = [None; 1];
    assert_eq!(
        align_observations(
            &unknown_anchor,
            &one_expected,
            &[],
            config(
                MissingObservationPolicy::Reject,
                UnknownTimestampPolicy::PreserveUnaligned,
                0
            ),
            &mut one_output,
        )
        .unwrap()
        .conservative_skew_ns,
        None,
    );
    assert_eq!(one_output[0].unwrap().target_timestamp(), None);
    assert_eq!(
        align_observations(
            &[unknown_anchor[0]],
            &one_expected,
            &[],
            config(
                MissingObservationPolicy::Reject,
                UnknownTimestampPolicy::Reject,
                0
            ),
            &mut one_output,
        ),
        Err(AlignmentError::UnknownTimestamp),
    );
    assert_eq!(one_output, [None]);

    struct ExplicitHostAnchor;
    impl TimestampAnchorAuthority for ExplicitHostAnchor {
        fn authority_id(&self) -> u64 {
            90
        }

        fn resolve(
            &self,
            _envelope_anchor: ObservationTimeAnchor,
            host_anchor_id: Option<TimestampAnchorId>,
        ) -> Option<TimestampAnchorId> {
            host_anchor_id.filter(|id| id.get() == 1)
        }
    }
    let authority = ExplicitHostAnchor;
    let known_first = TimedObservation::new(a, Some(TimestampAnchorId::new(1).unwrap()));
    let known_completion = TimedObservation::new(b, Some(TimestampAnchorId::new(1).unwrap()));
    assert_eq!(
        align_observations_with_anchor_authority(
            &[known_first, known_completion],
            &expected,
            &[],
            config(
                MissingObservationPolicy::Reject,
                UnknownTimestampPolicy::Reject,
                0
            ),
            &authority,
            &mut output,
        )
        .unwrap()
        .aligned,
        2,
    );
    assert!(matches!(
        output[0].unwrap().alignment_anchor(),
        Some(AlignmentAnchor::AuthorityMapped { authority_id: 90, anchor_id }) if anchor_id.get() == 1
    ));
    assert_eq!(
        align_observations_with_anchor_authority(
            &unknown_anchor,
            &one_expected,
            &[],
            config(
                MissingObservationPolicy::Reject,
                UnknownTimestampPolicy::Reject,
                0
            ),
            &authority,
            &mut one_output,
        )
        .unwrap()
        .aligned,
        1,
    );
}

#[test]
fn synchronization_list_is_bounded_and_errors_clear_output() {
    let sync = synchronization();
    let synchronizations = [sync; MAX_ALIGNMENT_SYNCHRONIZATIONS + 1];
    let item = envelope(1, Some(ClockStamp::new(9, 2, 10, 0).unwrap()), 12);
    let expected = [ObservationKey::from_observation(item)];
    let mut output = [None; 1];
    assert_eq!(
        align_observations(
            &[timed(item)],
            &expected,
            &synchronizations,
            config(
                MissingObservationPolicy::Reject,
                UnknownTimestampPolicy::Reject,
                0
            ),
            &mut output,
        ),
        Err(AlignmentError::TooManySynchronizations {
            maximum: MAX_ALIGNMENT_SYNCHRONIZATIONS
        }),
    );
    assert_eq!(output, [None]);
}
