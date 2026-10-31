use super::*;
use crate::{AppliedTuning, OverrunReport, SdrError, TimestampAnchor};

fn report(
    stream_id: Option<u64>,
    generation: Option<u32>,
    sequence: Option<u64>,
    timestamp: Option<ClockStamp>,
) -> SdrReadReport {
    SdrReadReport::new(
        8,
        2,
        timestamp,
        if timestamp.is_some() {
            TimestampAnchor::FirstSample
        } else {
            TimestampAnchor::Unknown
        },
        AppliedTuning::Unknown,
        OverrunReport::Unknown,
    )
    .unwrap()
    .with_stream_position(stream_id, generation, sequence)
    .unwrap()
}

#[test]
fn legacy_constructor_and_unknown_updates_preserve_last_known_order() {
    let mut tracker = SdrContinuityTracker::new(21, 4).unwrap();
    let first = report(
        Some(21),
        Some(4),
        Some(10),
        Some(ClockStamp::new(2, 1, 100, 0).unwrap()),
    );
    assert_eq!(
        tracker.observe_report(first),
        Ok(Some(SequenceEvent::First))
    );

    let unknown_generation = report(Some(21), None, Some(100), None);
    assert_eq!(tracker.observe_report(unknown_generation), Ok(None));
    assert_eq!(
        tracker.last_observation(),
        Some((10, Some(ClockStamp::new(2, 1, 100, 0).unwrap())))
    );

    let legacy_unknown = SdrReadReport::new(
        8,
        2,
        None,
        TimestampAnchor::Unknown,
        AppliedTuning::Unknown,
        OverrunReport::Unknown,
    )
    .unwrap();
    assert_eq!(legacy_unknown.stream_id(), None);
    assert_eq!(legacy_unknown.generation(), None);
    assert_eq!(legacy_unknown.sequence(), None);
    assert_eq!(tracker.observe_report(legacy_unknown), Ok(None));

    let next = report(Some(21), Some(4), Some(11), None);
    assert_eq!(
        tracker.observe_report(next),
        Ok(Some(SequenceEvent::Contiguous))
    );
    assert_eq!(tracker.last_observation(), Some((11, None)));
    assert_eq!(
        tracker.last_known_timestamp(),
        Some(ClockStamp::new(2, 1, 100, 0).unwrap())
    );
}

#[test]
fn tracker_rejects_duplicates_and_foreign_stream_without_advancing() {
    let mut tracker = SdrContinuityTracker::new(7, 2).unwrap();
    assert_eq!(
        tracker.observe_report(report(Some(7), Some(2), Some(5), None)),
        Ok(Some(SequenceEvent::First))
    );
    assert_eq!(
        tracker.observe_report(report(Some(7), Some(2), Some(5), None)),
        Err(SdrContinuityError::Contract(ContractError::InvalidSource)),
    );
    assert_eq!(
        tracker.observe_report(report(Some(8), Some(2), Some(6), None)),
        Err(SdrContinuityError::Contract(ContractError::InvalidSource)),
    );
    assert_eq!(tracker.last_observation(), Some((5, None)));
}

#[test]
fn optional_stream_fields_validate_zero_and_unbound_values() {
    let base = SdrReadReport::new(
        8,
        2,
        None,
        TimestampAnchor::Unknown,
        AppliedTuning::Unknown,
        OverrunReport::Unknown,
    )
    .unwrap();
    assert_eq!(
        base.with_stream_position(Some(0), Some(1), Some(0)),
        Err(SdrError::InvalidReport)
    );
    assert_eq!(
        base.with_stream_position(None, Some(1), None),
        Err(SdrError::InvalidReport)
    );
    assert_eq!(
        base.with_stream_position(None, None, Some(0)),
        Err(SdrError::InvalidReport)
    );
    assert_eq!(
        base.with_stream_position(Some(3), None, Some(0))
            .unwrap()
            .sequence(),
        Some(0)
    );
}
