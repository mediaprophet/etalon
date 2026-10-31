use super::{ClockStamp, IntervalRelation, TimeInterval};
use crate::ContractError;
use core::cmp::Ordering;

#[test]
fn timestamps_compare_only_inside_same_clock_epoch() {
    let a = ClockStamp::new(7, 2, 100, 0).unwrap();
    let b = ClockStamp::new(7, 2, 150, 0).unwrap();
    assert_eq!(a.compare_center(b), Ok(Ordering::Less));
    assert_eq!(b.delta_ns(a), Ok(50));
    assert_eq!(a.relation_to(b), Ok(IntervalRelation::Before));

    let other_clock = ClockStamp::new(8, 2, 150, 0).unwrap();
    let restarted = ClockStamp::new(7, 3, 150, 0).unwrap();
    assert_eq!(a.compare_center(other_clock), Err(ContractError::ClockDomainMismatch));
    assert_eq!(a.delta_ns(restarted), Err(ContractError::ClockDomainMismatch));
    assert_eq!(a.relation_to(other_clock), Err(ContractError::ClockDomainMismatch));
}

#[test]
fn interval_clamps_lower_bound_and_rejects_upper_overflow() {
    let interval = TimeInterval::around(3, 9).unwrap();
    assert_eq!(interval.lower_ns(), 0);
    assert_eq!(interval.upper_ns(), 12);
    assert_eq!(
        TimeInterval::around(u64::MAX, 1),
        Err(ContractError::InvalidTime)
    );
    assert_eq!(TimeInterval::new(2, 1), Err(ContractError::InvalidTime));
}

#[test]
fn uncertain_intervals_report_overlap_without_claiming_order() {
    let a = ClockStamp::new(1, 0, 10, 5).unwrap();
    let b = ClockStamp::new(1, 0, 18, 5).unwrap();
    assert_eq!(a.relation_to(b), Ok(IntervalRelation::Overlaps));
}

#[test]
fn identifier_and_full_width_integer_bounds_are_preserved() {
    assert_eq!(ClockStamp::new(0, 0, 0, 0), Err(ContractError::InvalidIdentifier));
    let early = ClockStamp::new(1, 0, 0, 0).unwrap();
    let late = ClockStamp::new(1, 0, u64::MAX, 0).unwrap();
    assert_eq!(late.delta_ns(early), Ok(u64::MAX as i128));
}
