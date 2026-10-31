use super::*;

fn stamp(clock: u64, epoch: u32, ticks: u64, uncertainty: u64) -> ClockStamp {
    ClockStamp::new(clock, epoch, ticks, uncertainty).unwrap()
}

fn calibration(max_drift_ppb: u64, max_uncertainty_ns: u64) -> ClockSynchronization {
    let first = ClockAnchor::new(stamp(1, 2, 1_000, 1), stamp(10, 3, 5_000, 1));
    let second = ClockAnchor::new(
        stamp(1, 2, 1_000_001_000, 1),
        stamp(10, 3, 1_000_005_000, 1),
    );
    ClockSynchronization::new(
        99,
        first,
        second,
        TimeInterval::new(1_001, 1_000_000_999).unwrap(),
        max_drift_ppb,
        2,
        max_uncertainty_ns,
    )
    .unwrap()
}

#[test]
fn two_explicit_anchors_map_with_bounded_uncertainty() {
    let calibration = calibration(1, 20);
    assert_eq!(calibration.synchronization_id(), 99);
    assert_eq!(calibration.measured_drift_ppb(), 0);
    let mapped = calibration.map(stamp(1, 2, 500_001_000, 2)).unwrap();
    assert_eq!(mapped.clock_id(), 10);
    assert_eq!(mapped.restart_epoch(), 3);
    assert_eq!(mapped.ticks_ns(), 500_005_000);
    assert_eq!(mapped.uncertainty_ns(), 7);
    assert_eq!(calibration.model_error_bound_ns(), 2);
}

#[test]
fn absent_synchronization_is_not_guessed() {
    assert_eq!(
        map_synchronized(None, stamp(1, 2, 5_000, 0)),
        Err(SynchronizationError::UnknownSynchronization)
    );
}

#[test]
fn wrong_source_domain_or_restart_epoch_is_rejected() {
    let calibration = calibration(1, 20);
    assert_eq!(
        calibration.map(stamp(2, 2, 500_000_000, 0)),
        Err(SynchronizationError::Contract(
            ContractError::ClockDomainMismatch
        ))
    );
    assert_eq!(
        calibration.map(stamp(1, 4, 500_000_000, 0)),
        Err(SynchronizationError::Contract(
            ContractError::ClockDomainMismatch
        ))
    );
}

#[test]
fn rejects_invalid_anchor_domains_order_and_clock_pairing() {
    let source = stamp(1, 0, 1_000, 0);
    let reference = stamp(10, 0, 5_000, 0);
    let interval = TimeInterval::new(1_000, 2_000).unwrap();
    let wrong_source = ClockAnchor::new(stamp(2, 0, 2_000, 0), stamp(10, 0, 6_000, 0));
    assert_eq!(
        ClockSynchronization::new(
            1,
            ClockAnchor::new(source, reference),
            wrong_source,
            interval,
            0,
            0,
            10,
        ),
        Err(SynchronizationError::InvalidAnchors)
    );

    let reversed = ClockAnchor::new(stamp(1, 0, 500, 0), stamp(10, 0, 4_000, 0));
    assert_eq!(
        ClockSynchronization::new(
            1,
            ClockAnchor::new(source, reference),
            reversed,
            interval,
            0,
            0,
            10,
        ),
        Err(SynchronizationError::InvalidAnchors)
    );
}

#[test]
fn drift_limit_and_validity_interval_are_enforced() {
    assert_eq!(calibration(0, 20).measured_drift_ppb(), 0);
    let first = ClockAnchor::new(stamp(1, 0, 0, 0), stamp(2, 0, 0, 0));
    let second = ClockAnchor::new(stamp(1, 0, 1_000_000, 0), stamp(2, 0, 1_001_000, 0));
    assert_eq!(
        ClockSynchronization::new(
            1,
            first,
            second,
            TimeInterval::new(0, 1_000_000).unwrap(),
            500,
            0,
            10,
        ),
        Err(SynchronizationError::ExcessiveDrift)
    );
    let calibration = calibration(1, 20);
    assert_eq!(
        calibration.map(stamp(1, 2, 1_000_001_001, 0)),
        Err(SynchronizationError::OutsideValidity)
    );
}

#[test]
fn uncertainty_budget_rejects_uncertain_inputs() {
    let calibration = calibration(1, 20);
    assert_eq!(
        calibration.map(stamp(1, 2, 500_001_000, 20)),
        Err(SynchronizationError::UncertaintyExceeded)
    );
}

#[test]
fn caller_model_error_is_included_in_the_reported_bound() {
    let first = ClockAnchor::new(stamp(1, 2, 1_000, 1), stamp(10, 3, 5_000, 1));
    let second = ClockAnchor::new(
        stamp(1, 2, 1_000_001_000, 1),
        stamp(10, 3, 1_000_005_000, 1),
    );
    let calibration = ClockSynchronization::new(
        99,
        first,
        second,
        TimeInterval::new(1_001, 1_000_000_999).unwrap(),
        0,
        17,
        20,
    )
    .unwrap();
    let mapped = calibration.map(stamp(1, 2, 500_001_000, 0)).unwrap();
    assert_eq!(mapped.uncertainty_ns(), 20);

    assert_eq!(
        ClockSynchronization::new(
            99,
            first,
            second,
            TimeInterval::new(1_001, 1_000_000_999).unwrap(),
            0,
            18,
            20,
        ),
        Err(SynchronizationError::UncertaintyExceeded)
    );
}

#[test]
fn source_clock_anchor_and_input_uncertainty_scale_into_reference_units() {
    let first = ClockAnchor::new(stamp(1, 0, 100, 2), stamp(2, 0, 200, 1));
    let second = ClockAnchor::new(stamp(1, 0, 1_100, 2), stamp(2, 0, 2_200, 1));
    let calibration = ClockSynchronization::new(
        1,
        first,
        second,
        TimeInterval::new(102, 1_098).unwrap(),
        1_000_000_000,
        2,
        20,
    )
    .unwrap();

    let mapped = calibration.map(stamp(1, 0, 600, 3)).unwrap();
    // 5 ns anchor bound + 6 ns scaled input bound + 2 ns model + 1 ns rounding.
    assert_eq!(mapped.uncertainty_ns(), 14);
}

#[test]
fn constructor_rejects_anchor_uncertainty_sum_overflow() {
    let first = ClockAnchor::new(stamp(1, 0, 0, u64::MAX - 1), stamp(2, 0, 0, u64::MAX - 1));
    let second = ClockAnchor::new(stamp(1, 0, u64::MAX, 0), stamp(2, 0, u64::MAX, 0));
    assert_eq!(
        ClockSynchronization::new(
            1,
            first,
            second,
            TimeInterval::new(u64::MAX - 1, u64::MAX).unwrap(),
            0,
            0,
            u64::MAX,
        ),
        Err(SynchronizationError::Contract(
            ContractError::ArithmeticOverflow
        ))
    );
}

#[test]
fn mapping_rejects_target_timestamp_overflow() {
    let first = ClockAnchor::new(stamp(1, 0, 0, 0), stamp(2, 0, u64::MAX - 100, 0));
    let second = ClockAnchor::new(stamp(1, 0, 100, 0), stamp(2, 0, u64::MAX, 0));
    let calibration = ClockSynchronization::new(
        1,
        first,
        second,
        TimeInterval::new(0, 100).unwrap(),
        10_000_000,
        0,
        10,
    )
    .unwrap();
    assert_eq!(
        calibration.map(stamp(1, 0, 100, 0)),
        Err(SynchronizationError::Contract(ContractError::InvalidTime))
    );
}

#[test]
fn uncertain_source_anchors_exclude_possible_extrapolation() {
    let first = ClockAnchor::new(stamp(1, 0, 1_000, 100), stamp(2, 0, 1_000, 0));
    let second = ClockAnchor::new(stamp(1, 0, 2_000, 100), stamp(2, 0, 2_000, 0));
    // Actual anchors may be 1100 -> 1000 and 1900 -> 2000. Then x=1000
    // maps to 875, outside the nominal mapping's 1000 +/- 101 bound.
    assert_eq!(
        ClockSynchronization::new(
            1,
            first,
            second,
            TimeInterval::new(1_000, 2_000).unwrap(),
            0,
            0,
            101
        ),
        Err(SynchronizationError::OutsideValidity)
    );
    let calibration = ClockSynchronization::new(
        1,
        first,
        second,
        TimeInterval::new(1_100, 1_900).unwrap(),
        0,
        0,
        101,
    )
    .unwrap();
    assert_eq!(
        calibration.map(stamp(1, 0, 1_000, 0)),
        Err(SynchronizationError::OutsideValidity)
    );
    for ticks in [1_100u64, 1_500, 1_900] {
        let mapped = calibration.map(stamp(1, 0, ticks, 0)).unwrap();
        let actual = ticks * 5 / 4 - 375;
        assert!(mapped.interval().lower_ns() <= actual);
        assert!(mapped.interval().upper_ns() >= actual);
    }
    assert_eq!(
        calibration.map(stamp(1, 0, 1_100, 1)),
        Err(SynchronizationError::OutsideValidity)
    );
}
