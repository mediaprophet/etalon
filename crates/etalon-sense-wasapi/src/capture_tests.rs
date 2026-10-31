use super::*;

#[test]
fn source_hash_is_stable_and_nonzero() {
    assert_eq!(
        stable_source_id(b"endpoint-id"),
        stable_source_id(b"endpoint-id")
    );
    assert_ne!(stable_source_id(b"endpoint-id"), 0);
}

#[test]
fn partial_packet_consumption_is_rejected_without_a_remainder_buffer() {
    assert!(packet_fits_whole(160, 480));
    assert!(packet_fits_whole(480, 480));
    assert!(!packet_fits_whole(481, 480));
    assert!(!packet_fits_whole(0, 480));
}

#[test]
fn total_capture_deadline_bounds_each_wait_using_host_clock() {
    let start = ClockStamp::new(7, 2, 1_000_000_000, 0).unwrap();
    let immediate = ClockStamp::new(7, 2, 1_000_000_000, 0).unwrap();
    let halfway = ClockStamp::new(7, 2, 1_025_000_000, 0).unwrap();
    let deadline = ClockStamp::new(7, 2, 1_050_000_000, 0).unwrap();

    assert_eq!(capture_wait_timeout_ms(start, immediate, 50, 1_000), Ok(50));
    assert_eq!(capture_wait_timeout_ms(start, halfway, 50, 1_000), Ok(25));
    assert_eq!(
        capture_wait_timeout_ms(start, deadline, 50, 1_000),
        Err(AdapterError::BackendFailure)
    );
    assert_eq!(capture_wait_timeout_ms(start, immediate, 50, 10), Ok(10));
    assert_eq!(
        capture_wait_timeout_ms(
            start,
            ClockStamp::new(8, 2, 1_000_000_000, 0).unwrap(),
            50,
            10
        ),
        Err(AdapterError::ClockDomainMismatch),
    );
}

#[test]
fn clock_uncertainty_can_expire_capture_budget_early_but_never_extend_it() {
    let start = ClockStamp::new(7, 2, 1_000_000_000, 5_000_000).unwrap();
    let now = ClockStamp::new(7, 2, 1_040_000_000, 6_000_000).unwrap();
    assert_eq!(
        capture_wait_timeout_ms(start, now, 50, 1_000),
        Err(AdapterError::BackendFailure)
    );
}

#[test]
fn repeated_wait_timeouts_detect_a_stalled_or_regressed_host_clock() {
    let before = ClockStamp::new(7, 2, 1_000, 0).unwrap();
    let same = ClockStamp::new(7, 2, 1_000, 0).unwrap();
    let advanced = ClockStamp::new(7, 2, 1_001, 0).unwrap();
    let regressed = ClockStamp::new(7, 2, 999, 0).unwrap();

    assert_eq!(timeout_clock_advanced(before, same), Ok(false));
    assert_eq!(timeout_clock_advanced(before, advanced), Ok(true));
    assert_eq!(
        timeout_clock_advanced(before, regressed),
        Err(AdapterError::BackendFailure)
    );
}

#[test]
fn failed_stop_keeps_session_and_generation_for_retry() {
    let mut active = Some(42u8);
    let mut generation = 7u32;
    let mut faulted = false;

    assert_eq!(
        finish_stop_state(
            Err(AdapterError::BackendFailure),
            &mut active,
            &mut generation,
            &mut faulted,
        ),
        Err(AdapterError::BackendFailure),
    );
    assert_eq!(active, Some(42));
    assert_eq!(generation, 7);
    assert!(faulted);

    assert_eq!(
        finish_stop_state(Ok(()), &mut active, &mut generation, &mut faulted),
        Ok(()),
    );
    assert_eq!(active, None);
    assert_eq!(generation, 8);
    assert!(!faulted);
}

#[test]
fn confirmed_stop_clears_session_even_when_generation_is_exhausted() {
    let mut active = Some(42u8);
    let mut generation = u32::MAX;
    let mut faulted = false;

    assert_eq!(
        finish_stop_state(Ok(()), &mut active, &mut generation, &mut faulted),
        Err(AdapterError::BackendFailure),
    );
    assert_eq!(active, None);
    assert_eq!(generation, u32::MAX);
    assert!(faulted);
}
