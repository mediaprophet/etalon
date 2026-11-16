use super::*;
use core::cell::Cell;

#[test]
fn indefinite_wait_continues_past_signed_gl_timeout() {
    let mut calls = 0;
    let result = wait_with_deadline(
        None,
        || Duration::MAX,
        |ns| {
            assert_eq!(ns, i32::MAX);
            calls += 1;
            Ok::<_, ()>(calls == 4)
        },
    );
    assert_eq!(result, Ok(true));
    assert_eq!(calls, 4);
}
#[test]
fn finite_long_wait_uses_one_total_deadline() {
    let elapsed = Cell::new(Duration::ZERO);
    let mut requested = [0; 5];
    let mut count = 0;
    let result = wait_with_deadline(
        Some(Duration::from_secs(10)),
        || elapsed.get(),
        |ns| {
            requested[count] = ns;
            count += 1;
            elapsed.set(elapsed.get() + Duration::from_nanos(ns as u64));
            Ok::<_, ()>(false)
        },
    );
    assert_eq!(result, Ok(false));
    assert_eq!(count, 5);
    assert_eq!(
        requested,
        [i32::MAX, i32::MAX, i32::MAX, i32::MAX, 1_410_065_412]
    );
    assert_eq!(elapsed.get(), Duration::from_secs(10));
}
#[test]
fn wall_clock_overhead_reduces_remaining_budget() {
    let elapsed = Cell::new(Duration::from_millis(2));
    let mut calls = 0;
    let result = wait_with_deadline(
        Some(Duration::from_millis(10)),
        || elapsed.get(),
        |ns| {
            calls += 1;
            assert_eq!(ns, 8_000_000);
            elapsed.set(Duration::from_millis(11));
            Ok::<_, ()>(false)
        },
    );
    assert_eq!(result, Ok(false));
    assert_eq!(calls, 1);
}
#[test]
fn early_timeout_retries_only_remaining_duration() {
    let elapsed = Cell::new(Duration::ZERO);
    let mut calls = 0;
    let result = wait_with_deadline(
        Some(Duration::from_millis(10)),
        || elapsed.get(),
        |ns| {
            calls += 1;
            if calls == 1 {
                assert_eq!(ns, 10_000_000);
                elapsed.set(Duration::from_millis(3));
            } else {
                assert_eq!(ns, 7_000_000);
                elapsed.set(Duration::from_millis(10));
            }
            Ok::<_, ()>(false)
        },
    );
    assert_eq!(result, Ok(false));
    assert_eq!(calls, 2);
}
#[test]
fn zero_timeout_is_one_nonblocking_probe() {
    for signalled in [false, true] {
        let mut calls = 0;
        assert_eq!(
            wait_with_deadline(
                Some(Duration::ZERO),
                || Duration::ZERO,
                |ns| {
                    calls += 1;
                    assert_eq!(ns, 0);
                    Ok::<_, ()>(signalled)
                }
            ),
            Ok(signalled)
        );
        assert_eq!(calls, 1);
    }
}
#[test]
fn expired_initial_deadline_does_not_add_a_blocking_wait() {
    let mut calls = 0;
    assert_eq!(
        wait_with_deadline(
            Some(Duration::from_nanos(1)),
            || Duration::MAX,
            |ns| {
                calls += 1;
                assert_eq!(ns, 0);
                Ok::<_, ()>(false)
            }
        ),
        Ok(false)
    );
    assert_eq!(calls, 1);
}
#[test]
fn maximum_duration_never_overflows_or_becomes_indefinite() {
    let mut calls = 0;
    assert_eq!(
        wait_with_deadline(
            Some(Duration::MAX),
            || Duration::ZERO,
            |ns| {
                calls += 1;
                assert_eq!(ns, i32::MAX);
                Ok::<_, ()>(true)
            }
        ),
        Ok(true)
    );
    assert_eq!(calls, 1);
}
#[test]
fn driver_failure_is_propagated_without_retry_or_timeout_conversion() {
    let mut calls = 0;
    assert_eq!(
        wait_with_deadline(
            None,
            || Duration::ZERO,
            |_| {
                calls += 1;
                Err::<bool, _>("driver failure")
            }
        ),
        Err("driver failure")
    );
    assert_eq!(calls, 1);
}
#[test]
fn success_ends_wait_without_an_extra_probe() {
    let mut calls = 0;
    assert_eq!(
        wait_with_deadline(
            Some(Duration::from_secs(1)),
            || Duration::ZERO,
            |_| {
                calls += 1;
                Ok::<_, ()>(true)
            }
        ),
        Ok(true)
    );
    assert_eq!(calls, 1);
}

#[test]
fn maximum_finite_duration_still_expires() {
    let checks = Cell::new(0);
    let mut calls = 0;
    let result = wait_with_deadline(
        Some(Duration::MAX),
        || {
            let previous = checks.get();
            checks.set(previous + 1);
            if previous == 0 {
                Duration::ZERO
            } else {
                Duration::MAX
            }
        },
        |ns| {
            calls += 1;
            assert_eq!(ns, i32::MAX);
            Ok::<_, ()>(false)
        },
    );
    assert_eq!(result, Ok(false));
    assert_eq!(calls, 1);
}
