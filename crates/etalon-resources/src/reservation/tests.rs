use super::*;
use std::sync::{Arc, Barrier};
use std::vec::Vec;

const LIMITS: AdmissionLimits = AdmissionLimits {
    global_capacity: 1_000,
    global_recorded: 200,
    partition_capacity: 150,
    partition_recorded: 0,
};

#[test]
fn exact_partition_capacity_and_failed_admission_preserve_accounting() {
    let pool = ReservationPool::<6>::new();
    let first = pool.try_reserve(5, 90, LIMITS).unwrap();
    assert_eq!(pool.total_bytes(), 90);
    assert_eq!(pool.class_bytes(5), Some(90));
    assert_eq!(
        pool.try_reserve(2, 61, LIMITS).unwrap_err(),
        ReservationError::BudgetExceeded {
            requested_bytes: 61,
            available_bytes: 60,
        }
    );
    assert_eq!(pool.total_bytes(), 90);
    assert_eq!(pool.class_bytes(2), Some(0));
    let exact = pool.try_reserve(2, 60, LIMITS).unwrap();
    assert_eq!(pool.total_bytes(), 150);
    drop(first);
    assert_eq!(pool.total_bytes(), 60);
    drop(exact);
    assert_eq!(pool.total_bytes(), 0);
}

#[test]
fn global_and_partition_recorded_usage_both_limit_capacity() {
    let pool = ReservationPool::<1>::new();
    for limits in [
        AdmissionLimits {
            global_recorded: 990,
            ..LIMITS
        },
        AdmissionLimits {
            partition_recorded: 140,
            ..LIMITS
        },
    ] {
        let exact = pool.try_reserve(0, 10, limits).unwrap();
        assert_eq!(
            pool.try_reserve(0, 1, limits).unwrap_err(),
            ReservationError::BudgetExceeded {
                requested_bytes: 1,
                available_bytes: 0
            }
        );
        drop(exact);
    }
}

#[test]
fn zero_reservations_and_all_class_indices() {
    let pool = ReservationPool::<6>::default();
    for class in 0..6 {
        let guard = pool.try_reserve(class, 0, LIMITS).unwrap();
        assert_eq!(guard.class(), class);
        assert_eq!(guard.bytes(), 0);
        assert_eq!(pool.class_bytes(class), Some(0));
        drop(guard);
    }
    assert_eq!(pool.total_bytes(), 0);
    for class in [6, usize::MAX] {
        assert_eq!(pool.class_bytes(class), None);
        assert_eq!(
            pool.try_reserve(class, 0, LIMITS).unwrap_err(),
            ReservationError::InvalidClass { class, classes: 6 }
        );
    }
    assert!(ReservationPool::<0>::new()
        .try_reserve(0, 0, LIMITS)
        .is_err());
}

#[test]
fn arithmetic_overflow_refuses_without_mutation() {
    let limits = AdmissionLimits {
        global_capacity: u64::MAX,
        global_recorded: 0,
        partition_capacity: u64::MAX,
        partition_recorded: 0,
    };
    let pool = ReservationPool::<1>::new();
    let one = pool.try_reserve(0, 1, limits).unwrap();
    assert_eq!(
        pool.try_reserve(0, u64::MAX, limits).unwrap_err(),
        ReservationError::ArithmeticOverflow {
            requested_bytes: u64::MAX,
            available_bytes: u64::MAX - 1
        }
    );
    assert_eq!(pool.total_bytes(), 1);
    drop(one);
    let maximum = pool.try_reserve(0, u64::MAX, limits).unwrap();
    assert_eq!(pool.class_bytes(0), Some(u64::MAX));
    drop(maximum);
    assert_eq!(pool.total_bytes(), 0);
}

#[test]
fn guards_remain_releasable_after_limit_shrink() {
    let pool = ReservationPool::<1>::new();
    let guard = pool.try_reserve(0, 100, LIMITS).unwrap();
    let shrunk = AdmissionLimits {
        partition_capacity: 50,
        ..LIMITS
    };
    assert_eq!(
        pool.try_reserve(0, 1, shrunk).unwrap_err(),
        ReservationError::BudgetExceeded {
            requested_bytes: 1,
            available_bytes: 0
        }
    );
    assert_eq!(pool.total_bytes(), 100);
    drop(guard);
    assert_eq!(pool.try_reserve(0, 50, shrunk).unwrap().bytes(), 50);
    assert_eq!(pool.total_bytes(), 0);
}

#[test]
fn moved_guard_releases_once_and_unwind_releases() {
    let pool = ReservationPool::<1>::new();
    let first = pool.try_reserve(0, 20, LIMITS).unwrap();
    let moved = first;
    drop(moved);
    assert_eq!(pool.total_bytes(), 0);
    let result = std::panic::catch_unwind(|| {
        let _guard = pool.try_reserve(0, 30, LIMITS).unwrap();
        panic!("test unwind");
    });
    assert!(result.is_err());
    assert_eq!(pool.total_bytes(), 0);
    assert_eq!(pool.class_bytes(0), Some(0));
}

#[test]
fn concurrent_reservations_cannot_exceed_fixed_snapshot() {
    let pool = Arc::new(ReservationPool::<6>::new());
    let held = Arc::new(Barrier::new(17));
    let release = Arc::new(Barrier::new(17));
    let mut workers = Vec::new();
    for class in 0..16 {
        let pool = Arc::clone(&pool);
        let held = Arc::clone(&held);
        let release = Arc::clone(&release);
        workers.push(std::thread::spawn(move || {
            let reservation = pool.try_reserve(class % 6, 10, LIMITS);
            held.wait();
            release.wait();
            reservation.is_ok()
        }));
    }
    held.wait();
    let held_total = pool.total_bytes();
    let held_classes = (0..6)
        .map(|class| pool.class_bytes(class).unwrap())
        .sum::<u64>();
    release.wait();
    let successes = workers
        .into_iter()
        .map(|worker| usize::from(worker.join().unwrap()))
        .sum::<usize>();
    assert_eq!(held_total, 150);
    assert_eq!(held_classes, 150);
    assert_eq!(successes, 15);
    assert_eq!(pool.total_bytes(), 0);
}
