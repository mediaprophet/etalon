use etalon_resources::{AdmissionLimits, ReservationError, ReservationPool};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::{Arc, Barrier};

struct CountingAllocator;
thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn count_allocation() {
    let _ = MEASURING.try_with(|active| {
        if active.get() {
            let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        }
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        // Forward the allocator's unchanged layout contract to System.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_allocation();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn limits() -> AdmissionLimits {
    AdmissionLimits {
        global_capacity: 1000,
        global_recorded: 200,
        partition_capacity: 150,
        partition_recorded: 0,
    }
}

#[test]
fn exact_remaining_boundary_preserves_total_class_and_error_contract() {
    let pool = ReservationPool::<6>::new();
    let field = pool.try_reserve(5, 90, limits()).unwrap();
    assert_eq!(field.bytes(), 90);
    assert_eq!(field.class(), 5);
    assert_eq!(pool.total_bytes(), 90);
    assert_eq!(pool.class_bytes(5), Some(90));
    assert_eq!(
        pool.try_reserve(2, 61, limits()).unwrap_err(),
        ReservationError::BudgetExceeded {
            requested_bytes: 61,
            available_bytes: 60
        }
    );
    assert_eq!(pool.total_bytes(), 90);
    assert_eq!(pool.class_bytes(2), Some(0));
    let frame = pool.try_reserve(2, 60, limits()).unwrap();
    assert_eq!(pool.total_bytes(), 150);
    drop(field);
    assert_eq!(pool.total_bytes(), 60);
    assert_eq!(pool.class_bytes(5), Some(0));
    assert_eq!(pool.class_bytes(2), Some(60));
    drop(frame);
    assert_eq!(pool.total_bytes(), 0);
}

#[test]
fn global_and_partition_recorded_usage_independently_limit_admission() {
    for l in [
        AdmissionLimits {
            global_capacity: 100,
            global_recorded: 91,
            partition_capacity: 100,
            partition_recorded: 0,
        },
        AdmissionLimits {
            global_capacity: 100,
            global_recorded: 0,
            partition_capacity: 100,
            partition_recorded: 91,
        },
    ] {
        let pool = ReservationPool::<1>::new();
        let guard = pool.try_reserve(0, 9, l).unwrap();
        assert_eq!(
            pool.try_reserve(0, 1, l).unwrap_err(),
            ReservationError::BudgetExceeded {
                requested_bytes: 1,
                available_bytes: 0
            }
        );
        drop(guard);
        assert_eq!(pool.total_bytes(), 0);
    }
}

#[test]
fn invalid_classes_and_empty_pool_do_not_panic_or_mutate() {
    let pool = ReservationPool::<6>::default();
    for class in [6, usize::MAX] {
        assert_eq!(
            pool.try_reserve(class, 0, limits()).unwrap_err(),
            ReservationError::InvalidClass { class, classes: 6 }
        );
        assert_eq!(pool.class_bytes(class), None);
        assert_eq!(pool.total_bytes(), 0);
    }
    let empty = ReservationPool::<0>::new();
    assert_eq!(
        empty.try_reserve(0, 1, limits()).unwrap_err(),
        ReservationError::InvalidClass {
            class: 0,
            classes: 0
        }
    );
}

#[test]
fn zero_request_is_neutral_at_zero_capacity() {
    let pool = ReservationPool::<1>::new();
    let l = AdmissionLimits {
        global_capacity: 0,
        global_recorded: 0,
        partition_capacity: 0,
        partition_recorded: 0,
    };
    let zero = pool.try_reserve(0, 0, l).unwrap();
    assert_eq!(zero.bytes(), 0);
    assert_eq!(pool.total_bytes(), 0);
    drop(zero);
    assert_eq!(pool.class_bytes(0), Some(0));
}

#[test]
fn exact_maximum_and_overflow_fail_without_partial_accounting() {
    let pool = ReservationPool::<2>::new();
    let l = AdmissionLimits {
        global_capacity: u64::MAX,
        global_recorded: 0,
        partition_capacity: u64::MAX,
        partition_recorded: 0,
    };
    let maximum = pool.try_reserve(0, u64::MAX, l).unwrap();
    assert_eq!(
        pool.try_reserve(1, 1, l).unwrap_err(),
        ReservationError::ArithmeticOverflow {
            requested_bytes: 1,
            available_bytes: 0
        }
    );
    assert_eq!(pool.total_bytes(), u64::MAX);
    assert_eq!(pool.class_bytes(1), Some(0));
    drop(maximum);
    assert_eq!(pool.total_bytes(), 0);
    let one = pool.try_reserve(1, 1, l).unwrap();
    assert_eq!(
        pool.try_reserve(0, u64::MAX, l).unwrap_err(),
        ReservationError::ArithmeticOverflow {
            requested_bytes: u64::MAX,
            available_bytes: u64::MAX - 1
        }
    );
    assert_eq!(pool.class_bytes(0), Some(0));
    drop(one);
}

#[test]
fn shrinking_snapshot_preserves_outstanding_guards_and_refuses_new_bytes() {
    let pool = ReservationPool::<2>::new();
    let guard = pool.try_reserve(1, 90, limits()).unwrap();
    for l in [
        AdmissionLimits {
            global_capacity: 50,
            ..limits()
        },
        AdmissionLimits {
            partition_capacity: 80,
            ..limits()
        },
        AdmissionLimits {
            partition_recorded: u64::MAX,
            ..limits()
        },
    ] {
        assert_eq!(
            pool.try_reserve(0, 1, l).unwrap_err(),
            ReservationError::BudgetExceeded {
                requested_bytes: 1,
                available_bytes: 0
            }
        );
        assert_eq!(pool.total_bytes(), 90);
    }
    drop(guard);
    assert_eq!(pool.total_bytes(), 0);
}

#[test]
fn unwind_releases_only_the_aborted_scopes_guard() {
    let pool = ReservationPool::<2>::new();
    let retained = pool.try_reserve(1, 40, limits()).unwrap();
    let result = std::panic::catch_unwind(|| {
        let _temporary = pool.try_reserve(0, 70, limits()).unwrap();
        panic!("intentional independent unwind fixture");
    });
    assert!(result.is_err());
    assert_eq!(pool.total_bytes(), 40);
    assert_eq!(pool.class_bytes(0), Some(0));
    drop(retained);
    assert_eq!(pool.total_bytes(), 0);
}

#[test]
fn concurrent_admission_and_drop_never_overbook_fixed_snapshot() {
    let pool = Arc::new(ReservationPool::<2>::new());
    let start = Arc::new(Barrier::new(9));
    let admitted = Arc::new(Barrier::new(9));
    let release = Arc::new(Barrier::new(9));
    let l = AdmissionLimits {
        global_capacity: 64,
        global_recorded: 0,
        partition_capacity: 64,
        partition_recorded: 0,
    };
    std::thread::scope(|scope| {
        for index in 0..8 {
            let (pool, start, admitted, release) = (
                pool.clone(),
                start.clone(),
                admitted.clone(),
                release.clone(),
            );
            scope.spawn(move || {
                start.wait();
                let guard = pool.try_reserve(index % 2, 16, l).ok();
                admitted.wait();
                release.wait();
                drop(guard);
            });
        }
        start.wait();
        admitted.wait();
        let total_while_held = pool.total_bytes();
        let classes_while_held = pool.class_bytes(0).unwrap() + pool.class_bytes(1).unwrap();
        release.wait();
        assert_eq!(total_while_held, 64);
        assert_eq!(classes_while_held, 64);
    });
    assert_eq!(pool.total_bytes(), 0);
    assert_eq!(pool.class_bytes(0), Some(0));
    assert_eq!(pool.class_bytes(1), Some(0));
}

#[test]
fn measured_admission_rejection_and_drop_allocate_zero_heap_bytes() {
    let pool = ReservationPool::<2>::new();
    ALLOCATIONS.with(|count| count.set(0));
    MEASURING.with(|active| active.set(true));
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            MEASURING.with(|active| active.set(false));
        }
    }
    let reset = Reset;
    let control = std::hint::black_box(Box::new([7u8; 64]));
    std::hint::black_box(&control);
    drop(control);
    let control_count = ALLOCATIONS.with(Cell::get);
    assert!(
        control_count > 0,
        "allocation counter positive control failed"
    );
    ALLOCATIONS.with(|count| count.set(0));
    for _ in 0..1000 {
        let guard = pool.try_reserve(0, 100, limits()).unwrap();
        assert!(pool.try_reserve(1, 51, limits()).is_err());
        assert!(pool.try_reserve(2, 1, limits()).is_err());
        drop(guard);
    }
    drop(reset);
    let count = ALLOCATIONS.with(Cell::get);
    assert_eq!(count, 0, "generic reservation success/error/drop allocated");
}
