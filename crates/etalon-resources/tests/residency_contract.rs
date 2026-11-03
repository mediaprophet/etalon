//! Independent preservation and safety-uplift cases frozen against Qualia's governor.
use etalon_resources::residency::{ResidencyError, ResidencyGovernor, ResidencyPoolCaps};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

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

fn caps(cpu: u64, gpu: u64) -> ResidencyPoolCaps {
    ResidencyPoolCaps {
        max_cpu_bytes: cpu,
        max_gpu_vram_bytes: gpu,
    }
}

#[test]
fn failed_resize_preserves_target_other_slots_totals_pin_and_tick() {
    for pin_target in [false, true] {
        let mut pool = ResidencyGovernor::<3>::new(caps(100, 200));
        pool.admit_or_evict_lru_zero_heap(1, 10, 20, pin_target, 7)
            .unwrap();
        pool.admit_or_evict_lru_zero_heap(2, 90, 180, true, 8)
            .unwrap();
        let target = *pool.get_slot(1).unwrap();
        let other = *pool.get_slot(2).unwrap();
        assert_eq!(
            pool.admit_or_evict_lru_zero_heap(1, 20, 21, false, 99),
            Err(ResidencyError::ResidencyBudgetExceeded {
                requested_cpu: 20,
                requested_gpu: 21,
                available_cpu: 10,
                available_gpu: 20
            })
        );
        assert_eq!(*pool.get_slot(1).unwrap(), target);
        assert_eq!(*pool.get_slot(2).unwrap(), other);
        assert_eq!(
            (pool.current_cpu_bytes(), pool.current_gpu_vram_bytes()),
            (100, 200)
        );
        assert_eq!(pool.occupied_slots(), 2);
        assert_eq!(pool.pinned_slots(), 1 + usize::from(pin_target));
    }
}

#[test]
fn aggregate_overflow_evicts_eligible_but_never_overadmits_pinned() {
    for gpu_axis in [false, true] {
        for pinned in [false, true] {
            let mut pool = ResidencyGovernor::<2>::new(caps(u64::MAX, u64::MAX));
            let bytes = if gpu_axis {
                (0, u64::MAX - 1)
            } else {
                (u64::MAX - 1, 0)
            };
            pool.admit_or_evict_lru_zero_heap(1, bytes.0, bytes.1, pinned, 1)
                .unwrap();
            let request = if gpu_axis { (0, 2) } else { (2, 0) };
            let result = pool.admit_or_evict_lru_zero_heap(2, request.0, request.1, false, 2);
            if pinned {
                assert!(matches!(
                    result,
                    Err(ResidencyError::ResidencyBudgetExceeded { .. })
                ));
                assert!(pool.contains(1));
                assert!(!pool.contains(2));
                assert_eq!(
                    (pool.current_cpu_bytes(), pool.current_gpu_vram_bytes()),
                    bytes
                );
            } else {
                assert_eq!(result, Ok(0));
                assert!(!pool.contains(1));
                assert_eq!(
                    (pool.current_cpu_bytes(), pool.current_gpu_vram_bytes()),
                    request
                );
            }
        }
    }
}

#[test]
fn maximum_ticks_are_evictable_and_ties_choose_lowest_slot() {
    let mut pool = ResidencyGovernor::<2>::new(caps(2, 2));
    assert_eq!(
        pool.admit_or_evict_lru_zero_heap(10, 1, 1, false, u64::MAX),
        Ok(0)
    );
    assert_eq!(
        pool.admit_or_evict_lru_zero_heap(20, 1, 1, false, u64::MAX),
        Ok(1)
    );
    assert_eq!(pool.admit_or_evict_lru_zero_heap(30, 1, 1, false, 0), Ok(0));
    assert!(!pool.contains(10));
    assert!(pool.contains(20));
    assert!(pool.touch(30, u64::MAX));
    assert_eq!(pool.admit_or_evict_lru_zero_heap(40, 1, 1, false, 3), Ok(0));
    assert!(!pool.contains(30));
    assert!(pool.contains(20));
}

#[test]
fn pin_updates_keep_distinct_same_size_resize_and_explicit_release_semantics() {
    let mut pool = ResidencyGovernor::<3>::new(caps(100, 100));
    pool.admit_or_evict_lru_zero_heap(1, 10, 10, false, 1)
        .unwrap();
    pool.admit_or_evict_lru_zero_heap(1, 10, 10, true, 2)
        .unwrap();
    pool.admit_or_evict_lru_zero_heap(1, 10, 10, false, 3)
        .unwrap();
    assert!(pool.get_slot(1).unwrap().pinned);
    assert_eq!(pool.get_slot(1).unwrap().last_used_tick, 3);
    pool.admit_or_evict_lru_zero_heap(1, 11, 10, false, 4)
        .unwrap();
    assert!(!pool.get_slot(1).unwrap().pinned);
    pool.admit_or_evict_lru_zero_heap(2, 10, 10, true, 5)
        .unwrap();
    assert_eq!(pool.cancel_all_unpinned(), 1);
    assert!(pool.contains(2));
    pool.cancel_pending_hydration(2).unwrap();
    pool.admit_or_evict_lru_zero_heap(3, 10, 10, true, 6)
        .unwrap();
    pool.release_resource(3).unwrap();
    assert_eq!(pool.occupied_slots(), 0);
    assert_eq!(
        (pool.current_cpu_bytes(), pool.current_gpu_vram_bytes()),
        (0, 0)
    );
}

#[test]
fn zero_slots_zero_bytes_hash_zero_and_pinned_slot_exhaustion() {
    let mut empty = ResidencyGovernor::<0>::new(caps(0, 0));
    assert_eq!(
        empty.admit_or_evict_lru_zero_heap(0, 0, 0, false, 0),
        Err(ResidencyError::SlotCapacityFull)
    );
    let mut pool = ResidencyGovernor::<1>::new(caps(0, 0));
    assert_eq!(pool.admit_or_evict_lru_zero_heap(0, 0, 0, true, 1), Ok(0));
    assert_eq!(pool.admit_or_evict_lru_zero_heap(0, 0, 0, false, 2), Ok(0));
    assert!(pool.get_slot(0).unwrap().pinned);
    assert_eq!(
        pool.admit_or_evict_lru_zero_heap(1, 0, 0, false, 3),
        Err(ResidencyError::SlotCapacityFull)
    );
    assert_eq!(pool.cancel_all_unpinned(), 0);
    assert!(!pool.touch(9, 4));
    pool.pin_resource(0, false).unwrap();
    assert_eq!(pool.admit_or_evict_lru_zero_heap(1, 0, 0, false, 5), Ok(0));
    assert!(!pool.contains(0));
    assert_eq!(pool.occupied_slots(), 1);
}

#[test]
fn each_budget_axis_and_exact_error_display_are_preserved() {
    let mut pool = ResidencyGovernor::<2>::new(caps(100, 200));
    pool.admit_or_evict_lru_zero_heap(1, 40, 70, true, 1)
        .unwrap();
    for (cpu, gpu, available_cpu, available_gpu) in [
        (101, 0, 60, 130),
        (0, 201, 60, 130),
        (61, 0, 60, 130),
        (0, 131, 60, 130),
    ] {
        assert_eq!(
            pool.admit_or_evict_lru_zero_heap(2, cpu, gpu, false, 2),
            Err(ResidencyError::ResidencyBudgetExceeded {
                requested_cpu: cpu,
                requested_gpu: gpu,
                available_cpu,
                available_gpu
            })
        );
    }
    let error = ResidencyError::ResidencyBudgetExceeded {
        requested_cpu: 61,
        requested_gpu: 3,
        available_cpu: 60,
        available_gpu: 130,
    };
    assert_eq!(
        error.to_string(),
        "residency budget exceeded: requested (CPU 61, GPU 3) > available (CPU 60, GPU 130)"
    );
    assert_eq!(
        ResidencyError::SlotCapacityFull.to_string(),
        "residency governor slot capacity full"
    );
    assert_eq!(
        ResidencyError::ResourceNotFound(15).to_string(),
        "resource 0x000000000000000f not found in residency pool"
    );
    assert_eq!(
        ResidencyError::ResourcePinned(15).to_string(),
        "resource 0x000000000000000f is pinned and cannot be evicted"
    );
    assert_eq!(
        ResidencyError::Cancelled.to_string(),
        "residency admission cancelled"
    );
    assert_eq!(
        pool.release_resource(9),
        Err(ResidencyError::ResourceNotFound(9))
    );
    assert_eq!(
        pool.pin_resource(9, true),
        Err(ResidencyError::ResourceNotFound(9))
    );
}

#[test]
fn measured_hot_paths_and_refusals_allocate_nothing_with_positive_control() {
    ALLOCATIONS.with(|count| count.set(0));
    MEASURING.with(|active| active.set(true));
    let control = std::hint::black_box(Box::new([7u8; 32]));
    std::hint::black_box(&control);
    drop(control);
    MEASURING.with(|active| active.set(false));
    assert!(
        ALLOCATIONS.with(Cell::get) > 0,
        "allocator positive control"
    );
    let mut pool = ResidencyGovernor::<2>::new(caps(100, 100));
    ALLOCATIONS.with(|count| count.set(0));
    MEASURING.with(|active| active.set(true));
    let first = pool.admit_or_evict_lru_zero_heap(1, 90, 90, true, 1);
    let second = pool.admit_or_evict_lru_zero_heap(2, 10, 10, false, 2);
    let refused = pool.admit_or_evict_lru_zero_heap(2, 11, 11, false, 3);
    let touched = pool.touch(2, 4);
    let evicted = pool.cancel_all_unpinned();
    let released = pool.release_resource(1);
    let missing = pool.cancel_pending_hydration(3);
    MEASURING.with(|active| active.set(false));
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
    assert!(first.is_ok() && second.is_ok() && refused.is_err() && touched);
    assert_eq!(evicted, 1);
    assert!(released.is_ok() && missing.is_err());
    assert_eq!(pool.occupied_slots(), 0);
}
