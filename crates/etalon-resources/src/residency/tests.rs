use super::*;

fn caps(bytes: u64) -> ResidencyPoolCaps {
    ResidencyPoolCaps {
        max_cpu_bytes: bytes,
        max_gpu_vram_bytes: bytes,
    }
}

#[test]
fn failed_resize_retains_every_observable_slot_value() {
    let mut pool = ResidencyGovernor::<3>::new(caps(100));
    pool.admit_or_evict_lru_zero_heap(1, 10, 10, false, 7)
        .unwrap();
    pool.admit_or_evict_lru_zero_heap(2, 90, 90, true, 8)
        .unwrap();
    let first = *pool.get_slot(1).unwrap();
    let second = *pool.get_slot(2).unwrap();
    assert_eq!(
        pool.admit_or_evict_lru_zero_heap(1, 20, 20, true, 9),
        Err(ResidencyError::ResidencyBudgetExceeded {
            requested_cpu: 20,
            requested_gpu: 20,
            available_cpu: 10,
            available_gpu: 10,
        })
    );
    assert_eq!(pool.get_slot(1), Some(&first));
    assert_eq!(pool.get_slot(2), Some(&second));
    assert_eq!(pool.current_cpu_bytes(), 100);
    assert_eq!(pool.current_gpu_vram_bytes(), 100);
    assert_eq!(pool.occupied_slots(), 2);
    assert_eq!(pool.pinned_slots(), 1);
}

#[test]
fn overflow_requires_eviction_and_pinned_overflow_refuses() {
    let mut pool = ResidencyGovernor::<2>::new(caps(u64::MAX));
    pool.admit_or_evict_lru_zero_heap(1, u64::MAX - 1, u64::MAX - 1, false, 1)
        .unwrap();
    pool.admit_or_evict_lru_zero_heap(2, 2, 2, false, 2)
        .unwrap();
    assert!(!pool.contains(1));
    assert_eq!(pool.current_cpu_bytes(), 2);
    pool.release_resource(2).unwrap();
    pool.admit_or_evict_lru_zero_heap(1, u64::MAX - 1, u64::MAX - 1, true, 1)
        .unwrap();
    assert!(pool
        .admit_or_evict_lru_zero_heap(2, 2, 2, false, 2)
        .is_err());
    assert!(pool.contains(1));
    assert_eq!(pool.current_gpu_vram_bytes(), u64::MAX - 1);
}

#[test]
fn max_tick_and_ties_choose_first_slot() {
    let mut pool = ResidencyGovernor::<2>::new(caps(100));
    pool.admit_or_evict_lru_zero_heap(1, 1, 1, false, u64::MAX)
        .unwrap();
    pool.admit_or_evict_lru_zero_heap(2, 1, 1, false, u64::MAX)
        .unwrap();
    assert_eq!(pool.admit_or_evict_lru_zero_heap(3, 1, 1, false, 0), Ok(0));
    assert!(!pool.contains(1));
    assert!(pool.contains(2));
}

#[test]
fn same_size_promotes_pin_resize_replaces_pin() {
    let mut pool = ResidencyGovernor::<1>::new(caps(100));
    pool.admit_or_evict_lru_zero_heap(0, 0, 0, false, 1)
        .unwrap();
    pool.admit_or_evict_lru_zero_heap(0, 0, 0, true, 2).unwrap();
    pool.admit_or_evict_lru_zero_heap(0, 0, 0, false, 3)
        .unwrap();
    assert!(pool.get_slot(0).unwrap().pinned);
    assert_eq!(pool.get_slot(0).unwrap().last_used_tick, 3);
    pool.admit_or_evict_lru_zero_heap(0, 1, 2, false, 4)
        .unwrap();
    assert!(!pool.get_slot(0).unwrap().pinned);
    assert_eq!(pool.current_gpu_vram_bytes(), 2);
}

#[test]
fn explicit_release_and_cancel_override_pin() {
    let mut pool = ResidencyGovernor::<3>::new(caps(100));
    pool.admit_or_evict_lru_zero_heap(1, 10, 20, true, 1)
        .unwrap();
    pool.admit_or_evict_lru_zero_heap(2, 10, 20, false, 2)
        .unwrap();
    pool.admit_or_evict_lru_zero_heap(3, 10, 20, true, 3)
        .unwrap();
    assert_eq!(pool.cancel_all_unpinned(), 1);
    assert!(pool.contains(1) && pool.contains(3));
    pool.release_resource(1).unwrap();
    pool.cancel_pending_hydration(3).unwrap();
    assert_eq!(pool.current_cpu_bytes(), 0);
    assert_eq!(pool.current_gpu_vram_bytes(), 0);
}

#[test]
fn empty_capacity_and_missing_resource_errors() {
    let mut empty = ResidencyGovernor::<0>::new(caps(0));
    assert_eq!(
        empty.admit_or_evict_lru_zero_heap(0, 0, 0, false, 0),
        Err(ResidencyError::SlotCapacityFull)
    );
    assert_eq!(
        empty.release_resource(0),
        Err(ResidencyError::ResourceNotFound(0))
    );
    assert_eq!(
        empty.pin_resource(0, true),
        Err(ResidencyError::ResourceNotFound(0))
    );
    assert!(!empty.touch(0, 0));
    assert_eq!(empty.cancel_all_unpinned(), 0);
}
