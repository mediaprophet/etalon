//! Independent frozen-original operation sequences. Cold Vec constructors are
//! outside measured hot calls. Metadata COW is not physical copying or rollback.
use etalon_inference::runtime::{
    budget::*,
    kv::{paged::*, prefix::*},
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
struct Counter;
thread_local! {static ACTIVE:Cell<bool>=const{Cell::new(false)};static COUNT:Cell<usize>=const{Cell::new(0)};}
fn tick() {
    let _ = ACTIVE.try_with(|a| {
        if a.get() {
            let _ = COUNT.try_with(|c| c.set(c.get() + 1));
        }
    });
}
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        tick();
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        tick();
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        tick();
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;
fn id(n: u64) -> PrefixIdentity {
    PrefixIdentity {
        words: [n, 2, 3, 4],
    }
}
fn budget() -> MemoryPoolBudget {
    MemoryPoolBudget::new(
        356,
        ModelMemoryProfile::new(40, 30, 20, 10).unwrap(),
        BlockGeometry::new(16, 32).unwrap(),
        8,
        2,
    )
    .unwrap()
}
#[test]
fn budget_boundaries_error_order_and_fields_match_original() {
    assert_eq!(
        ModelMemoryProfile::new(u64::MAX, 1, 0, 0),
        Err(BudgetError::IntegerOverflow)
    );
    assert_eq!(BlockGeometry::new(0, 32), Err(BudgetError::InvalidGeometry));
    let b = budget();
    assert_eq!(b.available_kv_bytes(), 256);
    assert_eq!(b.geometry.blocks_for_tokens(0), Ok(0));
    assert_eq!(b.geometry.blocks_for_tokens(16), Ok(1));
    assert_eq!(b.geometry.blocks_for_tokens(17), Ok(2));
    assert_eq!(
        b.geometry.blocks_for_tokens(u32::MAX),
        Err(BudgetError::IntegerOverflow)
    );
    assert_eq!(
        b.reserve_with_active_cow(17, 15, 1, 1, 160),
        Ok(RequestReservation {
            input_tokens: 17,
            reserved_output_tokens: 15,
            required_blocks: 2,
            cow_transient_blocks: 1,
            reserved_bytes: 96
        })
    );
    assert_eq!(
        b.reserve_with_active_cow(17, 15, 1, 1, 161),
        Err(BudgetError::InsufficientBudget {
            required_bytes: 257,
            available_bytes: 256
        })
    );
    assert_eq!(
        b.reserve_with_active_cow(17, 15, 1, 2, 0),
        Err(BudgetError::InsufficientCowHeadroom {
            requested_blocks: 3,
            max_cow_blocks: 2
        })
    );
    assert_eq!(
        b.reserve(128, 0, 1, 0),
        Err(BudgetError::PoolExhausted {
            requested_blocks: 9,
            max_blocks: 8
        })
    );
    assert_eq!(
        b.reserve(u32::MAX, 1, 0, 0),
        Err(BudgetError::IntegerOverflow)
    );
    assert_eq!(
        b.reserve(1, 0, 0, u64::MAX),
        Err(BudgetError::IntegerOverflow)
    );
    assert_eq!(b.reserve(0, 0, 0, 256).unwrap().reserved_bytes, 0);
    assert_eq!(
        BudgetError::IntegerOverflow.to_string(),
        "arithmetic overflow in memory budget calculation"
    );
    assert_eq!(
        BudgetError::PoolExhausted {
            requested_blocks: 9,
            max_blocks: 8
        }
        .to_string(),
        "physical block pool exhausted: requested 9 blocks, capacity is 8"
    );
}
#[test]
fn pool_allocation_order_cow_exhaustion_and_release_errors() {
    let mut p = BlockPool::new(3);
    assert_eq!(
        (p.allocate(), p.allocate(), p.allocate()),
        (Ok(0), Ok(1), Ok(2))
    );
    assert_eq!(p.allocate(), Err(PoolError::Exhausted));
    p.retain(0).unwrap();
    assert_eq!(p.writable(0), Err(PoolError::Exhausted));
    assert_eq!(p.ref_count(0), Some(2));
    p.release(2).unwrap();
    assert_eq!(
        p.writable(0),
        Ok(CopyOnWrite::Copy {
            source: 0,
            destination: 2
        })
    );
    assert_eq!((p.ref_count(0), p.ref_count(2)), (Some(1), Some(1)));
    assert_eq!(p.writable(2), Ok(CopyOnWrite::Existing(2)));
    p.release(1).unwrap();
    assert_eq!(p.release(1), Err(PoolError::DoubleRelease));
    assert_eq!(p.retain(1), Err(PoolError::InvalidBlock));
    assert_eq!(p.release(3), Err(PoolError::InvalidBlock));
    assert_eq!(p.ref_count(3), None);
    assert_eq!(p.writable(INVALID_BLOCK), Ok(CopyOnWrite::Allocated(1)));
    p.release(0).unwrap();
    p.release(2).unwrap();
    assert_eq!((p.allocate(), p.allocate()), (Ok(2), Ok(0)));
    let mut empty = BlockPool::new(0);
    assert_eq!(empty.allocate(), Err(PoolError::Exhausted));
}
#[test]
fn table_fork_and_install_roll_back_retains_and_identity_keeps_tail() {
    let c = PagedKvConfig::new(2, 1, 2, 17).unwrap();
    let mut out = [77; 6];
    assert_eq!(fill_identity_block_table(&c, &mut out), Ok(4));
    assert_eq!(out, [0, 1, 2, 3, 77, 77]);
    let mut small = [77; 3];
    assert_eq!(
        fill_identity_block_table(&c, &mut small),
        Err(TableError::OutputTooSmall)
    );
    assert_eq!(small, [77; 3]);
    let mut p = BlockPool::new(4);
    let block = p.allocate().unwrap();
    let mut source = SequenceBlockTable::new(3);
    source
        .entries_mut()
        .copy_from_slice(&[block, INVALID_BLOCK, 99]);
    let mut target = SequenceBlockTable::new(3);
    assert_eq!(
        source.fork_into(&mut target, &mut p),
        Err(TableError::Pool(PoolError::InvalidBlock))
    );
    assert_eq!(target.entries(), &[INVALID_BLOCK; 3]);
    assert_eq!(p.ref_count(block), Some(1));
    assert_eq!(
        target.install_shared_prefix(&[block, 99], &mut p),
        Err(TableError::Pool(PoolError::InvalidBlock))
    );
    assert_eq!(target.entries(), &[INVALID_BLOCK; 3]);
    assert_eq!(p.ref_count(block), Some(1));
    source.entries_mut()[2] = INVALID_BLOCK;
    source.fork_into(&mut target, &mut p).unwrap();
    assert_eq!(p.ref_count(block), Some(2));
    assert_eq!(
        source.fork_into(&mut target, &mut p),
        Err(TableError::TargetNotEmpty)
    );
    assert_eq!(
        target.ensure_writable(0, &mut p),
        Ok(CopyOnWrite::Copy {
            source: block,
            destination: 1
        })
    );
    assert_eq!(source.get(0), Some(0));
    assert_eq!(target.get(0), Some(1));
    assert_eq!(
        target.ensure_writable(3, &mut p),
        Err(TableError::OutOfRange)
    );
    source.release_all(&mut p).unwrap();
    target.release_all(&mut p).unwrap();
    assert_eq!(p.free_count(), 4);
}
#[test]
fn opaque_registry_identity_replacement_and_capacity_are_preserved() {
    assert_eq!(std::mem::size_of::<PrefixIdentity>(), 32);
    assert_eq!(std::mem::align_of::<PrefixIdentity>(), 8);
    assert_eq!(
        PrefixPageSet::<2>::new(id(1), &[], 0),
        Err(RegistryError::EmptyPageSet)
    );
    assert_eq!(
        PrefixPageSet::<1>::new(id(1), &[0, 1], 0),
        Err(RegistryError::PageCapacity)
    );
    let a = PrefixPageSet::<2>::new(id(1), &[1], 16).unwrap();
    assert_eq!(a.pages, [1, INVALID_BLOCK]);
    let b = PrefixPageSet::new(id(2), &[2], 32).unwrap();
    let c = PrefixPageSet::new(id(3), &[3], 48).unwrap();
    let d = PrefixPageSet::new(id(4), &[4], 64).unwrap();
    let mut r = PrefixPageRegistry::<2, 2>::new();
    assert_eq!(r.insert_replacing(a), Ok(None));
    assert_eq!(r.insert_replacing(b), Ok(None));
    assert_eq!(r.insert_replacing(c), Ok(Some(a)));
    let updated = PrefixPageSet::new(id(2), &[5], 99).unwrap();
    assert_eq!(r.insert_replacing(updated), Ok(Some(b)));
    assert_eq!(r.insert_replacing(d), Ok(Some(updated)));
    assert!(r.get(id(1)).is_none());
    assert!(r
        .get(PrefixIdentity {
            words: [3, 2, 3, 5]
        })
        .is_none());
    assert_eq!(r.remove_entry(id(3)), Some(c));
    assert!(!r.remove(id(3)));
    assert_eq!(
        PrefixPageRegistry::<0, 2>::new().insert(a),
        Err(RegistryError::RegistryFull)
    );
}
#[test]
fn prefix_store_ownership_survives_eviction_and_failed_publication() {
    let mut p = BlockPool::new(3);
    let a = p.allocate().unwrap();
    let b = p.allocate().unwrap();
    let c = p.allocate().unwrap();
    let mut store = PrefixKvStore::<2, 2>::new();
    let mut table = SequenceBlockTable::new(2);
    store
        .publish(PrefixPageSet::new(id(1), &[a], 16).unwrap(), &mut p)
        .unwrap();
    assert_eq!(store.attach(id(1), &mut table, &mut p), Ok(Some(16)));
    p.release(a).unwrap();
    assert_eq!(p.ref_count(a), Some(2));
    store
        .publish(PrefixPageSet::new(id(2), &[b], 32).unwrap(), &mut p)
        .unwrap();
    store
        .publish(PrefixPageSet::new(id(3), &[c], 48).unwrap(), &mut p)
        .unwrap();
    assert!(store.get(id(1)).is_none());
    assert_eq!(p.ref_count(a), Some(1));
    assert_eq!(table.get(0), Some(a));
    table.release_all(&mut p).unwrap();
    assert_eq!(p.ref_count(a), Some(0));
    assert_eq!(
        store.publish(PrefixPageSet::new(id(4), &[b, 99], 9).unwrap(), &mut p),
        Err(PrefixKvError::Pool(PoolError::InvalidBlock))
    );
    assert!(store.get(id(4)).is_none());
    assert_eq!(p.ref_count(b), Some(2));
    assert_eq!(store.remove(id(2), &mut p), Ok(true));
    assert_eq!(p.ref_count(b), Some(1));
    assert_eq!(store.remove(id(2), &mut p), Ok(false));
    assert_eq!(store.attach(id(8), &mut table, &mut p), Ok(None));
    assert_eq!(
        PrefixKvStore::<0, 2>::new().publish(PrefixPageSet::new(id(9), &[b], 9).unwrap(), &mut p),
        Err(PrefixKvError::Registry(RegistryError::RegistryFull))
    );
    assert_eq!(p.ref_count(b), Some(1));
    store.remove(id(3), &mut p).unwrap();
    p.release(b).unwrap();
    p.release(c).unwrap();
    assert_eq!(p.free_count(), 3);
}
#[test]
fn radix_matching_osrp_lru_ties_and_repointing_preserve_operation_order() {
    let mut p = BlockPool::new(4);
    let a = p.allocate().unwrap();
    let b = p.allocate().unwrap();
    let c = p.allocate().unwrap();
    let mut r = RadixPrefixIndex::<3>::new();
    let parent = r.insert_child_page(None, &[1, 2], a, 10).unwrap();
    r.insert_child_page(Some(parent), &[3, 4], b, 10).unwrap();
    r.insert_child_page(None, &[8], c, 10).unwrap();
    assert_eq!(
        r.insert_child_page(None, &[9], 0, 0),
        Err(RadixError::TreeFull)
    );
    let mut pages = [99; 4];
    assert_eq!(
        r.match_longest_prefix(&[1, 2, 3, 4], 20, &mut pages),
        Ok((4, 3, 2))
    );
    assert_eq!(pages, [a, b, 99, 99]);
    assert_eq!(
        r.match_longest_prefix(&[1, 2, 3], 21, &mut pages),
        Ok((2, 2, 1))
    );
    assert_eq!(r.match_longest_prefix(&[], 22, &mut pages), Ok((0, 0, 0)));
    let mut short = [99];
    assert_eq!(
        r.match_longest_prefix(&[1, 2, 3, 4], 23, &mut short),
        Err(RadixError::BufferTooSmall)
    );
    assert_eq!(short, [a]);
    assert_eq!(r.evict_lru_leaf(&mut p), Ok(Some(c)));
    assert_eq!(r.evict_lru_leaf(&mut p), Ok(Some(b)));
    assert_eq!(r.evict_lru_leaf(&mut p), Ok(Some(a)));
    assert_eq!(r.node_count(), 0);
    let a = p.allocate().unwrap();
    let b = p.allocate().unwrap();
    let mut r = RadixPrefixIndex::<2>::new();
    r.insert_child_page(None, &[1], a, 7).unwrap();
    r.insert_child_page(None, &[2], b, 7).unwrap();
    assert_eq!(r.evict_lru_leaf(&mut p), Ok(Some(a)));
    let mut t1 = SequenceBlockTable::new(1);
    let mut t2 = SequenceBlockTable::new(1);
    t1.install_shared_prefix(&[b], &mut p).unwrap();
    t2.install_shared_prefix(&[b], &mut p).unwrap();
    let canonical = p.allocate().unwrap();
    assert_eq!(
        RadixPrefixIndex::<2>::repoint_page_references(
            &mut [&mut t1, &mut t2],
            b,
            canonical,
            &mut p
        ),
        Ok(2)
    );
    assert_eq!((p.ref_count(b), p.ref_count(canonical)), (Some(1), Some(3)));
    assert_eq!(t1.get(0), Some(canonical));
    assert_eq!(t2.get(0), Some(canonical));
}
#[test]
fn known_invalid_state_partial_mutation_and_max_epoch_are_not_overclaimed() {
    let mut p = BlockPool::new(1);
    let a = p.allocate().unwrap();
    let mut t = SequenceBlockTable::new(2);
    t.entries_mut().copy_from_slice(&[a, 99]);
    assert_eq!(
        t.release_all(&mut p),
        Err(TableError::Pool(PoolError::InvalidBlock))
    );
    assert_eq!(t.entries(), &[INVALID_BLOCK, 99]);
    assert_eq!(p.free_count(), 1);
    let a = p.allocate().unwrap();
    let mut r = RadixPrefixIndex::<1>::new();
    r.insert_child_page(None, &[1], a, u64::MAX).unwrap();
    assert_eq!(r.evict_lru_leaf(&mut p), Ok(None));
    assert_eq!(r.node_count(), 1);
    assert_eq!(p.ref_count(a), Some(1));
    // No claim that direct malformed public page_count/parent indices are safe.
}
#[test]
fn prepared_hot_operations_and_refusals_allocate_zero_with_positive_control() {
    let b = budget();
    let mut pool = BlockPool::new(4);
    let mut t1 = SequenceBlockTable::new(2);
    let mut t2 = SequenceBlockTable::new(2);
    let mut store = PrefixKvStore::<1, 2>::new();
    let mut radix = RadixPrefixIndex::<2>::new();
    let mut output = [99; 2];
    COUNT.with(|v| v.set(0));
    ACTIVE.with(|v| v.set(true));
    let control = std::hint::black_box(Box::new([7u8; 71]));
    std::hint::black_box(&control);
    drop(control);
    ACTIVE.with(|v| v.set(false));
    assert!(COUNT.with(Cell::get) > 0);
    COUNT.with(|v| v.set(0));
    ACTIVE.with(|v| v.set(true));
    let admitted = b.reserve(16, 16, 1, 0);
    let refused = b.reserve(u32::MAX, 1, 0, 0);
    let page = pool.allocate().unwrap();
    t1.install_shared_prefix(&[page], &mut pool).unwrap();
    t1.fork_into(&mut t2, &mut pool).unwrap();
    let cow = t2.ensure_writable(0, &mut pool);
    store
        .publish(PrefixPageSet::new(id(1), &[page], 16).unwrap(), &mut pool)
        .unwrap();
    let rejected = store.publish(PrefixPageSet::new(id(2), &[99], 16).unwrap(), &mut pool);
    store.remove(id(1), &mut pool).unwrap();
    t1.release_all(&mut pool).unwrap();
    t2.release_all(&mut pool).unwrap();
    radix.insert_child_page(None, &[1, 2], page, 1).unwrap();
    let matched = radix.match_longest_prefix(&[1, 2], 2, &mut output);
    let evicted = radix.evict_lru_leaf(&mut pool);
    ACTIVE.with(|v| v.set(false));
    let count = COUNT.with(Cell::get);
    assert!(admitted.is_ok());
    assert_eq!(refused, Err(BudgetError::IntegerOverflow));
    assert!(matches!(cow, Ok(CopyOnWrite::Copy { .. })));
    assert!(rejected.is_err());
    assert_eq!(matched, Ok((2, 1, 1)));
    assert_eq!(evicted, Ok(Some(page)));
    assert_eq!(pool.free_count(), 4);
    assert_eq!(count, 0);
}
