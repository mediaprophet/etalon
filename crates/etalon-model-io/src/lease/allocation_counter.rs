//! Test-local raw allocation-call measurement on the current thread.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<u64> = const { Cell::new(0) };
}
fn count() {
    let _ = ACTIVE.try_with(|active| {
        if active.get() {
            let _ = CALLS.try_with(|n| n.set(n.get().saturating_add(1)));
        }
    });
}
struct Counter;
// SAFETY: every operation forwards the original pointer, size and alignment to
// System; thread-local counting does not dereference or retain allocation data.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;

#[derive(Debug)]
pub(super) struct AllocGuard;
impl AllocGuard {
    pub(super) fn begin(_label: &'static str, strict: bool) -> Self {
        assert!(strict, "only raw allocation-call measurement is supported");
        ACTIVE.with(|active| assert!(!active.replace(true), "nested measurement"));
        CALLS.set(0);
        Self
    }
    pub(super) fn check(self) -> Result<(), &'static str> {
        ACTIVE.set(false);
        if CALLS.get() == 0 {
            Ok(())
        } else {
            Err("allocation observed")
        }
    }
}
impl Drop for AllocGuard {
    fn drop(&mut self) {
        ACTIVE.set(false);
    }
}
pub(super) fn assert_zero_alloc(label: &'static str, operation: impl FnOnce()) {
    let guard = AllocGuard::begin(label, true);
    operation();
    assert!(guard.check().is_ok(), "allocation in {label}");
}

#[test]
fn positive_control_observes_freed_allocation() {
    let guard = AllocGuard::begin("positive control", true);
    let bytes = vec![7u8; 64];
    std::hint::black_box(&bytes);
    drop(bytes);
    assert!(guard.check().is_err());
}
#[test]
fn measurement_clears_on_check() {
    assert!(AllocGuard::begin("check", true).check().is_ok());
    assert!(!ACTIVE.get());
}
#[test]
fn measurement_clears_on_unwind() {
    let _ = std::panic::catch_unwind(|| {
        let _guard = AllocGuard::begin("unwind", true);
        panic!("fixture unwind");
    });
    assert!(!ACTIVE.get());
}
#[test]
fn unmeasured_allocations_do_not_count() {
    CALLS.set(0);
    let bytes = vec![0u8; 64];
    std::hint::black_box(&bytes);
    assert_eq!(CALLS.get(), 0);
}
#[test]
fn zero_allocation_operation_passes() {
    assert_zero_alloc("stack", || {
        std::hint::black_box([0u8; 64]);
    });
}
