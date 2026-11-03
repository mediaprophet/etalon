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

pub(super) struct AllocGuard;
impl AllocGuard {
    pub(super) fn begin(_label: &'static str, strict: bool) -> Self {
        assert!(strict);
        MEASURING.with(|active| assert!(!active.replace(true)));
        ALLOCATIONS.set(0);
        Self
    }
    pub(super) fn check(self) -> Result<(), usize> {
        MEASURING.set(false);
        match ALLOCATIONS.get() {
            0 => Ok(()),
            count => Err(count),
        }
    }
}
impl Drop for AllocGuard {
    fn drop(&mut self) {
        MEASURING.set(false);
    }
}
#[test]
fn positive_control_includes_freed_allocation() {
    let guard = AllocGuard::begin("positive", true);
    let bytes = vec![7u8; 89];
    std::hint::black_box(&bytes);
    drop(bytes);
    assert!(guard.check().is_err());
}
#[test]
fn measurement_clears_after_check() {
    assert!(AllocGuard::begin("clear", true).check().is_ok());
    assert!(!MEASURING.get());
}
#[test]
fn measurement_clears_on_unwind() {
    assert!(std::panic::catch_unwind(|| {
        let _guard = AllocGuard::begin("unwind", true);
        panic!("controlled measurement unwind");
    })
    .is_err());
    assert!(!MEASURING.get());
}
#[test]
fn cold_allocations_do_not_count() {
    ALLOCATIONS.set(0);
    let bytes = vec![1u8; 81];
    std::hint::black_box(&bytes);
    assert_eq!(ALLOCATIONS.get(), 0);
}
#[test]
fn stack_operation_is_zero_heap() {
    let guard = AllocGuard::begin("stack", true);
    std::hint::black_box([4u8; 96]);
    assert!(guard.check().is_ok());
}
