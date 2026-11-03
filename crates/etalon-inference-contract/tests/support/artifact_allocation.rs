use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
}
struct Counter;
#[global_allocator]
static ALLOCATOR: Counter = Counter;
fn record() {
    if MEASURING.try_with(Cell::get).unwrap_or(false) {
        let _ = COUNT.try_with(|n| n.set(n.get() + 1));
    }
}
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
pub fn measure<T>(f: impl FnOnce() -> T) -> (T, usize) {
    assert!(!MEASURING.with(Cell::get));
    COUNT.with(|n| n.set(0));
    MEASURING.with(|m| m.set(true));
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            MEASURING.with(|m| m.set(false));
        }
    }
    let guard = Reset;
    let result = f();
    drop(guard);
    (result, COUNT.with(Cell::get))
}
pub fn positive_control() -> usize {
    measure(|| {
        let layout = Layout::from_size_align(32, 8).unwrap();
        let pointer = unsafe { std::alloc::alloc(layout) };
        assert!(!pointer.is_null());
        std::hint::black_box(pointer);
        unsafe { std::alloc::dealloc(pointer, layout) };
    })
    .1
}
