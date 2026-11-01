use super::*;
use std::{
    cell::Cell,
    task::{Wake, Waker},
};

struct Noop;
impl Wake for Noop {
    fn wake(self: Arc<Self>) {}
}

#[test]
fn cancellation_releases_busy_state_and_retires_late_callbacks() {
    let active = AtomicBool::new(false);
    let state = MapCompletion::new();
    let cleaned = Cell::new(false);
    let guard = RequestGuard::new(&active, &state, || cleaned.set(true)).unwrap();
    let generation = guard.generation;
    assert!(matches!(
        RequestGuard::new(&active, &state, || {}),
        Err(GpuError::Busy)
    ));
    drop(guard);
    assert!(cleaned.get());
    assert!(!active.load(Ordering::Acquire));
    let next = RequestGuard::new(&active, &state, || {}).unwrap();
    state.complete(generation, true);
    let waker = Waker::from(Arc::new(Noop));
    let mut cx = core::task::Context::from_waker(&waker);
    assert_eq!(state.status(next.generation, &mut cx), None);
    state.complete(next.generation, true);
    assert_eq!(state.status(next.generation, &mut cx), Some(Ok(())));
}

#[test]
fn timeout_can_fail_an_operation_after_map_success_while_scopes_are_pending() {
    let state = MapCompletion::new();
    let generation = state.begin().unwrap();
    state.complete(generation, true);
    state.fail(generation, GpuError::PollTimedOut);
    let waker = Waker::from(Arc::new(Noop));
    let mut cx = core::task::Context::from_waker(&waker);
    assert_eq!(
        state.status(generation, &mut cx),
        Some(Err(GpuError::PollTimedOut))
    );
    state.fail(generation, GpuError::MapFailed);
    assert_eq!(
        state.status(generation, &mut cx),
        Some(Err(GpuError::PollTimedOut))
    );
}
