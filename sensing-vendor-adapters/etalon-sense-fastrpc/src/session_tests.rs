use std::cell::Cell;
use super::*;

std::thread_local! {
    static CALLS: Cell<u32> = const { Cell::new(0) };
    static CLOSES: Cell<u32> = const { Cell::new(0) };
    static INVOKE_STATUS: Cell<i32> = const { Cell::new(0) };
    static CLOSE_STATUS: Cell<i32> = const { Cell::new(0) };
    static MUTATE_ARGS: Cell<bool> = const { Cell::new(false) };
}

struct Lease(Cell<bool>);
impl DeploymentLease for Lease { fn current(&self) -> bool { self.0.get() } }

unsafe extern "C" fn invoke(_handle: u64, _scalars: u32, args: *mut RemoteArg) -> i32 {
    CALLS.with(|calls| calls.set(calls.get() + 1));
    if MUTATE_ARGS.with(Cell::get) {
        unsafe { (*args.add(1)).buf.len = 0; }
    }
    INVOKE_STATUS.with(Cell::get)
}

unsafe extern "C" fn close(_handle: u64) -> i32 {
    CLOSES.with(|calls| calls.set(calls.get() + 1));
    CLOSE_STATUS.with(Cell::get)
}

fn session(lease: &Lease) -> FastRpcSession<'_> {
    CALLS.with(|v| v.set(0));
    CLOSES.with(|v| v.set(0));
    INVOKE_STATUS.with(|v| v.set(0));
    CLOSE_STATUS.with(|v| v.set(0));
    MUTATE_ARGS.with(|v| v.set(false));
    FastRpcSession { lease,
        #[cfg(any(target_os = "linux", target_os = "android"))]
        library: None,
        invoke, close, handle: Some(0), state: SessionState::Open,
        target_id: 1, generation: 1, domain: 3, session: 0, max_pairs: 1, _thread_affinity: PhantomData }
}

fn request() -> [u8; 68] {
    let mut request = [0; 68];
    let identity = etalon_sense_dsp_wire::RequestIdentity::new(1, 1, 1).unwrap();
    etalon_sense_dsp_wire::encode_request(identity, 1, &[0; 4], &mut request).unwrap();
    request
}

#[test]
fn a_successful_explicit_close_is_not_repeated_by_drop() {
    let lease = Lease(Cell::new(true));
    let mut owner = session(&lease);
    assert_eq!(owner.close(), Ok(()));
    assert_eq!(owner.state(), SessionState::Closed);
    drop(owner);
    assert_eq!(CLOSES.with(Cell::get), 1);
}

#[test]
fn failed_close_is_unknown_and_never_retried() {
    let lease = Lease(Cell::new(true));
    let mut owner = session(&lease);
    CLOSE_STATUS.with(|v| v.set(-1));
    assert_eq!(owner.close(), Err(FastRpcError::Close(-1)));
    assert_eq!(owner.close(), Err(FastRpcError::CloseOutcomeUnknown));
    drop(owner);
    assert_eq!(CLOSES.with(Cell::get), 1);
}

#[test]
fn revoked_admission_blocks_invoke_but_allows_cleanup() {
    let lease = Lease(Cell::new(true));
    let mut owner = session(&lease);
    lease.0.set(false);
    let mut output = [0xaa; 72];
    assert_eq!(owner.invoke_buffers(&request(), &mut output), Err(FastRpcError::DeploymentAdmission));
    assert_eq!(output, [0; 72]);
    assert_eq!(CALLS.with(Cell::get), 0);
    assert_eq!(owner.close(), Ok(()));
}

#[test]
fn native_failure_clears_output_and_faults_session() {
    let lease = Lease(Cell::new(true));
    let mut owner = session(&lease);
    INVOKE_STATUS.with(|v| v.set(-7));
    let mut output = [0xaa; 72];
    assert_eq!(owner.invoke_buffers(&request(), &mut output), Err(FastRpcError::Invoke(-7)));
    assert_eq!(output, [0; 72]);
    assert_eq!(owner.state(), SessionState::Faulted);
    assert_eq!(owner.invoke_buffers(&request(), &mut output), Err(FastRpcError::InvalidState));
    assert_eq!(CALLS.with(Cell::get), 1);
}

#[test]
fn changed_native_argument_lengths_are_rejected() {
    let lease = Lease(Cell::new(true));
    let mut owner = session(&lease);
    MUTATE_ARGS.with(|v| v.set(true));
    let mut output = [0xaa; 72];
    assert_eq!(owner.invoke_buffers(&request(), &mut output), Err(FastRpcError::ArgumentMutation));
    assert_eq!(output, [0; 72]);
}

#[test]
fn session_generation_mismatch_never_reaches_native_runtime() {
    let lease = Lease(Cell::new(true));
    let mut owner = session(&lease);
    let mut encoded = [0; 68];
    let identity = etalon_sense_dsp_wire::RequestIdentity::new(1, 2, 1).unwrap();
    etalon_sense_dsp_wire::encode_request(identity, 1, &[0; 4], &mut encoded).unwrap();
    let mut output = [0xaa; 72];
    assert_eq!(owner.invoke_buffers(&encoded, &mut output), Err(FastRpcError::Wire));
    assert_eq!(output, [0; 72]);
    assert_eq!(CALLS.with(Cell::get), 0);
}

#[test]
fn independent_sessions_never_enter_native_close_concurrently() {
    use std::sync::{atomic::{AtomicUsize, Ordering}, Arc, Barrier};

    static ACTIVE: AtomicUsize = AtomicUsize::new(0);
    static MAX_ACTIVE: AtomicUsize = AtomicUsize::new(0);
    static COMPLETED: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn observed_close(_handle: u64) -> i32 {
        let active = ACTIVE.fetch_add(1, Ordering::SeqCst) + 1;
        MAX_ACTIVE.fetch_max(active, Ordering::SeqCst);
        // Keep the simulated native lifetime open across scheduling points.
        for _ in 0..32 { std::thread::yield_now(); }
        ACTIVE.fetch_sub(1, Ordering::SeqCst);
        COMPLETED.fetch_add(1, Ordering::SeqCst);
        0
    }

    ACTIVE.store(0, Ordering::SeqCst);
    MAX_ACTIVE.store(0, Ordering::SeqCst);
    COMPLETED.store(0, Ordering::SeqCst);
    let start = Arc::new(Barrier::new(8));
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let start = Arc::clone(&start);
            scope.spawn(move || {
                // Thread-affine sessions are independently constructed.
                let lease = Lease(Cell::new(true));
                start.wait();
                for _ in 0..16 {
                    let mut owner = session(&lease);
                    owner.close = observed_close;
                    assert_eq!(owner.close(), Ok(()));
                }
            });
        }
    });
    assert_eq!(COMPLETED.load(Ordering::SeqCst), 128);
    assert_eq!(ACTIVE.load(Ordering::SeqCst), 0);
    assert_eq!(MAX_ACTIVE.load(Ordering::SeqCst), 1);
}
