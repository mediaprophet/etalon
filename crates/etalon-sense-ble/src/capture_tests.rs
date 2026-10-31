use super::*;
use crate::config::{MAX_CAPTURE_SAMPLES, MAX_SESSION_OBSERVATIONS};
use core::cell::Cell;
use etalon_sense_core::{
    adapter::{AdapterId, ExtendedDiscoveredSource},
    authority::{
        AuthorizedCaptureRequest, CaptureAuthority, CaptureAuthorizationRequest,
        CapturePolicyDecision, CaptureSource, HostCapturePolicy,
    },
    time::ClockStamp,
};
use std::sync::{atomic::AtomicU64, mpsc, Arc, Mutex};
use std::thread;

fn sample(index: u64) -> BleAdvertisementSample {
    BleAdvertisementSample {
        bluetooth_address: index,
        rssi_dbm: -(index as i16),
        os_timestamp_100ns: index as i64 * 10,
        os_address_type: None,
    }
}

fn config(source_id: u64, max_batch: usize) -> Result<BleConfig, AdapterError> {
    BleConfig::new(
        AdapterId::new(4)?,
        source_id,
        700,
        701,
        702,
        max_batch,
        MAX_SESSION_OBSERVATIONS,
        5_000,
    )
}

struct TestGate {
    permit: bool,
}

impl BleCollectionGate for TestGate {
    fn with_permitted_collection(
        &self,
        _scope: BleCollectionScope,
        collect: &mut dyn FnMut(),
    ) -> bool {
        if self.permit {
            collect();
        }
        self.permit
    }
}

struct PanickingGate;

impl BleCollectionGate for PanickingGate {
    fn with_permitted_collection(
        &self,
        _scope: BleCollectionScope,
        _collect: &mut dyn FnMut(),
    ) -> bool {
        panic!("test host gate panic");
    }
}

fn scope() -> BleCollectionScope {
    struct TestClock;
    impl TrustedCaptureClock for TestClock {
        fn now(&mut self) -> Result<ClockStamp, AdapterError> {
            Ok(ClockStamp::new(1, 2, 10, 0).unwrap())
        }
    }
    struct TestPolicy;
    impl HostCapturePolicy for TestPolicy {
        fn decide(&self, _: &CaptureAuthorizationRequest, _: ClockStamp) -> CapturePolicyDecision {
            CapturePolicyDecision::Permit {
                expires_at: ClockStamp::new(1, 2, 100, 0).unwrap(),
            }
        }
    }
    let capability = config(42, 8).unwrap().capability();
    let source = ExtendedDiscoveredSource::new(42, capability).unwrap();
    let request =
        CaptureAuthorizationRequest::new(CaptureSource::Extended(source), None, None, 8, 9)
            .unwrap();
    let issued = CaptureAuthority::issue(request, &TestPolicy, &mut TestClock).unwrap();
    match issued.into_capture_request() {
        AuthorizedCaptureRequest::Extended(request) => BleCollectionScope::from_request(&request),
        _ => panic!("expected extended capture request"),
    }
}

#[test]
fn collection_scope_preserves_the_authorized_request() {
    let scope = scope();
    assert_eq!(scope.source().source_id(), 42);
    assert_eq!(scope.purpose_id(), 9);
    assert_eq!(scope.authorized_at(), ClockStamp::new(1, 2, 10, 0).unwrap());
    assert_eq!(scope.expires_at(), ClockStamp::new(1, 2, 100, 0).unwrap());
}

#[test]
fn fixed_queue_keeps_order_across_ring_wrap() {
    let mut queue = FixedQueue::new();
    for index in 0..QUEUE_CAPACITY as u64 {
        assert!(queue.push(sample(index)));
    }

    let mut prefix = [BleAdvertisementSample::default(); QUEUE_CAPACITY - 4];
    assert!(queue.pop_exact(&mut prefix));
    for (index, item) in prefix.iter().enumerate() {
        assert_eq!(*item, sample(index as u64));
    }
    for index in QUEUE_CAPACITY as u64..QUEUE_CAPACITY as u64 + 6 {
        assert!(queue.push(sample(index)));
    }

    let mut wrapped = [BleAdvertisementSample::default(); 10];
    assert!(queue.pop_exact(&mut wrapped));
    for (offset, item) in wrapped.iter().enumerate() {
        assert_eq!(*item, sample((QUEUE_CAPACITY - 4 + offset) as u64));
    }
    assert_eq!(queue.len(), 0);
}

#[test]
fn full_queue_drops_newest_and_records_the_drop() {
    let mut queue = FixedQueue::new();
    let dropped = AtomicU64::new(0);
    for index in 0..QUEUE_CAPACITY as u64 {
        assert!(queue.push(sample(index)));
    }
    if !queue.push(sample(QUEUE_CAPACITY as u64)) {
        increment_dropped(&dropped);
    }
    assert_eq!(dropped.load(Ordering::Relaxed), 1);

    let mut all = [BleAdvertisementSample::default(); QUEUE_CAPACITY];
    assert!(queue.pop_exact(&mut all));
    for (index, item) in all.iter().enumerate() {
        assert_eq!(*item, sample(index as u64));
    }
}

#[test]
fn insufficient_batch_does_not_consume_queued_samples() {
    let mut queue = FixedQueue::new();
    assert!(queue.push(sample(9)));
    let mut too_many = [BleAdvertisementSample::default(); 2];
    assert!(!queue.pop_exact(&mut too_many));
    let mut one = [BleAdvertisementSample::default(); 1];
    assert!(queue.pop_exact(&mut one));
    assert_eq!(one[0], sample(9));
}

#[test]
fn config_requires_nonzero_source_and_bounded_batch() {
    assert_eq!(config(0, 1), Err(AdapterError::InvalidIdentifier));
    assert_eq!(config(42, 0), Err(AdapterError::InvalidIdentifier));
    assert_eq!(
        config(42, MAX_CAPTURE_SAMPLES + 1),
        Err(AdapterError::InvalidIdentifier)
    );

    let valid = config(42, 8).unwrap();
    assert_eq!(valid.source_id(), 42);
    assert_eq!(valid.max_batch_samples(), 8);
    assert_eq!(valid.capability().modality_id(), 700);
    assert_eq!(valid.capability().quantity_id(), 701);
    assert_eq!(valid.capability().unit_id(), Some(702));
    assert_eq!(valid.capability().encoding(), SampleEncoding::I16);
    assert_eq!(valid.capability().max_extents(), [8, 0, 0, 0]);
    assert_eq!(valid.max_session_observations(), MAX_SESSION_OBSERVATIONS);
    assert_eq!(valid.max_session_duration_ms(), 5_000);
}

#[test]
fn denied_live_gate_does_not_run_collection_closure() {
    let gate = TestGate { permit: false };
    let mut collected = false;
    let mut collect = || collected = true;
    assert!(!run_collection_gate(&gate, scope(), &mut collect));
    assert!(!collected);
}

#[test]
fn permitted_live_gate_runs_collection_closure() {
    let gate = TestGate { permit: true };
    let collected = Cell::new(false);
    let mut collect = || collected.set(true);
    assert!(run_collection_gate(&gate, scope(), &mut collect));
    assert!(collected.get());
}

#[test]
fn panicking_host_gate_fails_closed_without_unwinding() {
    let gate = PanickingGate;
    let mut collected = false;
    let mut collect = || collected = true;
    assert!(!run_collection_gate(&gate, scope(), &mut collect));
    assert!(!collected);
}

#[test]
fn failed_start_is_unknown_until_bounded_stop_confirms_state() {
    let signal = StopSignal::new();
    assert_eq!(signal.start_with(|| (false, ())).unwrap(), (false, ()));
    let stop_checked = Cell::new(false);
    assert!(signal
        .stop_with(|| {
            stop_checked.set(true);
            true
        })
        .unwrap());
    assert!(stop_checked.get());
}

#[test]
fn failed_stop_can_be_retried_without_restarting() {
    let signal = StopSignal::new();
    assert_eq!(signal.start_with(|| (true, ())).unwrap(), (true, ()));
    assert!(!signal.stop_with(|| false).unwrap());
    assert!(signal.stop_with(|| true).unwrap());
}

#[test]
fn stop_worker_cannot_stop_watcher_before_start_transition_finishes() {
    let signal = Arc::new(StopSignal::new());
    let events = Arc::new(Mutex::new(Vec::new()));
    let (start_entered_tx, start_entered_rx) = mpsc::channel();
    let (finish_start_tx, finish_start_rx) = mpsc::channel();

    let starting_signal = Arc::clone(&signal);
    let starting_events = Arc::clone(&events);
    let start_thread = thread::spawn(move || {
        starting_signal
            .start_with(|| {
                assert!(!starting_signal.is_started());
                starting_events.lock().unwrap().push("start-enter");
                start_entered_tx.send(()).unwrap();
                finish_start_rx.recv().unwrap();
                starting_events.lock().unwrap().push("start-return");
                (true, ())
            })
            .unwrap()
    });

    start_entered_rx.recv().unwrap();
    signal.request_stop();
    let stopping_signal = Arc::clone(&signal);
    let stopping_events = Arc::clone(&events);
    let stop_thread = thread::spawn(move || {
        stopping_signal
            .stop_with(|| {
                stopping_events.lock().unwrap().push("stop");
                true
            })
            .unwrap()
    });

    finish_start_tx.send(()).unwrap();
    assert_eq!(start_thread.join().unwrap(), (true, ()));
    assert!(stop_thread.join().unwrap());
    assert_eq!(
        *events.lock().unwrap(),
        ["start-enter", "start-return", "stop"]
    );
}

#[test]
fn halt_waits_for_in_flight_callback_before_closing_collection() {
    let state = Arc::new(CallbackState::new(8));
    let callback_state = Arc::clone(&state);
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let callback = thread::spawn(move || {
        let _gate = callback_state.callback_gate.lock().unwrap();
        entered_tx.send(()).unwrap();
        release_rx.recv().unwrap();
    });
    entered_rx.recv().unwrap();

    let halting_state = Arc::clone(&state);
    let (halt_started_tx, halt_started_rx) = mpsc::channel();
    let halter = thread::spawn(move || {
        halt_started_tx.send(()).unwrap();
        halting_state.halt();
    });
    halt_started_rx.recv().unwrap();
    assert!(!state.halted.load(Ordering::Acquire));
    release_tx.send(()).unwrap();
    callback.join().unwrap();
    halter.join().unwrap();
    assert!(state.halted.load(Ordering::Acquire));
    assert!(!state.is_collecting());
}
