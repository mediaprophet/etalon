use etalon_wasm::{WorkerError, WorkerState, WorkerSteward, MAX_IN_FLIGHT_REQUESTS};
use std::sync::atomic::Ordering;

#[test]
fn admission_is_bounded_and_completion_is_exactly_once_in_both_modes() {
    for isolated in [false, true] {
        let mut worker = WorkerSteward::new(isolated);
        assert_eq!(worker.begin_request(), Err(WorkerError::NotReady));
        worker.mark_ready();
        let tokens: Vec<_> = (0..MAX_IN_FLIGHT_REQUESTS)
            .map(|_| worker.begin_request().unwrap())
            .collect();
        assert_eq!(worker.begin_request(), Err(WorkerError::Backpressure));
        assert_eq!(worker.pending_count(), MAX_IN_FLIGHT_REQUESTS);
        assert_eq!(
            worker.configure_transport(!isolated),
            Err(WorkerError::Backpressure)
        );
        for token in tokens.iter().rev() {
            assert_eq!(worker.request_token(token.id), Some(*token));
            assert!(worker.complete_request(*token));
            assert!(!worker.complete_request(*token));
        }
        assert_eq!(worker.pending_count(), 0);
        assert_eq!(worker.fallback_transport.pending_count(), 0);
        assert_eq!(worker.state(), WorkerState::Ready);
        assert!(worker.begin_request().unwrap().id > tokens.last().unwrap().id);
    }
}

#[test]
fn stale_frame_model_device_and_session_callbacks_cannot_complete_new_work() {
    for isolated in [false, true] {
        let mut worker = WorkerSteward::new(isolated);
        worker.mark_ready();
        let frame = worker.begin_request().unwrap();
        worker.set_completion_context(1, 0).unwrap();
        let model = worker.begin_request().unwrap();
        assert!(!worker.complete_request(frame));
        assert_eq!(worker.pending_count(), 1);
        worker.set_completion_context(1, 1).unwrap();
        let device = worker.begin_request().unwrap();
        assert!(!worker.complete_request(model));
        worker.notify_device_lost();
        assert_eq!(worker.state(), WorkerState::Suspended);
        assert_eq!(worker.begin_request(), Err(WorkerError::NotReady));
        assert!(!worker.complete_request(device));
        assert_eq!(worker.recover(0), Err(WorkerError::StaleGeneration));
        worker.recover(1).unwrap();
        let session = worker.begin_request().unwrap();
        assert_eq!(session.stamp.session, 1);
        assert_eq!(session.stamp.device, 1);
        assert!(!worker.complete_request(device));
        assert_eq!(worker.pending_count(), 1);
        assert_eq!(
            worker.set_completion_context(0, 1),
            Err(WorkerError::StaleGeneration)
        );
        assert_eq!(
            worker.set_completion_context(1, 0),
            Err(WorkerError::StaleGeneration)
        );
        assert!(worker.complete_request(session));
        worker.drain();
        worker.recover(2).unwrap();
        let current = worker.begin_request().unwrap();
        assert!(!worker.complete_request(session));
        assert!(worker.complete_request(current));
    }
}

#[test]
fn draining_waits_for_completion_or_cancellation_then_recovery_requires_new_session() {
    let mut worker = WorkerSteward::new(false);
    worker.mark_ready();
    let first = worker.begin_request().unwrap();
    let second = worker.begin_request().unwrap();
    worker.drain();
    assert_eq!(worker.state(), WorkerState::Draining);
    assert_eq!(worker.begin_request(), Err(WorkerError::NotReady));
    assert!(worker.complete_request(first));
    assert_eq!(worker.state(), WorkerState::Draining);
    assert!(worker.cancel_request(second));
    assert!(!worker.cancel_request(second));
    assert!(!worker.complete_request(second));
    assert_eq!(worker.state(), WorkerState::Suspended);
    assert_eq!(worker.fallback_transport.pending_count(), 0);
    worker.recover(1).unwrap();
    assert_eq!(worker.state(), WorkerState::Ready);
    worker.drain();
    assert_eq!(worker.state(), WorkerState::Suspended);
}

#[test]
fn termination_retires_requests_and_canvas_and_cannot_be_recovered() {
    let mut worker = WorkerSteward::new(false);
    worker.mark_ready();
    assert!(worker.attach_offscreen_canvas());
    assert!(!worker.attach_offscreen_canvas());
    let token = worker.begin_request().unwrap();
    worker.terminate();
    worker.terminate();
    worker.mark_ready();
    worker.notify_device_lost();
    assert_eq!(worker.state(), WorkerState::Terminated);
    assert!(!worker.attach_offscreen_canvas());
    assert!(!worker.detach_offscreen_canvas());
    assert!(!worker.complete_request(token));
    assert_eq!(worker.pending_count(), 0);
    assert_eq!(worker.fallback_transport.pending_count(), 0);
    assert_eq!(worker.recover(1), Err(WorkerError::NotReady));
    assert_eq!(worker.begin_request(), Err(WorkerError::NotReady));
}

#[test]
fn exhausted_request_identity_fails_closed_without_pending_entries() {
    for isolated in [false, true] {
        let mut worker = WorkerSteward::new(isolated);
        worker.mark_ready();
        worker
            .next_request_id
            .store(u64::MAX - 1, Ordering::Relaxed);
        let last = worker.begin_request().unwrap();
        assert_eq!(last.id, u64::MAX - 1);
        assert_eq!(worker.begin_request(), Err(WorkerError::IdentityExhausted));
        assert!(worker.complete_request(last));
        assert_eq!(worker.begin_request(), Err(WorkerError::IdentityExhausted));
        assert_eq!(worker.allocate_request_id(), 0);
        assert_eq!(worker.pending_count(), 0);
        assert_eq!(worker.fallback_transport.pending_count(), 0);
        worker.next_request_id.store(0, Ordering::Relaxed);
        assert_eq!(
            worker.try_allocate_request_id(),
            Err(WorkerError::IdentityExhausted)
        );
    }
}
