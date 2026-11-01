use super::*;
use std::task::Wake;

struct Counter(std::sync::atomic::AtomicUsize);
impl Wake for Counter {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn callback_wakes_once_and_retains_first_completion() {
    let counter = Arc::new(Counter(std::sync::atomic::AtomicUsize::new(0)));
    let state = Mutex::new(Completion {
        waker: Some(Waker::from(counter.clone())),
        ..Completion::default()
    });
    finish(&state, Ok(()));
    finish(&state, Err(TokenReadbackError::MappingFailed));
    assert_eq!(counter.0.load(Ordering::Relaxed), 1);
    assert!(state.lock().unwrap().result.take().unwrap().is_ok());
    finish(&state, Err(TokenReadbackError::MissingCallback));
    assert!(state.lock().unwrap().result.is_none());
}

#[test]
fn completion_before_first_poll_is_retained() {
    let state = Mutex::new(Completion::default());
    finish(&state, Err(TokenReadbackError::MappingFailed));
    assert!(matches!(
        state.lock().unwrap().result.take(),
        Some(Err(TokenReadbackError::MappingFailed))
    ));
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use core::{future::Future, task::Context};
    use std::cell::Cell;

    fn waker() -> Waker {
        Waker::from(Arc::new(Counter(std::sync::atomic::AtomicUsize::new(0))))
    }

    #[test]
    fn repeated_nonblocking_polls_progress_until_mapping_completes() {
        let state = Mutex::new(Completion::default());
        let calls = Cell::new(0);
        let mut wait = Box::pin(NativeMapWait {
            completion: &state,
            progress: || {
                calls.set(calls.get() + 1);
                if calls.get() == 3 {
                    finish(&state, Ok(()));
                }
                Ok(())
            },
            expired: || false,
        });
        let waker = waker();
        let mut cx = Context::from_waker(&waker);
        assert!(wait.as_mut().poll(&mut cx).is_pending());
        assert_eq!(calls.get(), 1);
        assert!(wait.as_mut().poll(&mut cx).is_pending());
        assert_eq!(calls.get(), 2);
        assert!(matches!(wait.as_mut().poll(&mut cx), Poll::Ready(Ok(()))));
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn callback_completed_before_poll_does_not_poll_or_self_wake() {
        let state = Mutex::new(Completion::default());
        finish(&state, Ok(()));
        let counter = Arc::new(Counter(std::sync::atomic::AtomicUsize::new(0)));
        let waker = Waker::from(counter.clone());
        let mut cx = Context::from_waker(&waker);
        let mut wait = Box::pin(NativeMapWait {
            completion: &state,
            progress: || -> Result<(), etalon::PollError> {
                panic!("completed mapping must not poll device")
            },
            expired: || -> bool { panic!("completed mapping must not inspect deadline") },
        });
        assert!(matches!(wait.as_mut().poll(&mut cx), Poll::Ready(Ok(()))));
        assert_eq!(counter.0.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn native_poll_errors_finish_the_wait() {
        let state = Mutex::new(Completion::default());
        let mut wait = Box::pin(NativeMapWait {
            completion: &state,
            progress: || Err(etalon::PollError::CantWait),
            expired: || false,
        });
        let waker = waker();
        let mut cx = Context::from_waker(&waker);
        assert!(matches!(
            wait.as_mut().poll(&mut cx),
            Poll::Ready(Err(TokenReadbackError::Poll(etalon::PollError::CantWait)))
        ));
    }

    #[test]
    fn expired_deadline_does_not_poll_and_cannot_override_completion() {
        let state = Mutex::new(Completion::default());
        let mut wait = Box::pin(NativeMapWait {
            completion: &state,
            progress: || -> Result<(), etalon::PollError> {
                panic!("expired wait must not poll device")
            },
            expired: || true,
        });
        let waker = waker();
        let mut cx = Context::from_waker(&waker);
        assert!(matches!(
            wait.as_mut().poll(&mut cx),
            Poll::Ready(Err(TokenReadbackError::Poll(etalon::PollError::Timeout)))
        ));
        drop(wait);
        finish(&state, Ok(()));
        assert!(state.lock().unwrap().result.is_none());
    }

    #[test]
    fn dropping_pending_wait_releases_owner_and_late_callback_cannot_poll() {
        struct Release<'a>(&'a Cell<bool>);
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }

        let state = Mutex::new(Completion::default());
        let calls = Cell::new(0);
        let released = Cell::new(false);
        let mut operation = Box::pin(async {
            let _owner = Release(&released);
            NativeMapWait {
                completion: &state,
                progress: || {
                    calls.set(calls.get() + 1);
                    Ok(())
                },
                expired: || false,
            }
            .await
        });
        let waker = waker();
        let mut cx = Context::from_waker(&waker);
        assert!(operation.as_mut().poll(&mut cx).is_pending());
        assert_eq!(calls.get(), 1);
        assert!(!released.get());
        drop(operation);
        assert!(released.get());
        finish(&state, Err(TokenReadbackError::MappingFailed));
        assert_eq!(calls.get(), 1);
    }
}
