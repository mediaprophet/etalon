use super::ForwardArena;
use core::task::{Poll, Waker};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenReadbackError {
    #[error("blocking token readback is unsupported on wasm32; await read_token_async instead")]
    BlockingUnsupported,
    #[error("a token readback is already in progress")]
    Busy,
    #[error("GPU token mapping failed")]
    MappingFailed,
    #[error("GPU token mapped range is unavailable")]
    MappedRangeUnavailable,
    #[error("GPU polling failed: {0}")]
    Poll(#[from] etalon::PollError),
    #[error("GPU polling worker could not start: {0}")]
    PollWorker(std::io::Error),
    #[error("GPU polling returned without completing token mapping")]
    MissingCallback,
}

#[derive(Default)]
struct Completion {
    result: Option<Result<(), TokenReadbackError>>,
    done: bool,
    waker: Option<Waker>,
}

fn finish(state: &Mutex<Completion>, result: Result<(), TokenReadbackError>) {
    let waker = {
        let mut state = state.lock().unwrap();
        if state.done {
            return;
        }
        state.done = true;
        state.result = Some(result);
        state.waker.take()
    };
    if let Some(waker) = waker {
        waker.wake();
    }
}

struct MappingGuard<'a> {
    buffer: &'a etalon::Buffer,
    busy: &'a AtomicBool,
}

impl MappingGuard<'_> {
    fn token(&self) -> Result<u32, TokenReadbackError> {
        let raw = self
            .buffer
            .slice(..4)
            .get_mapped_range()
            .map_err(|_| TokenReadbackError::MappedRangeUnavailable)?;
        let bytes = raw
            .get(..4)
            .ok_or(TokenReadbackError::MappedRangeUnavailable)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }
}

impl Drop for MappingGuard<'_> {
    fn drop(&mut self) {
        self.buffer.unmap();
        self.busy.store(false, Ordering::Release);
    }
}

impl ForwardArena {
    fn begin_token_map(
        &self,
    ) -> Result<(MappingGuard<'_>, Arc<Mutex<Completion>>), TokenReadbackError> {
        self.readback_busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .map_err(|_| TokenReadbackError::Busy)?;
        let guard = MappingGuard {
            buffer: &self.readback_stage_buf,
            busy: &self.readback_busy,
        };
        let state = Arc::new(Mutex::new(Completion::default()));
        let callback_state = Arc::clone(&state);
        self.readback_stage_buf
            .slice(..4)
            .map_async(etalon::MapMode::Read, move |result| {
                finish(
                    &callback_state,
                    result.map_err(|_| TokenReadbackError::MappingFailed),
                );
            });
        Ok((guard, state))
    }

    /// Submit the token copy first. Dropping the future cancels and unmaps staging.
    pub async fn try_read_token_async(
        &self,
        device: &etalon::Device,
    ) -> Result<u32, TokenReadbackError> {
        let (guard, state) = self.begin_token_map()?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            let device = device.clone();
            let state = Arc::clone(&state);
            std::thread::Builder::new()
                .name("etalon-token-map".into())
                .spawn(move || {
                    // Native needs active polling; the callback completes state before poll returns.
                    match device.poll(etalon::PollType::wait_or_timeout(
                        core::time::Duration::from_secs(5),
                    )) {
                        Err(error) => finish(&state, Err(TokenReadbackError::Poll(error))),
                        Ok(_) => {
                            let done = state.lock().unwrap().done;
                            if !done {
                                finish(&state, Err(TokenReadbackError::MissingCallback));
                            }
                        }
                    }
                })
                .map_err(TokenReadbackError::PollWorker)?;
        }
        #[cfg(target_arch = "wasm32")]
        let _ = device; // Browser WebBuffer::map_async registers Promise callbacks; poll is a no-op.
        core::future::poll_fn(|cx| {
            let mut state = state.lock().unwrap();
            if let Some(result) = state.result.take() {
                Poll::Ready(result)
            } else {
                state.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        })
        .await?;
        guard.token()
    }

    pub async fn read_token_async(&self, device: &etalon::Device) -> Option<u32> {
        self.try_read_token_async(device).await.ok()
    }

    pub fn try_read_token_blocking(
        &self,
        device: &etalon::Device,
    ) -> Result<u32, TokenReadbackError> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = device;
            Err(TokenReadbackError::BlockingUnsupported)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (guard, state) = self.begin_token_map()?;
            device.poll(etalon::PollType::wait_or_timeout(
                core::time::Duration::from_secs(5),
            ))?;
            state
                .lock()
                .unwrap()
                .result
                .take()
                .ok_or(TokenReadbackError::MissingCallback)??;
            guard.token()
        }
    }
}

#[cfg(test)]
mod tests {
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
}
