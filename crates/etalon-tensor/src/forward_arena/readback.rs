use super::ForwardArena;
use core::task::{Context, Poll, Waker};
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
    /// Retained for source compatibility; async token reads no longer start a worker.
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

fn poll_completion(
    state: &Mutex<Completion>,
    cx: &mut Context<'_>,
) -> Poll<Result<(), TokenReadbackError>> {
    let mut state = state.lock().unwrap();
    if let Some(result) = state.result.take() {
        Poll::Ready(result)
    } else {
        if !state
            .waker
            .as_ref()
            .is_some_and(|saved| saved.will_wake(cx.waker()))
        {
            state.waker = Some(cx.waker().clone());
        }
        Poll::Pending
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeMapWait<'a, F, D> {
    completion: &'a Mutex<Completion>,
    progress: F,
    expired: D,
}

#[cfg(not(target_arch = "wasm32"))]
impl<F, D> core::future::Future for NativeMapWait<'_, F, D>
where
    F: FnMut() -> Result<(), etalon::PollError> + Unpin,
    D: FnMut() -> bool + Unpin,
{
    type Output = Result<(), TokenReadbackError>;

    fn poll(self: core::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if let ready @ Poll::Ready(_) = poll_completion(this.completion, cx) {
            return ready;
        }
        if (this.expired)() {
            finish(
                this.completion,
                Err(TokenReadbackError::Poll(etalon::PollError::Timeout)),
            );
        } else if let Err(error) = (this.progress)() {
            finish(this.completion, Err(TokenReadbackError::Poll(error)));
        }
        let result = poll_completion(this.completion, cx);
        if result.is_pending() {
            // One nonblocking device poll per executor turn; native callbacks need active progress.
            cx.waker().wake_by_ref();
        }
        result
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
    /// Native executors drive one nonblocking device poll per turn until completion or five seconds.
    /// No background worker survives cancellation. Browser callbacks use the event loop.
    pub async fn try_read_token_async(
        &self,
        device: &etalon::Device,
    ) -> Result<u32, TokenReadbackError> {
        let (guard, state) = self.begin_token_map()?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            let started = std::time::Instant::now();
            NativeMapWait {
                completion: &state,
                progress: || device.poll(etalon::PollType::Poll).map(|_| ()),
                expired: || started.elapsed() >= core::time::Duration::from_secs(5),
            }
            .await?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = device; // Browser map Promise callbacks progress through the event loop.
            core::future::poll_fn(|cx| poll_completion(&state, cx)).await?;
        }
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
mod tests;
