//! Nonblocking map ownership tied to worker completion generations.
use crate::worker_steward::{RequestToken, WorkerError, WorkerSteward};
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

pub type SharedWorker = Arc<Mutex<WorkerSteward>>;

pub fn ready_worker(is_isolated: bool) -> SharedWorker {
    let mut worker = WorkerSteward::new(is_isolated);
    worker.set_ready();
    Arc::new(Mutex::new(worker))
}

#[derive(Default)]
struct Signal {
    result: Option<bool>,
    waker: Option<Waker>,
}

pub struct PendingMap {
    buffer: etalon::Buffer,
    worker: SharedWorker,
    token: Option<RequestToken>,
    stamp: crate::CompletionStamp,
    signal: Arc<Mutex<Signal>>,
    published: Option<bool>,
    unmap_on_drop: bool,
}

impl PendingMap {
    pub fn begin(
        slice: etalon::BufferSlice<'_>,
        worker: SharedWorker,
    ) -> Result<Self, WorkerError> {
        let token = worker.lock().unwrap().begin_request()?;
        let signal = Arc::new(Mutex::new(Signal::default()));
        let callback = signal.clone();
        let buffer = slice.buffer().clone();
        slice.map_async(etalon::MapMode::Read, move |result| {
            let waker = {
                let mut state = callback.lock().unwrap();
                state.result = Some(result.is_ok());
                state.waker.take()
            };
            if let Some(waker) = waker {
                waker.wake();
            }
        });
        Ok(Self {
            buffer,
            worker,
            token: Some(token),
            stamp: token.stamp,
            signal,
            published: None,
            unmap_on_drop: true,
        })
    }

    pub fn try_complete(&mut self) -> Option<bool> {
        if let Some(result) = self.published {
            if result {
                let current = {
                    let worker = self.worker.lock().unwrap();
                    worker.is_ready() && worker.completion_stamp() == self.stamp
                };
                if !current {
                    self.buffer.unmap();
                    self.unmap_on_drop = false;
                    self.published = Some(false);
                    return Some(false);
                }
            }
            return Some(result);
        }
        let mapped = self.signal.lock().unwrap().result.take()?;
        let current = self
            .token
            .take()
            .is_some_and(|token| self.worker.lock().unwrap().complete_request(token));
        let accepted = mapped && current;
        if !accepted {
            self.buffer.unmap();
            self.unmap_on_drop = false;
        }
        self.published = Some(accepted);
        Some(accepted)
    }

    pub fn buffer(&self) -> &etalon::Buffer {
        &self.buffer
    }

    /// Transfer successful mapped ownership to an existing caller that already unmaps it.
    pub fn release_mapped(mut self) -> bool {
        if self.try_complete() != Some(true) {
            return false;
        }
        self.unmap_on_drop = false;
        true
    }
}

impl Future for PendingMap {
    type Output = bool;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<bool> {
        // Register before checking completion to avoid a callback/waker race.
        self.signal.lock().unwrap().waker = Some(cx.waker().clone());
        match self.try_complete() {
            Some(result) => Poll::Ready(result),
            None => Poll::Pending,
        }
    }
}

impl Drop for PendingMap {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            self.worker.lock().unwrap().cancel_request(token);
        }
        if self.unmap_on_drop {
            self.buffer.unmap();
        }
    }
}

pub async fn await_map(slice: etalon::BufferSlice<'_>, worker: SharedWorker) -> bool {
    let Ok(mut pending) = PendingMap::begin(slice, worker) else {
        return false;
    };
    if (&mut pending).await {
        pending.release_mapped()
    } else {
        false
    }
}

thread_local! {
    static BROWSER_WORKER: SharedWorker = ready_worker(false);
}
pub fn browser_worker() -> SharedWorker {
    BROWSER_WORKER.with(Arc::clone)
}

pub fn recover_browser_worker() -> Result<(), WorkerError> {
    let worker = browser_worker();
    let mut worker = worker.lock().unwrap();
    if worker.state() == crate::worker_steward::WorkerState::Suspended {
        let session = worker
            .generation()
            .checked_add(1)
            .ok_or(WorkerError::IdentityExhausted)?;
        worker.recover(session)?;
    }
    if !worker.is_ready() {
        return Err(WorkerError::NotReady);
    }
    Ok(())
}

pub fn begin_browser_model() -> Result<crate::CompletionStamp, WorkerError> {
    recover_browser_worker()?;
    let worker = browser_worker();
    let mut worker = worker.lock().unwrap();
    let stamp = worker.completion_stamp();
    let model = stamp
        .model
        .checked_add(1)
        .ok_or(WorkerError::IdentityExhausted)?;
    worker.set_completion_context(stamp.frame, model)?;
    Ok(worker.completion_stamp())
}
