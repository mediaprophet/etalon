use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Condvar, Mutex,
};

use crate::BleAdvertisementSample;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WatcherLifecycle {
    Ready,
    Starting,
    Started,
    Stopping,
    Stopped,
    Failed,
}

pub(super) const QUEUE_CAPACITY: usize = 64;

pub(super) struct CallbackState {
    pub(super) queue: Mutex<FixedQueue>,
    pub(super) callback_gate: Mutex<()>,
    pub(super) dropped: AtomicU64,
    pub(super) halted: AtomicBool,
    pub(super) collection_closed: AtomicBool,
    accepted_total: AtomicUsize,
    max_total: usize,
}

impl CallbackState {
    pub(super) const fn new(max_total: usize) -> Self {
        Self {
            queue: Mutex::new(FixedQueue::new()),
            callback_gate: Mutex::new(()),
            dropped: AtomicU64::new(0),
            halted: AtomicBool::new(false),
            collection_closed: AtomicBool::new(false),
            accepted_total: AtomicUsize::new(0),
            max_total,
        }
    }

    /// Close the callback path at a serialized boundary, then discard pending
    /// observations. Any callback already holding the gate completes before
    /// this method returns; later callbacks see `halted` before reading fields.
    pub(super) fn halt(&self) {
        match self.callback_gate.lock() {
            Ok(_callbacks) => {
                self.halted.store(true, Ordering::Release);
                if let Ok(mut queue) = self.queue.lock() {
                    queue.clear();
                }
            }
            Err(_) => {
                // A poisoned callback gate cannot authorize further reads.
                self.halted.store(true, Ordering::Release);
                if let Ok(mut queue) = self.queue.try_lock() {
                    queue.clear();
                }
            }
        }
    }

    /// Callback-only variant: caller already owns `callback_gate`.
    pub(super) fn halt_from_callback(&self) {
        self.halted.store(true, Ordering::Release);
        if let Ok(mut queue) = self.queue.try_lock() {
            queue.clear();
        }
    }

    pub(super) fn clear_queue(&self) -> Result<(), ()> {
        self.queue
            .lock()
            .map(|mut queue| queue.clear())
            .map_err(|_| ())
    }

    /// Wait for any in-flight event callback before discarding its queue data.
    pub(super) fn quiesce_and_clear(&self) -> Result<(), ()> {
        let _callbacks = self.callback_gate.lock().map_err(|_| ())?;
        self.clear_queue()
    }

    pub(super) fn close_collection(&self) {
        self.collection_closed.store(true, Ordering::Release);
    }

    pub(super) fn is_collecting(&self) -> bool {
        !self.halted.load(Ordering::Acquire) && !self.collection_closed.load(Ordering::Acquire)
    }

    /// Return true when this accepted record reaches the session limit.
    pub(super) fn note_accepted(&self) -> bool {
        let previous = self.accepted_total.fetch_add(1, Ordering::AcqRel);
        previous.saturating_add(1) >= self.max_total
    }
}

pub(super) struct FixedQueue {
    entries: [BleAdvertisementSample; QUEUE_CAPACITY],
    head: usize,
    len: usize,
}

impl FixedQueue {
    pub(super) const fn new() -> Self {
        Self {
            entries: [BleAdvertisementSample {
                bluetooth_address: 0,
                rssi_dbm: 0,
                os_timestamp_100ns: 0,
                os_address_type: None,
            }; QUEUE_CAPACITY],
            head: 0,
            len: 0,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn push(&mut self, sample: BleAdvertisementSample) -> bool {
        if self.len == QUEUE_CAPACITY {
            return false;
        }
        let tail = (self.head + self.len) % QUEUE_CAPACITY;
        self.entries[tail] = sample;
        self.len += 1;
        true
    }

    pub(super) fn pop_exact(&mut self, output: &mut [BleAdvertisementSample]) -> bool {
        if output.is_empty() || output.len() > self.len {
            return false;
        }
        let count = output.len();
        for slot in output.iter_mut() {
            *slot = self.entries[self.head];
            self.entries[self.head] = BleAdvertisementSample::default();
            self.head = (self.head + 1) % QUEUE_CAPACITY;
        }
        self.len -= count;
        true
    }

    pub(super) fn clear(&mut self) {
        self.entries.fill(BleAdvertisementSample::default());
        self.head = 0;
        self.len = 0;
    }
}

pub(super) fn increment_dropped(counter: &AtomicU64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
        Some(count.saturating_add(1))
    });
}

pub(super) struct StopSignal {
    requested: AtomicBool,
    started: AtomicBool,
    lifecycle: Mutex<WatcherLifecycle>,
    wake: Condvar,
}

impl StopSignal {
    pub(super) const fn new() -> Self {
        Self {
            requested: AtomicBool::new(false),
            started: AtomicBool::new(false),
            lifecycle: Mutex::new(WatcherLifecycle::Ready),
            wake: Condvar::new(),
        }
    }

    pub(super) fn requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    pub(super) fn is_started(&self) -> bool {
        self.started.load(Ordering::Acquire) && !self.requested()
    }

    /// Serialize the final live gate check and OS Start against the stop worker.
    /// The closure returns `(OS start succeeded, caller result)` independently:
    /// a gate may deny after invoking Start, in which case the worker still
    /// must stop the now-active watcher.
    pub(super) fn start_with<T, F>(&self, start: F) -> Result<(bool, T), ()>
    where
        F: FnOnce() -> (bool, T),
    {
        let mut state = self.lifecycle.lock().map_err(|_| ())?;
        if self.requested() || *state != WatcherLifecycle::Ready {
            return Err(());
        }
        *state = WatcherLifecycle::Starting;
        let (started, result) = start();
        self.started.store(started, Ordering::Release);
        // A failed Start does not prove the OS watcher stayed stopped. Mark it
        // uncertain so the stop worker performs bounded Stop/status verification.
        *state = if started {
            WatcherLifecycle::Started
        } else {
            WatcherLifecycle::Failed
        };
        self.wake.notify_all();
        Ok((started, result))
    }

    /// Stop only after Start has either completed or definitively failed.
    /// The stop closure runs under the lifecycle lock so no Start can race it.
    pub(super) fn stop_with<F>(&self, stop: F) -> Result<bool, ()>
    where
        F: FnOnce() -> bool,
    {
        let mut state = self.lifecycle.lock().map_err(|_| ())?;
        while *state == WatcherLifecycle::Starting {
            state = self.wake.wait(state).map_err(|_| ())?;
        }
        match *state {
            WatcherLifecycle::Ready | WatcherLifecycle::Stopped => Ok(true),
            WatcherLifecycle::Started | WatcherLifecycle::Failed => {
                *state = WatcherLifecycle::Stopping;
                self.started.store(false, Ordering::Release);
                let stopped = stop();
                *state = if stopped {
                    WatcherLifecycle::Stopped
                } else {
                    WatcherLifecycle::Failed
                };
                self.wake.notify_all();
                Ok(stopped)
            }
            WatcherLifecycle::Starting | WatcherLifecycle::Stopping => Err(()),
        }
    }

    pub(super) fn wait_slice(&self) -> Result<(), ()> {
        let guard = self.lifecycle.lock().map_err(|_| ())?;
        if self.requested() {
            return Ok(());
        }
        self.wake
            .wait_timeout(guard, std::time::Duration::from_millis(50))
            .map(|_| ())
            .map_err(|_| ())
    }

    pub(super) fn request_stop(&self) {
        self.requested.store(true, Ordering::Release);
        self.wake.notify_all();
    }
}
