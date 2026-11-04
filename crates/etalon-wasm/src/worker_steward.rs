//! Bounded worker ownership and completion lifecycle shared by both transports.

use super::msg_transport::TransferableMessageTransport;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub const MAX_IN_FLIGHT_REQUESTS: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportMode {
    SharedArrayBufferRing,
    TransferableArrayBuffer,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerState {
    Initializing,
    Ready,
    Running,
    Suspended,
    Terminated,
    Draining,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompletionStamp {
    pub session: u32,
    pub device: u32,
    pub frame: u64,
    pub model: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestToken {
    pub id: u64,
    pub stamp: CompletionStamp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerError {
    NotReady,
    Backpressure,
    IdentityExhausted,
    StaleGeneration,
}

pub struct WorkerSteward {
    pub transport_mode: TransportMode,
    pub state: WorkerState,
    pub next_request_id: AtomicU64,
    pub has_offscreen_canvas: AtomicBool,
    pub fallback_transport: TransferableMessageTransport,
    stamp: CompletionStamp,
    pending: [Option<RequestToken>; MAX_IN_FLIGHT_REQUESTS],
}

impl WorkerSteward {
    pub fn new(is_isolated: bool) -> Self {
        Self {
            transport_mode: if is_isolated {
                TransportMode::SharedArrayBufferRing
            } else {
                TransportMode::TransferableArrayBuffer
            },
            state: WorkerState::Initializing,
            next_request_id: AtomicU64::new(1),
            has_offscreen_canvas: AtomicBool::new(false),
            fallback_transport: TransferableMessageTransport::new(),
            stamp: CompletionStamp {
                session: 0,
                device: 0,
                frame: 0,
                model: 0,
            },
            pending: [None; MAX_IN_FLIGHT_REQUESTS],
        }
    }

    pub fn set_ready(&mut self) {
        if self.state == WorkerState::Initializing {
            self.state = WorkerState::Ready;
        }
    }

    pub fn mark_ready(&mut self) {
        self.set_ready();
    }
    pub fn state(&self) -> WorkerState {
        self.state
    }
    pub fn generation(&self) -> u32 {
        self.stamp.session
    }
    pub fn completion_stamp(&self) -> CompletionStamp {
        self.stamp
    }

    pub fn attach_offscreen_canvas(&self) -> bool {
        self.state != WorkerState::Terminated
            && !self.has_offscreen_canvas.swap(true, Ordering::SeqCst)
    }

    pub fn detach_offscreen_canvas(&self) -> bool {
        self.has_offscreen_canvas.swap(false, Ordering::SeqCst)
    }

    pub fn try_allocate_request_id(&self) -> Result<u64, WorkerError> {
        self.next_request_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| {
                (id != 0).then(|| id.checked_add(1)).flatten()
            })
            .map_err(|_| WorkerError::IdentityExhausted)
    }

    /// Zero is reserved for exhaustion in the compatibility API.
    pub fn allocate_request_id(&self) -> u64 {
        self.try_allocate_request_id().unwrap_or(0)
    }

    pub fn is_ready(&self) -> bool {
        matches!(self.state, WorkerState::Ready | WorkerState::Running)
    }

    pub fn begin_request(&mut self) -> Result<RequestToken, WorkerError> {
        if !self.is_ready() {
            return Err(WorkerError::NotReady);
        }
        let slot = self
            .pending
            .iter()
            .position(Option::is_none)
            .ok_or(WorkerError::Backpressure)?;
        let token = RequestToken {
            id: self.try_allocate_request_id()?,
            stamp: self.stamp,
        };
        if self.transport_mode == TransportMode::TransferableArrayBuffer
            && !self.fallback_transport.track_request(token.id)
        {
            return Err(WorkerError::Backpressure);
        }
        self.pending[slot] = Some(token);
        self.state = WorkerState::Running;
        Ok(token)
    }

    pub fn cancel_request(&mut self, token: RequestToken) -> bool {
        let Some(slot) = self.pending.iter().position(|entry| *entry == Some(token)) else {
            return false;
        };
        self.pending[slot] = None;
        self.fallback_transport.complete_request(token.id);
        self.finish_drain();
        true
    }

    pub fn complete_request(&mut self, token: RequestToken) -> bool {
        let current = token.stamp == self.stamp
            && matches!(
                self.state,
                WorkerState::Ready | WorkerState::Running | WorkerState::Draining
            );
        self.cancel_request(token) && current
    }

    pub fn request_token(&self, id: u64) -> Option<RequestToken> {
        self.pending
            .iter()
            .flatten()
            .find(|token| token.id == id)
            .copied()
    }

    pub fn configure_transport(&mut self, shared_ring_attached: bool) -> Result<(), WorkerError> {
        if self.pending_count() != 0 {
            return Err(WorkerError::Backpressure);
        }
        self.transport_mode = if shared_ring_attached {
            TransportMode::SharedArrayBufferRing
        } else {
            TransportMode::TransferableArrayBuffer
        };
        Ok(())
    }

    pub fn pending_count(&self) -> usize {
        self.pending.iter().flatten().count()
    }

    fn finish_drain(&mut self) {
        if self.pending_count() == 0 {
            self.state = match self.state {
                WorkerState::Draining => WorkerState::Suspended,
                WorkerState::Running => WorkerState::Ready,
                other => other,
            };
        }
    }

    pub fn drain(&mut self) {
        if self.is_ready() {
            self.state = WorkerState::Draining;
            self.finish_drain();
        }
    }

    fn invalidate_pending(&mut self) {
        self.pending.fill(None);
        self.fallback_transport.pending_requests.clear();
    }

    /// Frame/model changes retire callbacks without waiting on the browser event loop.
    pub fn set_completion_context(&mut self, frame: u64, model: u64) -> Result<(), WorkerError> {
        if frame < self.stamp.frame || model < self.stamp.model {
            return Err(WorkerError::StaleGeneration);
        }
        if frame != self.stamp.frame || model != self.stamp.model {
            self.invalidate_pending();
            self.stamp.frame = frame;
            self.stamp.model = model;
            self.finish_drain();
        }
        Ok(())
    }

    pub fn notify_device_lost(&mut self) {
        if matches!(self.state, WorkerState::Suspended | WorkerState::Terminated) {
            return;
        }
        self.invalidate_pending();
        match self.stamp.device.checked_add(1) {
            Some(device) => {
                self.stamp.device = device;
                self.state = WorkerState::Suspended;
            }
            None => self.terminate(),
        }
    }

    pub fn recover(&mut self, new_generation: u32) -> Result<(), WorkerError> {
        if self.state != WorkerState::Suspended {
            return Err(WorkerError::NotReady);
        }
        if new_generation <= self.stamp.session {
            return Err(WorkerError::StaleGeneration);
        }
        self.invalidate_pending();
        self.stamp.session = new_generation;
        self.state = WorkerState::Ready;
        Ok(())
    }

    pub fn terminate(&mut self) {
        self.invalidate_pending();
        self.detach_offscreen_canvas();
        self.state = WorkerState::Terminated;
    }
}
