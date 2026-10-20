//! Worker event loop steward and OffscreenCanvas ownership manager.

use super::msg_transport::TransferableMessageTransport;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Operational mode of the worker transport layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportMode {
    SharedArrayBufferRing,
    TransferableArrayBuffer,
}

/// Lifecycle state of the dedicated GPU worker steward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerState {
    Initializing,
    Ready,
    Running,
    Suspended,
    Terminated,
}

/// Steward managing dedicated worker GPU execution and canvas transfer.
pub struct WorkerSteward {
    pub transport_mode: TransportMode,
    pub state: WorkerState,
    pub next_request_id: AtomicU64,
    pub has_offscreen_canvas: AtomicBool,
    pub fallback_transport: TransferableMessageTransport,
}

impl WorkerSteward {
    pub fn new(is_isolated: bool) -> Self {
        let transport_mode = if is_isolated {
            TransportMode::SharedArrayBufferRing
        } else {
            TransportMode::TransferableArrayBuffer
        };

        Self {
            transport_mode,
            state: WorkerState::Initializing,
            next_request_id: AtomicU64::new(1),
            has_offscreen_canvas: AtomicBool::new(false),
            fallback_transport: TransferableMessageTransport::new(),
        }
    }

    pub fn set_ready(&mut self) {
        self.state = WorkerState::Ready;
    }

    pub fn attach_offscreen_canvas(&self) -> bool {
        !self.has_offscreen_canvas.swap(true, Ordering::SeqCst)
    }

    pub fn detach_offscreen_canvas(&self) -> bool {
        self.has_offscreen_canvas.swap(false, Ordering::SeqCst)
    }

    pub fn allocate_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn is_ready(&self) -> bool {
        self.state == WorkerState::Ready || self.state == WorkerState::Running
    }
}
