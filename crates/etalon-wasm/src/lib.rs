//! Dual-transport worker bridge and OffscreenCanvas steward for WebAssembly.

pub mod msg_transport;
pub mod sab_ring;
pub mod worker_steward;

pub use msg_transport::{MessageHeader, TransferableMessageTransport, TransferablePacket};
pub use sab_ring::{SabRingHeader, SharedArrayBufferRing, SAB_RING_MAGIC, SAB_RING_VERSION};
pub use worker_steward::{TransportMode, WorkerState, WorkerSteward};
