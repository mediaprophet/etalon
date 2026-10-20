//! Transferable ArrayBuffer fallback when SharedArrayBuffer is unavailable.

pub struct TransferableMessageTransport {
    pub pending_messages: usize,
}
