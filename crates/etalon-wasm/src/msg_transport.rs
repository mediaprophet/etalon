//! Transferable ArrayBuffer fallback when SharedArrayBuffer is unavailable.
//!
//! Handles framed message transfer via standard WebWorker `postMessage`
//! with transferable ArrayBuffers when cross-origin isolation (COOP/COEP) is absent.

use bytemuck::{Pod, Zeroable};

/// Framed message header preceding every transferable payload.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct MessageHeader {
    pub msg_id: u64,
    pub msg_type: u32,
    pub payload_len: u32,
}

const _: () = {
    assert!(core::mem::size_of::<MessageHeader>() == 16);
};

/// A transferable packet ready for zero-copy ownership transfer across worker boundaries.
#[derive(Clone, Debug, PartialEq)]
pub struct TransferablePacket {
    pub header: MessageHeader,
    pub payload: Vec<u8>,
}

impl TransferablePacket {
    pub fn new(msg_id: u64, msg_type: u32, payload: Vec<u8>) -> Self {
        let payload_len = payload.len() as u32;
        Self {
            header: MessageHeader {
                msg_id,
                msg_type,
                payload_len,
            },
            payload,
        }
    }

    /// Serialize into a single contiguous byte buffer ready for postMessage transferable array.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.payload.len());
        out.extend_from_slice(bytemuck::bytes_of(&self.header));
        out.extend_from_slice(&self.payload);
        out
    }

    /// Parse a transferable byte buffer into a structured message packet.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 16 {
            return None;
        }
        let header_bytes: [u8; 16] = bytes[0..16].try_into().ok()?;
        let header: MessageHeader = bytemuck::cast(header_bytes);
        if bytes.len() < 16 + (header.payload_len as usize) {
            return None;
        }
        let payload = bytes[16..16 + (header.payload_len as usize)].to_vec();
        Some(Self { header, payload })
    }
}

/// Fallback transport managing queued transferable message packets.
pub struct TransferableMessageTransport {
    pub pending_requests: Vec<u64>,
}

impl Default for TransferableMessageTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl TransferableMessageTransport {
    pub fn new() -> Self {
        Self {
            pending_requests: Vec::new(),
        }
    }

    pub fn track_request(&mut self, msg_id: u64) {
        self.pending_requests.push(msg_id);
    }

    pub fn complete_request(&mut self, msg_id: u64) -> bool {
        if let Some(pos) = self.pending_requests.iter().position(|&id| id == msg_id) {
            self.pending_requests.swap_remove(pos);
            true
        } else {
            false
        }
    }

    pub fn pending_count(&self) -> usize {
        self.pending_requests.len()
    }
}
