//! Unified Decode (B=1), Prefill (B<=64), and Verify (B<=16) forward execution arena.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForwardMode {
    Prefill { batch_len: u32 },
    DecodeSingle { pos: u32 },
    SpeculativeVerify { start_pos: u32, batch_len: u32 },
}

pub struct ForwardArena {
    pub max_batch: u32,
    pub hidden_dim: u32,
    pub num_layers: u32,
}

impl ForwardArena {
    pub fn new(max_batch: u32, hidden_dim: u32, num_layers: u32) -> Self {
        Self {
            max_batch,
            hidden_dim,
            num_layers,
        }
    }

    /// Readback size in bytes for a given forward mode.
    /// Eliminates full logit tensor readbacks (reduces 9.72 MB -> 4-64 bytes).
    pub fn readback_byte_size(&self, mode: ForwardMode) -> usize {
        match mode {
            ForwardMode::Prefill { .. } => 0,
            ForwardMode::DecodeSingle { .. } => 4,
            ForwardMode::SpeculativeVerify { batch_len, .. } => (batch_len as usize) * 4,
        }
    }
}
