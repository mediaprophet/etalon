//! Unified Decode (B=1), Prefill (B<=64), and Verify (B<=16) forward execution arena.
//!
//! Collapses the three duplicated arenas into a single unified allocation pool,
//! eliminating ~420 bind groups per model and reducing speculative verify readback
//! from 9.72 MB down to 4-64 bytes.

mod allocation;
mod readback;

pub use allocation::{TensorArenaError, DEFAULT_FORWARD_ARENA_BUDGET};
pub use readback::TokenReadbackError;

/// Operational mode of the forward execution arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForwardMode {
    /// Populate KV cache without full logit GEMM or norm readback.
    Prefill { batch_len: u32 },
    /// Single-token autoregressive decode step (pos, reads back 4 bytes).
    DecodeSingle { pos: u32 },
    /// Speculative multi-token verification (start_pos, batch_len <= 16, reads back batch_len * 4 bytes).
    SpeculativeVerify { start_pos: u32, batch_len: u32 },
}

/// Unified forward execution arena on an Etalon device.
pub struct ForwardArena {
    readback_busy: std::sync::atomic::AtomicBool,
    pub max_batch: u32,
    pub hidden_dim: u32,
    pub num_layers: u32,
    pub hidden_state_buf: etalon::Buffer,
    pub residual_buf: etalon::Buffer,
    pub norm_buf: etalon::Buffer,
    pub readback_stage_buf: etalon::Buffer,
}

impl ForwardArena {
    pub fn new(device: &etalon::Device, max_batch: u32, hidden_dim: u32, num_layers: u32) -> Self {
        Self::try_new_with_budget(
            device,
            max_batch,
            hidden_dim,
            num_layers,
            DEFAULT_FORWARD_ARENA_BUDGET,
        )
        .expect("ForwardArena allocation rejected")
    }

    pub fn try_new_with_budget(
        device: &etalon::Device,
        max_batch: u32,
        hidden_dim: u32,
        num_layers: u32,
        byte_budget: u64,
    ) -> Result<Self, TensorArenaError> {
        let sizes = allocation::AllocationSizes::checked(
            max_batch,
            hidden_dim,
            num_layers,
            &device.limits(),
            byte_budget,
        )?;
        let byte_size = sizes.storage;

        let usage_st = etalon::BufferUsages::STORAGE
            | etalon::BufferUsages::COPY_SRC
            | etalon::BufferUsages::COPY_DST;

        let hidden_state_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-forward-arena-hidden"),
            size: byte_size.max(256),
            usage: usage_st,
            mapped_at_creation: false,
        });

        let residual_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-forward-arena-residual"),
            size: byte_size.max(256),
            usage: usage_st,
            mapped_at_creation: false,
        });

        let norm_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-forward-arena-norm"),
            size: byte_size.max(256),
            usage: usage_st,
            mapped_at_creation: false,
        });

        let readback_stage_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-forward-arena-readback-stage"),
            size: sizes.staging,
            usage: etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            readback_busy: std::sync::atomic::AtomicBool::new(false),
            max_batch,
            hidden_dim,
            num_layers,
            hidden_state_buf,
            residual_buf,
            norm_buf,
            readback_stage_buf,
        })
    }

    /// Readback size in bytes for a given forward mode.
    /// Eliminates full logit tensor readbacks (reduces 9.72 MB -> 4-64 bytes).
    #[inline(always)]
    pub fn readback_byte_size(&self, mode: ForwardMode) -> usize {
        usize::try_from(
            self.try_readback_byte_size(mode)
                .expect("invalid forward readback mode"),
        )
        .expect("forward readback size exceeds address space")
    }

    pub fn try_readback_byte_size(&self, mode: ForwardMode) -> Result<u64, TensorArenaError> {
        let batch = match mode {
            ForwardMode::Prefill { batch_len } => batch_len,
            ForwardMode::DecodeSingle { .. } => 1,
            ForwardMode::SpeculativeVerify { batch_len, .. } => {
                if batch_len > 16 {
                    return Err(TensorArenaError::InvalidBatch);
                }
                batch_len
            }
        };
        if batch == 0 || batch > self.max_batch {
            return Err(TensorArenaError::InvalidBatch);
        }
        Ok(if matches!(mode, ForwardMode::Prefill { .. }) {
            0
        } else {
            u64::from(batch) * 4
        })
    }

    /// Record a copy of the final token index / top-k candidates into the readback staging buffer.
    pub fn record_readback_copy(
        &self,
        encoder: &mut etalon::CommandEncoder,
        token_src: &etalon::Buffer,
        mode: ForwardMode,
    ) {
        self.try_record_readback_copy(encoder, token_src, mode)
            .expect("invalid forward readback copy");
    }

    pub fn try_record_readback_copy(
        &self,
        encoder: &mut etalon::CommandEncoder,
        token_src: &etalon::Buffer,
        mode: ForwardMode,
    ) -> Result<(), TensorArenaError> {
        let size = self.try_readback_byte_size(mode)?;
        if size > token_src.size() || size > self.readback_stage_buf.size() {
            return Err(TensorArenaError::ReadbackBufferTooSmall);
        }
        if size > 0 {
            encoder.copy_buffer_to_buffer(token_src, 0, &self.readback_stage_buf, 0, size);
        }
        Ok(())
    }

    /// Blocking compatibility wrapper; WebGPU callers must await `read_token_async`.
    pub fn read_token_blocking(&self, device: &etalon::Device) -> Option<u32> {
        self.try_read_token_blocking(device).ok()
    }
}
