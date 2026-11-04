pub const SLOT: wgpu::BufferAddress = 256;
pub const RESIDENT_LOGITS_CHUNK: usize = 131_072;

pub const GEMM_SLOTS_PER_LAYER: u64 = 7;
pub const ATTN_SLOTS_PER_LAYER: u64 = 3;

pub const ELEM_SLOT_RMS: wgpu::BufferAddress = 0;
pub const ELEM_SLOT_ADD: wgpu::BufferAddress = SLOT;
pub const ELEM_SLOT_SILU: wgpu::BufferAddress = SLOT * 2;
pub const ELEM_SLOTS: u64 = 3;
