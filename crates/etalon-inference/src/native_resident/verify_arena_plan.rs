use crate::ordinary::gpu_params::*;
pub const VERIFY_BATCH_MAX: usize = 16;
pub struct VerifyLayerBinds {
    pub rms_attn: wgpu::BindGroup,
    pub k_gemm: wgpu::BindGroup,
    pub k_write: wgpu::BindGroup,
    pub v_gemm: wgpu::BindGroup,
    pub v_write: wgpu::BindGroup,
    pub q: wgpu::BindGroup,
    pub o: wgpu::BindGroup,
    pub add1: wgpu::BindGroup,
    pub rms_ffn: wgpu::BindGroup,
    pub gate: wgpu::BindGroup,
    pub up: wgpu::BindGroup,
    pub silu: wgpu::BindGroup,
    pub down: wgpu::BindGroup,
    pub add2: wgpu::BindGroup,
}

pub struct VerifyLayerProtos {
    pub gemm: [GemmGpuParams; 6],
    pub attn: [AttentionGpuParams; 3],
    pub elem: [ElemGpuParams; 5],
}

/// One vocab chunk's logits GEMV (writes `B` rows into `logits_out` at the chunk offset).
pub struct VerifyChunk {
    pub gemm: wgpu::BindGroup,
    pub rows: u32,
    /// GEMM param slot in the arena.
    pub slot: u64,
    /// Constant GEMM params (only `n_batch` varies per call).
    pub params: GemmGpuParams,
}

pub struct VerifyArenaPlan {
    pub map_status: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// (model byte base, tensor_data_start, n_layer); invalidated on model swap.
    pub key: (u64, u64, u32),
    pub n_embd: usize,
    pub n_ffn: usize,
    pub kv_dim: usize,
    pub n_head: u32,
    pub n_kv_head: u32,
    pub vocab: usize,
    pub max_context: u32,
    pub protos: Vec<VerifyLayerProtos>,
    pub layers: Vec<VerifyLayerBinds>,
    /// Output RMSNorm (hidden_a → normed) bind group; elem param at `tail_rms_slot`.
    pub rms_out: wgpu::BindGroup,
    pub tail_rms_slot: u64,
    pub chunks: Vec<VerifyChunk>,
    pub param_arena: wgpu::Buffer,
    pub scratch: Vec<u8>,
    /// Embedded-token upload target + residual stream.
    pub embedding_scratch: Vec<f32>,
    pub hidden_a: wgpu::Buffer,
    /// `B × vocab` logits (STORAGE|COPY_SRC).
    pub logits_out: wgpu::Buffer,
    /// `B × vocab` readback staging (MAP_READ|COPY_DST).
    pub logits_staging: wgpu::Buffer,
    pub use_coop: bool,
}

pub enum VerifyArenaState {
    Unbuilt,
    Ineligible((u64, u64, u32)),
    Ready(Box<VerifyArenaPlan>),
}
