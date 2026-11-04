use crate::ordinary::gpu_params::*;
pub struct PrefillLayerBinds {
    pub rms_attn: wgpu::BindGroup,
    /// Dual K+V when SoA (shared act); else use k_gemm/v_gemm.
    pub dual_kv: Option<wgpu::BindGroup>,
    pub k_gemm: wgpu::BindGroup,
    pub k_write: wgpu::BindGroup,
    pub v_gemm: wgpu::BindGroup,
    pub v_write: wgpu::BindGroup,
    pub q_gemm: wgpu::BindGroup,
    pub q: wgpu::BindGroup,
    /// O-proj + residual → hidden_b (coop residual GEMV); drops separate add1.
    pub o_resid: wgpu::BindGroup,
    pub rms_ffn: wgpu::BindGroup,
    /// T-A1 fused SwiGLU when present; else gate/up/silu.
    pub fused_ffn: Option<wgpu::BindGroup>,
    pub gate: wgpu::BindGroup,
    pub up: wgpu::BindGroup,
    pub silu: wgpu::BindGroup,
    /// Down-proj + residual → hidden_a; drops separate add2.
    pub down_resid: wgpu::BindGroup,
}

/// Per-layer prototype params: everything set except the per-call batch fields
/// (`n_batch`/`batch`/`num_tokens_in_batch`/`batch_start_token_idx`).
pub struct PrefillLayerProtos {
    /// K, V, Q, O, gate, up, down.
    pub gemm: [GemmGpuParams; 7],
    /// k_write, v_write, q-sdpa.
    pub attn: [AttentionGpuParams; 3],
    /// rms_attn, add1, rms_ffn, silu, add2.
    pub elem: [ElemGpuParams; 5],
}

pub struct PrefillArenaPlan {
    pub map_status: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// (model byte base, tensor_data_start, n_layer); invalidated on model swap.
    pub key: (u64, u64, u32),
    pub n_embd: usize,
    pub n_ffn: usize,
    pub kv_dim: usize,
    pub q_dim: usize,
    pub n_head: u32,
    pub n_kv_head: u32,
    pub max_context: u32,
    pub protos: Vec<PrefillLayerProtos>,
    pub layers: Vec<PrefillLayerBinds>,
    /// Uniform arena: `n_layer × 14` slots, rewritten per call with the batch fields.
    pub param_arena: wgpu::Buffer,
    /// Reused CPU staging for the param arena (no per-call heap alloc).
    pub scratch: Vec<u8>,
    /// Embedded-token upload target; also the residual stream (layer in/out).
    pub hidden_a: wgpu::Buffer,
    pub use_coop: bool,
}

pub enum PrefillArenaState {
    Unbuilt,
    /// Build failed for this model — don't retry every prefill.
    Ineligible((u64, u64, u32)),
    Ready(Box<PrefillArenaPlan>),
}
