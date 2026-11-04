use crate::ordinary::gpu_params::*;
pub struct LayerBinds {
    pub rms_attn: wgpu::BindGroup,
    /// Triple Q+K+V GEMV (shared act, GQA) when SoA — preferred over dual+q.
    pub triple_qkv: Option<wgpu::BindGroup>,
    /// Dual K+V GEMV (shared act) when SoA and triple unavailable; else k_gemm/v_gemm.
    pub dual_kv: Option<wgpu::BindGroup>,
    pub k_gemm: wgpu::BindGroup,
    pub k_write: wgpu::BindGroup,
    pub v_gemm: wgpu::BindGroup,
    pub v_write: wgpu::BindGroup,
    /// Coop GEMV: normed → q_proj (decoupled from attention shader).
    pub q_gemm: wgpu::BindGroup,
    /// Q-SDPA over precomputed Q (`proj_row_stride = q_dim`).
    pub q: wgpu::BindGroup,
    /// O-proj + residual (coop_gemv_residual) → hidden_b.
    pub o_resid: wgpu::BindGroup,
    pub rms_ffn: wgpu::BindGroup,
    /// T-A1: silu(gate·x)·(up·x) in one pass when set; else use gate/up/silu.
    pub fused_ffn: Option<wgpu::BindGroup>,
    pub gate: Option<wgpu::BindGroup>,
    pub up: Option<wgpu::BindGroup>,
    pub silu: Option<wgpu::BindGroup>,
    /// Down-proj + residual (coop_gemv_residual) → hidden_a.
    pub down_resid: wgpu::BindGroup,
}

/// One output-projection vocab chunk (GEMV + top-1 reduction).
pub struct OutChunk {
    pub gemm: wgpu::BindGroup,
    pub topk: wgpu::BindGroup,
    pub rows: u32,
    pub cand_count: usize,
}

/// Per-layer prototype attention params; per token only the position/mask
/// fields are patched before the single dynamic-arena upload.
pub struct LayerProtos {
    pub k_write: AttentionGpuParams,
    pub v_write: AttentionGpuParams,
    pub q: AttentionGpuParams,
}

pub struct ResidentDecodePlan {
    /// (mmap base, tensor_data_start, n_layer) — invalidates on model swap.
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
    pub layer_protos: Vec<LayerProtos>,
    pub layers: Vec<LayerBinds>,
    pub rms_out: wgpu::BindGroup,
    pub out_chunks: Vec<OutChunk>,
    pub total_cands: usize,
    /// Dynamic (per-token) attention uniform arena: `n_layer × 3` slots.
    pub attn_dyn_arena: wgpu::Buffer,
    /// Reused CPU staging for the dynamic arena (no per-token heap alloc).
    pub dyn_scratch: Vec<u8>,
    /// Embedded-token upload target; also the residual stream (layer in/out).
    pub hidden_a: wgpu::Buffer,
    /// Post-output-norm hidden (logits GEMV input); `COPY_SRC` for sample-path readback.
    pub normed: wgpu::Buffer,
    /// Candidate readback staging: `[vals(total_cands) | idxs(total_cands)]`.
    pub staging: wgpu::Buffer,
    /// Full-hidden MAP_READ staging for sampler-compatible resident (n_embd f32).
    pub hidden_staging: wgpu::Buffer,
    pub use_coop: bool,
    /// Multi-row coop GEMV (8 rows/WG) — armed for Q4_K_SOA decode (3B bandwidth path).
    pub use_multirow: bool,
    /// Warp GEMV (32 thr/row) for Q4_K_SOA — more FMA/thread than 256-wide coop.
    pub use_warp: bool,
    /// Multi-row fused FFN (4 rows/WG) for Q4_K_SOA.
    pub use_ffn_mr: bool,
    /// Warp fused FFN (32 thr/row) for Q4_K_SOA.
    pub use_ffn_warp: bool,
    /// True when every layer has fused_ffn bind groups (T-A1).
    /// Read by lab audit via `ffn_fusion_in_resident()`; kept on plan for diagnostics.
    #[allow(dead_code)]
    pub use_fused_ffn: bool,
    /// Exact compute dispatches encoded for one token on this prepared plan.
    pub dispatches_per_token: u32,
    /// Candidate bytes copied into the MAP_READ staging buffer per token.
    pub readback_bytes_per_token: u32,
}

pub enum ResidentDecodeState {
    Unbuilt,
    /// Build failed for this model — don't retry every token.
    Ineligible((u64, u64, u32)),
    Ready(Box<ResidentDecodePlan>),
}

/// Greedy → argmax token; sample path → post-norm hidden was written to the caller buffer.
pub enum ResidentTokenOutcome {
    Argmax(ResidentArgmax),
    HiddenReady,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidentArgmax {
    pub best_token_id: u32,
    pub max_logit: f32,
}
