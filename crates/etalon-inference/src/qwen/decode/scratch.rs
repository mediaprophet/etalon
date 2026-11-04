use super::*;
/// All per-step scratch, allocated once and shared across every layer.  The
/// GDN and QSA paths run sequentially, so one set of buffers serves both.
pub struct Qwen4ExpDecodeScratch {
    pub(super) raw_row: Vec<u8>,
    pub(super) attn_raw: Vec<u8>,
    pub(super) ffn_raw: Vec<u8>,
    pub(super) embedding: Vec<f32>,
    pub(super) mixed: Vec<f32>,
    pub(super) attn_hyper_norm: Vec<f32>,
    pub(super) attn_hyper_normalized: Vec<f32>,
    pub(super) attn_hyper_low_rank: Vec<f32>,
    pub(super) attn_hyper_gates: Vec<f32>,
    pub(super) attn_hyper_projection: Vec<f32>,
    pub(super) ffn_hyper_norm: Vec<f32>,
    pub(super) ffn_hyper_normalized: Vec<f32>,
    pub(super) ffn_hyper_low_rank: Vec<f32>,
    pub(super) ffn_hyper_gates: Vec<f32>,
    pub(super) ffn_hyper_projection: Vec<f32>,
    pub(super) gdn_raw: Vec<u8>,
    pub(super) gdn_projection: Vec<f32>,
    pub(super) gdn_qkv: Vec<f32>,
    pub(super) gdn_gate: Vec<f32>,
    pub(super) gdn_alpha: Vec<f32>,
    pub(super) gdn_beta: Vec<f32>,
    pub(super) gdn_convolved: Vec<f32>,
    pub(super) gdn_head_norm: Vec<f32>,
    pub(super) gdn_head_vector: Vec<f32>,
    pub(super) gdn_output_inner: Vec<f32>,
    pub(super) gdn_head_a: Vec<f32>,
    pub(super) gdn_dt_bias: Vec<f32>,
    pub(super) qsa_raw: Vec<u8>,
    pub(super) qsa_projection: Vec<f32>,
    pub(super) qsa_q_full: Vec<f32>,
    pub(super) qsa_key: Vec<f32>,
    pub(super) qsa_value: Vec<f32>,
    pub(super) qsa_index_query: Vec<f32>,
    pub(super) qsa_index_key: Vec<f32>,
    pub(super) qsa_q_norm: Vec<f32>,
    pub(super) qsa_k_norm: Vec<f32>,
    pub(super) qsa_index_q_norm: Vec<f32>,
    pub(super) qsa_index_k_norm: Vec<f32>,
    pub(super) qsa_selected: Vec<usize>,
    pub(super) qsa_scores: Vec<f32>,
    pub(super) qsa_block_selected: Vec<usize>,
    pub(super) qsa_block_scores: Vec<f32>,
    pub(super) qsa_attention: Vec<f32>,
    pub(super) moe_raw: Vec<u8>,
    pub(super) moe_dequantized: Vec<f32>,
    pub(super) moe_router_logits: Vec<f32>,
    pub(super) moe_top_indices: [usize; 10],
    pub(super) moe_top_logits: [f32; 10],
    pub(super) moe_top_weights: [f32; 10],
    pub(super) moe_gate: Vec<f32>,
    pub(super) moe_up: Vec<f32>,
    pub(super) moe_activation: Vec<f32>,
    pub(super) moe_expert_out: Vec<f32>,
    pub(super) attn_mixed: Vec<f32>,
    pub(super) token_mixer_branch: Vec<f32>,
    pub(super) ffn_mixed: Vec<f32>,
    pub(super) moe_branch: Vec<f32>,
    pub(super) attn_inject: [f32; 4],
    pub(super) ffn_inject: [f32; 4],
    pub(super) ple_raw: Vec<u8>,
    pub(super) ple_projection: Vec<f32>,
    pub(super) ple_embedding: Vec<f32>,
    pub(super) ple_key: Vec<f32>,
    pub(super) ple_value: Vec<f32>,
    pub(super) ple_query_norm: Vec<f32>,
    pub(super) ple_gated: Vec<f32>,
    pub(super) ple_conv_norm: Vec<f32>,
    pub(super) ple_conv_weights: Vec<f32>,
    pub(super) ple_norm_key: Vec<f32>,
    pub(super) ple_norm_query: Vec<f32>,
    pub(super) ple_norm_conv: Vec<f32>,
    pub(super) argmax_row: Vec<f32>,
    pub(super) footprint: u64,
}

/// Counting allocator for the fixed decode-scratch set, so receipts report a
/// measured footprint rather than a hand-computed constant.
struct ScratchAlloc {
    pub(super) bytes: u64,
}

impl ScratchAlloc {
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, Qwen4ExpDecodeError> {
        self.bytes += (n * 4) as u64;
        allocation::zeros(n)
    }
    fn u8s(&mut self, n: usize) -> Result<Vec<u8>, Qwen4ExpDecodeError> {
        self.bytes += n as u64;
        allocation::zeros(n)
    }
    fn usizes(&mut self, n: usize) -> Result<Vec<usize>, Qwen4ExpDecodeError> {
        self.bytes += (n * core::mem::size_of::<usize>()) as u64;
        allocation::zeros(n)
    }
}

impl Qwen4ExpDecodeScratch {
    /// Creates a default `Qwen4ExpDecodeScratch`.
    pub fn new(hidden: usize) -> Self {
        Self::new_with_budget(hidden, u64::MAX).expect("Qwen decode scratch allocation failed")
    }

    pub fn new_with_budget(hidden: usize, byte_budget: u64) -> Result<Self, Qwen4ExpDecodeError> {
        let mut alloc = ScratchAlloc { bytes: 0 };
        let wide = hidden
            .checked_mul(QWEN4EXP_HYPER_STREAMS)
            .ok_or(Qwen4ExpDecodeError::SizeOverflow)?;
        wide.checked_mul(16)
            .ok_or(Qwen4ExpDecodeError::SizeOverflow)?;
        let q_width = QWEN4EXP_QSA_QUERY_HEADS * QWEN4EXP_QSA_HEAD_DIM;
        let kv_width = QWEN4EXP_QSA_KV_HEADS * QWEN4EXP_QSA_HEAD_DIM;
        let gdn_inner = QWEN4EXP_GDN_HEADS * QWEN4EXP_GDN_HEAD_DIM;
        let mut required = 0usize;
        allocation::add_bytes(&mut required, RAW_ROW_BYTES, 1)?;
        allocation::add_bytes(&mut required, RAW_ROW_BYTES, 1)?;
        allocation::add_bytes(&mut required, RAW_ROW_BYTES, 1)?;
        allocation::add_bytes(&mut required, hidden, 4)?;
        allocation::add_bytes(&mut required, hidden, 4)?;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, HYPER_LOW_RANK, 4)?;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, HYPER_LOW_RANK, 4)?;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, RAW_ROW_BYTES, 1)?;
        allocation::add_bytes(&mut required, gdn_inner, 4)?;
        allocation::add_bytes(&mut required, GDN_CONV_WIDTH, 4)?;
        allocation::add_bytes(&mut required, gdn_inner, 4)?;
        allocation::add_bytes(&mut required, QWEN4EXP_GDN_HEADS, 4)?;
        allocation::add_bytes(&mut required, QWEN4EXP_GDN_HEADS, 4)?;
        allocation::add_bytes(&mut required, GDN_CONV_WIDTH, 4)?;
        allocation::add_bytes(&mut required, QWEN4EXP_GDN_HEAD_DIM, 4)?;
        allocation::add_bytes(&mut required, QWEN4EXP_GDN_HEAD_DIM, 4)?;
        allocation::add_bytes(&mut required, gdn_inner, 4)?;
        allocation::add_bytes(&mut required, QWEN4EXP_GDN_HEADS, 4)?;
        allocation::add_bytes(&mut required, QWEN4EXP_GDN_HEADS, 4)?;
        allocation::add_bytes(&mut required, RAW_ROW_BYTES, 1)?;
        allocation::add_bytes(&mut required, q_width, 4)?;
        allocation::add_bytes(&mut required, q_width * 2, 4)?;
        allocation::add_bytes(&mut required, kv_width, 4)?;
        allocation::add_bytes(&mut required, kv_width, 4)?;
        allocation::add_bytes(&mut required, QSA_INDEX_QUERY_WIDTH, 4)?;
        allocation::add_bytes(&mut required, QWEN4EXP_QSA_INDEXER_HEAD_DIM, 4)?;
        allocation::add_bytes(&mut required, QWEN4EXP_QSA_HEAD_DIM, 4)?;
        allocation::add_bytes(&mut required, QWEN4EXP_QSA_HEAD_DIM, 4)?;
        allocation::add_bytes(&mut required, QWEN4EXP_QSA_INDEXER_HEAD_DIM, 4)?;
        allocation::add_bytes(&mut required, QWEN4EXP_QSA_INDEXER_HEAD_DIM, 4)?;
        allocation::add_bytes(
            &mut required,
            QWEN4EXP_QSA_MAX_SELECTED,
            core::mem::size_of::<usize>(),
        )?;
        allocation::add_bytes(&mut required, QWEN4EXP_QSA_MAX_SELECTED, 4)?;
        allocation::add_bytes(
            &mut required,
            QWEN4EXP_QSA_TOP_BLOCKS,
            core::mem::size_of::<usize>(),
        )?;
        allocation::add_bytes(&mut required, QWEN4EXP_QSA_TOP_BLOCKS, 4)?;
        allocation::add_bytes(&mut required, q_width, 4)?;
        allocation::add_bytes(&mut required, RAW_ROW_BYTES, 1)?;
        allocation::add_bytes(&mut required, hidden, 4)?;
        allocation::add_bytes(&mut required, ROUTER_EXPERTS, 4)?;
        allocation::add_bytes(&mut required, MOE_INTERMEDIATE, 4)?;
        allocation::add_bytes(&mut required, MOE_INTERMEDIATE, 4)?;
        allocation::add_bytes(&mut required, MOE_INTERMEDIATE, 4)?;
        allocation::add_bytes(&mut required, hidden, 4)?;
        allocation::add_bytes(&mut required, hidden, 4)?;
        allocation::add_bytes(&mut required, hidden, 4)?;
        allocation::add_bytes(&mut required, hidden, 4)?;
        allocation::add_bytes(&mut required, hidden, 4)?;
        allocation::add_bytes(&mut required, RAW_ROW_BYTES, 1)?;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, hidden, 4)?;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, hidden, 4)?;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, wide * 4, 4)?;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, wide, 4)?;
        allocation::add_bytes(&mut required, hidden, 4)?;
        allocation::admit(required, byte_budget)?;
        Ok(Self {
            raw_row: alloc.u8s(RAW_ROW_BYTES)?,
            attn_raw: alloc.u8s(RAW_ROW_BYTES)?,
            ffn_raw: alloc.u8s(RAW_ROW_BYTES)?,
            embedding: alloc.f32s(hidden)?,
            mixed: alloc.f32s(hidden)?,
            attn_hyper_norm: alloc.f32s(wide)?,
            attn_hyper_normalized: alloc.f32s(wide)?,
            attn_hyper_low_rank: alloc.f32s(HYPER_LOW_RANK)?,
            attn_hyper_gates: alloc.f32s(wide)?,
            attn_hyper_projection: alloc.f32s(wide)?,
            ffn_hyper_norm: alloc.f32s(wide)?,
            ffn_hyper_normalized: alloc.f32s(wide)?,
            ffn_hyper_low_rank: alloc.f32s(HYPER_LOW_RANK)?,
            ffn_hyper_gates: alloc.f32s(wide)?,
            ffn_hyper_projection: alloc.f32s(wide)?,
            gdn_raw: alloc.u8s(RAW_ROW_BYTES)?,
            gdn_projection: alloc.f32s(gdn_inner)?,
            gdn_qkv: alloc.f32s(GDN_CONV_WIDTH)?,
            gdn_gate: alloc.f32s(gdn_inner)?,
            gdn_alpha: alloc.f32s(QWEN4EXP_GDN_HEADS)?,
            gdn_beta: alloc.f32s(QWEN4EXP_GDN_HEADS)?,
            gdn_convolved: alloc.f32s(GDN_CONV_WIDTH)?,
            gdn_head_norm: alloc.f32s(QWEN4EXP_GDN_HEAD_DIM)?,
            gdn_head_vector: alloc.f32s(QWEN4EXP_GDN_HEAD_DIM)?,
            gdn_output_inner: alloc.f32s(gdn_inner)?,
            gdn_head_a: alloc.f32s(QWEN4EXP_GDN_HEADS)?,
            gdn_dt_bias: alloc.f32s(QWEN4EXP_GDN_HEADS)?,
            qsa_raw: alloc.u8s(RAW_ROW_BYTES)?,
            qsa_projection: alloc.f32s(q_width)?,
            qsa_q_full: alloc.f32s(q_width * 2)?,
            qsa_key: alloc.f32s(kv_width)?,
            qsa_value: alloc.f32s(kv_width)?,
            qsa_index_query: alloc.f32s(QSA_INDEX_QUERY_WIDTH)?,
            qsa_index_key: alloc.f32s(QWEN4EXP_QSA_INDEXER_HEAD_DIM)?,
            qsa_q_norm: alloc.f32s(QWEN4EXP_QSA_HEAD_DIM)?,
            qsa_k_norm: alloc.f32s(QWEN4EXP_QSA_HEAD_DIM)?,
            qsa_index_q_norm: alloc.f32s(QWEN4EXP_QSA_INDEXER_HEAD_DIM)?,
            qsa_index_k_norm: alloc.f32s(QWEN4EXP_QSA_INDEXER_HEAD_DIM)?,
            qsa_selected: alloc.usizes(QWEN4EXP_QSA_MAX_SELECTED)?,
            qsa_scores: alloc.f32s(QWEN4EXP_QSA_MAX_SELECTED)?,
            qsa_block_selected: alloc.usizes(QWEN4EXP_QSA_TOP_BLOCKS)?,
            qsa_block_scores: alloc.f32s(QWEN4EXP_QSA_TOP_BLOCKS)?,
            qsa_attention: alloc.f32s(q_width)?,
            moe_raw: alloc.u8s(RAW_ROW_BYTES)?,
            moe_dequantized: alloc.f32s(hidden)?,
            moe_router_logits: alloc.f32s(ROUTER_EXPERTS)?,
            moe_top_indices: [0; 10],
            moe_top_logits: [0.0; 10],
            moe_top_weights: [0.0; 10],
            moe_gate: alloc.f32s(MOE_INTERMEDIATE)?,
            moe_up: alloc.f32s(MOE_INTERMEDIATE)?,
            moe_activation: alloc.f32s(MOE_INTERMEDIATE)?,
            moe_expert_out: alloc.f32s(hidden)?,
            attn_mixed: alloc.f32s(hidden)?,
            token_mixer_branch: alloc.f32s(hidden)?,
            ffn_mixed: alloc.f32s(hidden)?,
            moe_branch: alloc.f32s(hidden)?,
            attn_inject: [0.0; 4],
            ffn_inject: [0.0; 4],
            ple_raw: alloc.u8s(RAW_ROW_BYTES)?,
            ple_projection: alloc.f32s(wide)?,
            ple_embedding: alloc.f32s(hidden)?,
            ple_key: alloc.f32s(wide)?,
            ple_value: alloc.f32s(hidden)?,
            ple_query_norm: alloc.f32s(wide)?,
            ple_gated: alloc.f32s(wide)?,
            ple_conv_norm: alloc.f32s(wide)?,
            ple_conv_weights: alloc.f32s(wide * 4)?,
            ple_norm_key: alloc.f32s(wide)?,
            ple_norm_query: alloc.f32s(wide)?,
            ple_norm_conv: alloc.f32s(wide)?,
            argmax_row: alloc.f32s(hidden)?,
            footprint: alloc.bytes,
        })
    }

    /// Measured bytes held by this scratch set (fixed at construction).
    pub fn scratch_bytes(&self) -> u64 {
        self.footprint
    }

    /// The hidden vector produced by the most recent final mixer step —
    /// what the vocabulary projection consumed.
    pub fn final_mixed(&self) -> &[f32] {
        &self.mixed
    }
}
