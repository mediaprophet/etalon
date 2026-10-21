//! Full multi-layer transformer mega-pass execution, parameter packs, and single-submission orchestrator.
//!
//! Ported from QualiaDB's `inference::cuda_lane::mega_pass` without global mutable state.
//! Chains RMSNorm, Q4_K / Q8_0 / FP32 weight projections, RoPE, KV cache update, GQA online-softmax
//! SDPA, SwiGLU FFN, output RMSNorm, LM Head, and argmax token selection into a single execution pass.

use std::collections::HashSet;
use thiserror::Error;

use super::attention::online_softmax_rescale;
use super::dml_lane::{f16_bits_to_f32, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_SIZE};

pub const MAX_MEGA_PASS_LAYERS: usize = 128;
pub const MAX_CUDA_CONTEXT_WINDOW: u32 = 32_768;
pub const CUDA_SOA_SLAB_BYTES: u64 = (5 * 1024 * 1024 * 1024) / 2;
pub const Q4_K_SOA_BLOCK_ELEMS: usize = 256;
pub const Q4_K_SOA_BLOCK_BYTES: usize = 144;
pub const Q8_0_BLOCK_ELEMS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MegaPassWeightLayout {
    Fp32,
    Q8_0,
    Q4KSoa,
}

impl MegaPassWeightLayout {
    #[inline]
    pub const fn row_bytes(self, cols: usize) -> Option<usize> {
        if cols == 0 {
            return None;
        }
        match self {
            Self::Fp32 => Some(cols * 4),
            Self::Q8_0 => {
                if !cols.is_multiple_of(Q8_0_BLOCK_ELEMS) {
                    return None;
                }
                Some((cols / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES)
            }
            Self::Q4KSoa => {
                if cols.is_multiple_of(Q4_K_SOA_BLOCK_ELEMS) {
                    Some((cols / Q4_K_SOA_BLOCK_ELEMS) * Q4_K_SOA_BLOCK_BYTES)
                } else if cols.is_multiple_of(Q4_K_BLOCK_SIZE) {
                    Some((cols / Q4_K_BLOCK_SIZE) * Q4_K_BLOCK_BYTES)
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MegaPassLayerWeights<'a> {
    pub attn_norm: &'a [f32],
    pub q_raw: &'a [u8],
    pub k_raw: &'a [u8],
    pub v_raw: &'a [u8],
    pub o_raw: &'a [u8],
    pub ffn_norm: &'a [f32],
    pub gate_raw: &'a [u8],
    pub up_raw: &'a [u8],
    pub down_raw: &'a [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MegaPassLayerDims {
    pub q_in: usize,
    pub q_out: usize,
    pub kv_in: usize,
    pub kv_out: usize,
    pub o_in: usize,
    pub o_out: usize,
    pub gate_in: usize,
    pub gate_out: usize,
    pub up_in: usize,
    pub up_out: usize,
    pub down_in: usize,
    pub down_out: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayerWeightKeys {
    pub kq: u64,
    pub kk: u64,
    pub kvw: u64,
    pub ko: u64,
    pub kg: u64,
    pub ku: u64,
    pub kd: u64,
}

pub trait MegaPassPlanView {
    fn weight_layout(&self) -> MegaPassWeightLayout;
    fn layer_count(&self) -> usize;
    fn layer_weights(&self, layer: usize) -> Option<MegaPassLayerWeights<'_>>;
    fn layer_dims(&self, layer: usize) -> Option<&MegaPassLayerDims>;
    fn layer_weight_keys(&self, layer: usize) -> Option<[u64; 7]>;
    fn layer_norm_keys(&self, layer: usize) -> Option<[u64; 2]>;
}

/// Concrete owned/borrowed mega-pass execution plan implementing [`MegaPassPlanView`].
#[derive(Clone, Debug)]
pub struct MegaPassPlan<'a> {
    pub layout: MegaPassWeightLayout,
    pub layers: Vec<MegaPassLayerWeights<'a>>,
    pub dims: Vec<MegaPassLayerDims>,
    pub weight_keys: Vec<[u64; 7]>,
    pub norm_keys: Vec<[u64; 2]>,
}

impl<'a> MegaPassPlan<'a> {
    pub fn new(
        layout: MegaPassWeightLayout,
        layers: Vec<MegaPassLayerWeights<'a>>,
        dims: Vec<MegaPassLayerDims>,
    ) -> Self {
        let count = layers.len();
        let weight_keys = (0..count)
            .map(|l| {
                let b = ((l + 1) * 16) as u64;
                [b, b + 1, b + 2, b + 3, b + 4, b + 5, b + 6]
            })
            .collect();
        let norm_keys = (0..count)
            .map(|l| {
                let b = ((l + 1) * 256) as u64;
                [b, b + 1]
            })
            .collect();
        Self {
            layout,
            layers,
            dims,
            weight_keys,
            norm_keys,
        }
    }
}

impl<'a> MegaPassPlanView for MegaPassPlan<'a> {
    fn weight_layout(&self) -> MegaPassWeightLayout {
        self.layout
    }
    fn layer_count(&self) -> usize {
        self.layers.len()
    }
    fn layer_weights(&self, layer: usize) -> Option<MegaPassLayerWeights<'_>> {
        self.layers.get(layer).copied()
    }
    fn layer_dims(&self, layer: usize) -> Option<&MegaPassLayerDims> {
        self.dims.get(layer)
    }
    fn layer_weight_keys(&self, layer: usize) -> Option<[u64; 7]> {
        self.weight_keys.get(layer).copied()
    }
    fn layer_norm_keys(&self, layer: usize) -> Option<[u64; 2]> {
        self.norm_keys.get(layer).copied()
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MegaPassError {
    #[error("invalid mega-pass model or step dimensions: {0}")]
    InvalidDimensions(&'static str),
    #[error("heterogeneous layer shapes are not eligible for single-graph capture")]
    HeterogeneousLayerShapes,
    #[error("layer {layer} weight buffer is shorter than required by its quantized row layout")]
    TruncatedLayerWeight { layer: usize },
    #[error("prepared weight key {key:#018x} at layer {layer} is not resident on device")]
    MissingResidentWeight { layer: usize, key: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MegaPassStepDescriptor {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub n_layer: u32,
    pub token_idx: u32,
    pub max_context: u32,
    pub layer_stride: u32,
    pub slot_kv_elems: u32,
    pub kv_block_size: u32,
    pub rope_base: f32,
    pub rope_scale: f32,
    pub rms_eps: f32,
    pub lm_head_in: usize,
    pub lm_head_out: usize,
    pub output_norm_key: Option<u64>,
    pub lm_head_key: Option<u64>,
    pub input_token: Option<(u32, u64)>,
}

impl MegaPassStepDescriptor {
    pub fn validate(&self, plan_layers: usize) -> Result<(), MegaPassError> {
        if self.n_embd == 0
            || self.n_head == 0
            || self.n_kv == 0
            || self.head_dim == 0
            || self.head_dim > 256
            || !self.head_dim.is_multiple_of(2)
            || self.n_kv > self.n_head
            || !self.n_head.is_multiple_of(self.n_kv)
        {
            return Err(MegaPassError::InvalidDimensions(
                "embedding or head dimensions out of bounds",
            ));
        }
        if self.n_layer == 0
            || (self.n_layer as usize) > plan_layers
            || (self.n_layer as usize) > MAX_MEGA_PASS_LAYERS
        {
            return Err(MegaPassError::InvalidDimensions(
                "layer count exceeds plan or MAX_MEGA_PASS_LAYERS",
            ));
        }
        if self.max_context == 0
            || self.max_context > MAX_CUDA_CONTEXT_WINDOW
            || self.token_idx >= self.max_context
            || self.kv_block_size == 0
        {
            return Err(MegaPassError::InvalidDimensions(
                "context window, token index, or KV block size invalid",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MegaPassParameterPacks {
    pub step_params: [u32; 3],
    pub common_qkv: [u32; 5],
    pub common_rope_q: [u32; 5],
    pub common_rope_k: [u32; 5],
    pub common_kv: [u32; 5],
    pub common_sdpa: [u32; 9],
    pub common_o: [u32; 3],
    pub common_ffn: [u32; 4],
    pub common_down: [u32; 3],
    pub common_sdpa_scale: [u32; 1],
    pub common_output_norm: [u32; 2],
    pub common_argmax: [u32; 1],
    pub lm_dims: Option<[u32; 3]>,
    pub params_key: u64,
    pub graph_key: u64,
}

pub fn weight_fingerprint(raw: &[u8], n_in: usize, n_out: usize) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in raw {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    for b in (n_in as u64).to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    for b in (n_out as u64).to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

#[inline]
pub const fn wmma_usable_for_batch(batch_m: usize, n_in: usize, n_out: usize) -> bool {
    batch_m >= 8 && n_in > 0 && n_out > 0 && n_in.is_multiple_of(256) && n_out.is_multiple_of(16)
}

#[inline]
pub const fn attention_segments_for_pos(token_idx: u32) -> u32 {
    let seq_len = token_idx.saturating_add(1);
    let segs = seq_len.div_ceil(256);
    if segs < 1 {
        1
    } else if segs > 32 {
        32
    } else {
        segs
    }
}

pub fn collect_and_validate_layer_keys(
    plan: &impl MegaPassPlanView,
    resident_weights: &HashSet<u64>,
    n_embd: usize,
    n_layer: usize,
    out: &mut [LayerWeightKeys; MAX_MEGA_PASS_LAYERS],
) -> Result<(), MegaPassError> {
    let layout = plan.weight_layout();
    for (layer_idx, slot) in out.iter_mut().enumerate().take(n_layer) {
        let dims = plan
            .layer_dims(layer_idx)
            .ok_or(MegaPassError::InvalidDimensions("missing layer dims"))?;
        let weights = plan
            .layer_weights(layer_idx)
            .ok_or(MegaPassError::InvalidDimensions("missing layer weights"))?;
        let row_in = layout
            .row_bytes(n_embd)
            .ok_or(MegaPassError::InvalidDimensions("invalid n_embd alignment"))?;
        let row_o = layout
            .row_bytes(dims.o_in)
            .ok_or(MegaPassError::InvalidDimensions("invalid o_in alignment"))?;
        let row_gate = layout.row_bytes(dims.gate_in).ok_or(
            MegaPassError::InvalidDimensions("invalid gate_in alignment"),
        )?;
        let row_down = layout.row_bytes(dims.down_in).ok_or(
            MegaPassError::InvalidDimensions("invalid down_in alignment"),
        )?;

        if weights.q_raw.len() < row_in.saturating_mul(dims.q_out)
            || weights.k_raw.len() < row_in.saturating_mul(dims.kv_out)
            || weights.v_raw.len() < row_in.saturating_mul(dims.kv_out)
            || weights.o_raw.len() < row_o.saturating_mul(dims.o_out)
            || weights.gate_raw.len() < row_gate.saturating_mul(dims.gate_out)
            || weights.up_raw.len() < row_gate.saturating_mul(dims.up_out)
            || weights.down_raw.len() < row_down.saturating_mul(dims.down_out)
        {
            return Err(MegaPassError::TruncatedLayerWeight { layer: layer_idx });
        }

        let [kq, kk, kvw, ko, kg, ku, kd] = plan
            .layer_weight_keys(layer_idx)
            .ok_or(MegaPassError::InvalidDimensions("missing layer weight keys"))?;
        let [attn_norm, ffn_norm] = plan
            .layer_norm_keys(layer_idx)
            .ok_or(MegaPassError::InvalidDimensions("missing layer norm keys"))?;

        for key in [kq, kk, kvw, ko, kg, ku, kd, attn_norm, ffn_norm] {
            if !resident_weights.contains(&key) {
                return Err(MegaPassError::MissingResidentWeight {
                    layer: layer_idx,
                    key,
                });
            }
        }
        *slot = LayerWeightKeys {
            kq,
            kk,
            kvw,
            ko,
            kg,
            ku,
            kd,
        };
    }
    Ok(())
}

pub fn build_parameter_packs(
    desc: &MegaPassStepDescriptor,
    plan: &impl MegaPassPlanView,
    first_weight_key: u64,
) -> Result<MegaPassParameterPacks, MegaPassError> {
    desc.validate(plan.layer_count())?;
    let layout = plan.weight_layout();
    let first_dims = *plan
        .layer_dims(0)
        .ok_or(MegaPassError::InvalidDimensions("missing layer 0 dims"))?;
    if (1..desc.n_layer as usize).any(|l| plan.layer_dims(l) != Some(&first_dims)) {
        return Err(MegaPassError::HeterogeneousLayerShapes);
    }

    let q_dim = desc.n_head.saturating_mul(desc.head_dim);
    let kv_dim = desc.n_kv.saturating_mul(desc.head_dim);
    let ring_slot = desc.token_idx % desc.max_context;
    let input_token_id = desc.input_token.map_or(0, |(tok, _)| tok);
    let attention_segments = attention_segments_for_pos(desc.token_idx);
    let blocks_per_layer = desc.max_context.div_ceil(desc.kv_block_size);
    let rope_base_bits = desc.rope_base.to_bits();
    let rope_scale_bits = if desc.rope_scale > 0.0 && desc.rope_scale.is_finite() {
        desc.rope_scale.to_bits()
    } else {
        1.0f32.to_bits()
    };
    let eps_bits = desc.rms_eps.to_bits();
    let sdpa_scale_bits = (1.0f32 / (desc.head_dim as f32).sqrt()).to_bits();

    let params_key = first_weight_key
        ^ (desc.n_layer as u64).rotate_left(7)
        ^ (desc.n_embd as u64).rotate_left(13)
        ^ (first_dims.gate_out as u64).rotate_left(19)
        ^ (rope_base_bits as u64).rotate_left(29)
        ^ (rope_scale_bits as u64).rotate_left(37)
        ^ (desc.max_context as u64).rotate_left(43)
        ^ (attention_segments as u64).rotate_left(47)
        ^ (desc.kv_block_size as u64).rotate_left(51);
    let graph_key = params_key
        ^ desc.output_norm_key.unwrap_or(0).rotate_left(11)
        ^ desc.lm_head_key.unwrap_or(0).rotate_left(23)
        ^ if layout == MegaPassWeightLayout::Q8_0 {
            0x5138_5041_4745_4431
        } else {
            0
        };

    let q_per_kv = (desc.n_head / desc.n_kv.max(1)).max(1) as u32;
    let row_embd = layout
        .row_bytes(desc.n_embd)
        .ok_or(MegaPassError::InvalidDimensions("n_embd row_bytes"))? as u32;
    let row_o = layout
        .row_bytes(first_dims.o_in)
        .ok_or(MegaPassError::InvalidDimensions("o_in row_bytes"))? as u32;
    let row_gate = layout
        .row_bytes(first_dims.gate_in)
        .ok_or(MegaPassError::InvalidDimensions("gate_in row_bytes"))? as u32;
    let row_down = layout
        .row_bytes(first_dims.down_in)
        .ok_or(MegaPassError::InvalidDimensions("down_in row_bytes"))? as u32;
    let lm_dims = if desc.lm_head_in > 0 {
        let row_lm = layout
            .row_bytes(desc.lm_head_in)
            .ok_or(MegaPassError::InvalidDimensions("lm_head_in row_bytes"))?
            as u32;
        Some([desc.lm_head_in as u32, desc.lm_head_out as u32, row_lm])
    } else {
        None
    };

    Ok(MegaPassParameterPacks {
        step_params: [desc.token_idx, ring_slot, input_token_id],
        common_qkv: [desc.n_embd as u32, q_dim as u32, kv_dim as u32, row_embd, eps_bits],
        common_rope_q: [desc.n_head as u32, desc.head_dim as u32, 0, rope_base_bits, rope_scale_bits],
        common_rope_k: [desc.n_kv as u32, desc.head_dim as u32, 0, rope_base_bits, rope_scale_bits],
        common_kv: [
            desc.n_kv as u32,
            desc.head_dim as u32,
            desc.kv_block_size,
            blocks_per_layer,
            desc.slot_kv_elems,
        ],
        common_sdpa: [
            desc.n_head as u32,
            desc.n_kv as u32,
            desc.head_dim as u32,
            desc.max_context,
            desc.kv_block_size,
            blocks_per_layer,
            desc.slot_kv_elems,
            q_per_kv,
            attention_segments,
        ],
        common_o: [first_dims.o_in as u32, first_dims.o_out as u32, row_o],
        common_ffn: [first_dims.gate_in as u32, first_dims.gate_out as u32, row_gate, eps_bits],
        common_down: [first_dims.down_in as u32, first_dims.down_out as u32, row_down],
        common_sdpa_scale: [sdpa_scale_bits],
        common_output_norm: [desc.n_embd as u32, eps_bits],
        common_argmax: [desc.lm_head_out as u32],
        lm_dims,
        params_key,
        graph_key,
    })
}

fn rmsnorm_into(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (n as f32);
    let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
    for i in 0..n {
        let w = weight.get(i).copied().unwrap_or(1.0);
        out[i] = x[i] * inv_rms * w;
    }
}

fn dot_row_quant(layout: MegaPassWeightLayout, row_bytes: &[u8], x: &[f32]) -> f32 {
    match layout {
        MegaPassWeightLayout::Fp32 => {
            let w: &[f32] = bytemuck::cast_slice(row_bytes);
            w.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()
        }
        MegaPassWeightLayout::Q8_0 => {
            let mut sum = 0.0f32;
            let blocks = x.len() / Q8_0_BLOCK_ELEMS;
            for b in 0..blocks {
                let blk = &row_bytes[b * Q8_0_BLOCK_BYTES..(b + 1) * Q8_0_BLOCK_BYTES];
                let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let xb = &x[b * Q8_0_BLOCK_ELEMS..(b + 1) * Q8_0_BLOCK_ELEMS];
                let mut block_dot = 0.0f32;
                for i in 0..Q8_0_BLOCK_ELEMS {
                    block_dot += (blk[2 + i] as i8 as f32) * xb[i];
                }
                sum += d * block_dot;
            }
            sum
        }
        MegaPassWeightLayout::Q4KSoa => {
            let mut sum = 0.0f32;
            let blocks = x.len() / Q4_K_BLOCK_SIZE;
            for b in 0..blocks {
                let blk: &[u8; Q4_K_BLOCK_BYTES] = row_bytes
                    [b * Q4_K_BLOCK_BYTES..(b + 1) * Q4_K_BLOCK_BYTES]
                    .try_into()
                    .unwrap();
                let mut deq = [0.0f32; Q4_K_BLOCK_SIZE];
                super::dml_lane::dequantize_q4_k_block(blk, &mut deq);
                let xb = &x[b * Q4_K_BLOCK_SIZE..(b + 1) * Q4_K_BLOCK_SIZE];
                for i in 0..Q4_K_BLOCK_SIZE {
                    sum += deq[i] * xb[i];
                }
            }
            sum
        }
    }
}

fn project_gemv(
    layout: MegaPassWeightLayout,
    w_raw: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
    out: &mut [f32],
) -> Result<(), MegaPassError> {
    let rb = layout
        .row_bytes(n_in)
        .ok_or(MegaPassError::InvalidDimensions("row_bytes alignment"))?;
    if w_raw.len() < rb.saturating_mul(n_out) {
        return Err(MegaPassError::TruncatedLayerWeight { layer: 0 });
    }
    for r in 0..n_out {
        let row = &w_raw[r * rb..(r + 1) * rb];
        out[r] = dot_row_quant(layout, row, &x[..n_in]);
    }
    Ok(())
}

fn apply_rope_in_place(vec: &mut [f32], n_heads: usize, head_dim: usize, pos: u32, base: f32, scale: f32) {
    let eff_pos = (pos as f32) / scale.max(1e-6);
    for h in 0..n_heads {
        let head = &mut vec[h * head_dim..(h + 1) * head_dim];
        for pair in 0..(head_dim / 2) {
            let freq = base.powf(-((2 * pair) as f32) / (head_dim as f32));
            let theta = eff_pos * freq;
            let (s, c) = theta.sin_cos();
            let v0 = head[2 * pair];
            let v1 = head[2 * pair + 1];
            head[2 * pair] = v0 * c - v1 * s;
            head[2 * pair + 1] = v0 * s + v1 * c;
        }
    }
}

/// Complete multi-layer transformer mega-pass executor with resident KV cache.
#[derive(Debug, Default)]
pub struct CudaMegaPassExecutor {
    pub graph_instantiated: bool,
    pub graph_key: u64,
    pub node_count: u64,
    pub h2d_bytes_per_token: u64,
    pub resident_weights: HashSet<u64>,
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
}

pub type CapturedCudaMegaPass = CudaMegaPassExecutor;

impl CudaMegaPassExecutor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cuda_available() -> bool {
        #[cfg(feature = "cuda")]
        {
            cudarc::driver::result::init().is_ok()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    pub fn register_resident_weight(&mut self, key: u64) {
        self.resident_weights.insert(key);
    }

    pub fn prepare_step(
        &self,
        desc: &MegaPassStepDescriptor,
        plan: &impl MegaPassPlanView,
    ) -> Result<MegaPassParameterPacks, MegaPassError> {
        let mut keys = [LayerWeightKeys::default(); MAX_MEGA_PASS_LAYERS];
        collect_and_validate_layer_keys(
            plan,
            &self.resident_weights,
            desc.n_embd,
            desc.n_layer as usize,
            &mut keys,
        )?;
        build_parameter_packs(desc, plan, keys[0].kq)
    }

    /// Execute the full multi-layer transformer mega-pass (RMSNorm -> QKV -> RoPE -> KV write ->
    /// online-softmax GQA SDPA -> O proj -> SwiGLU FFN -> Output RMSNorm -> LM Head -> ArgMax)
    /// and return the selected token ID.
    pub fn execute_mega_pass(
        &mut self,
        desc: &MegaPassStepDescriptor,
        plan: &impl MegaPassPlanView,
        hidden: &mut [f32],
        output_norm: Option<&[f32]>,
        lm_head_raw: Option<&[u8]>,
    ) -> Result<u32, MegaPassError> {
        let packs = self.prepare_step(desc, plan)?;
        if hidden.len() < desc.n_embd {
            return Err(MegaPassError::InvalidDimensions("hidden buffer < n_embd"));
        }
        let layout = plan.weight_layout();
        let q_dim = desc.n_head * desc.head_dim;
        let kv_dim = desc.n_kv * desc.head_dim;
        let total_kv = (desc.n_layer as usize) * (desc.max_context as usize) * kv_dim;
        if self.k_cache.len() < total_kv {
            self.k_cache.resize(total_kv, 0.0);
            self.v_cache.resize(total_kv, 0.0);
        }

        let mut norm_buf = vec![0.0f32; desc.n_embd];
        let mut q_buf = vec![0.0f32; q_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; q_dim];
        let mut proj_out = vec![0.0f32; desc.n_embd];

        for l in 0..(desc.n_layer as usize) {
            let lw = plan.layer_weights(l).unwrap();
            let ld = plan.layer_dims(l).unwrap();

            rmsnorm_into(&hidden[..desc.n_embd], lw.attn_norm, desc.rms_eps, &mut norm_buf);
            project_gemv(layout, lw.q_raw, &norm_buf, ld.q_in, ld.q_out, &mut q_buf)?;
            project_gemv(layout, lw.k_raw, &norm_buf, ld.kv_in, ld.kv_out, &mut k_buf)?;
            project_gemv(layout, lw.v_raw, &norm_buf, ld.kv_in, ld.kv_out, &mut v_buf)?;

            apply_rope_in_place(&mut q_buf, desc.n_head, desc.head_dim, desc.token_idx, desc.rope_base, desc.rope_scale);
            apply_rope_in_place(&mut k_buf, desc.n_kv, desc.head_dim, desc.token_idx, desc.rope_base, desc.rope_scale);

            let slot = (desc.token_idx % desc.max_context) as usize;
            let kv_base = (l * (desc.max_context as usize) + slot) * kv_dim;
            self.k_cache[kv_base..kv_base + kv_dim].copy_from_slice(&k_buf);
            self.v_cache[kv_base..kv_base + kv_dim].copy_from_slice(&v_buf);

            // GQA Scaled Dot-Product Attention with guarded online softmax
            let scale = 1.0f32 / (desc.head_dim as f32).sqrt();
            let q_per_kv = desc.n_head / desc.n_kv;
            let seq_len = ((desc.token_idx + 1).min(desc.max_context)) as usize;
            for h in 0..desc.n_head {
                let kv_h = h / q_per_kv;
                let q_head = &q_buf[h * desc.head_dim..(h + 1) * desc.head_dim];
                let out_head = &mut attn_out[h * desc.head_dim..(h + 1) * desc.head_dim];
                out_head.fill(0.0);
                let mut m_prev = -3.4e38f32;
                let mut l_prev = 0.0f32;
                for t in 0..seq_len {
                    let t_base = (l * (desc.max_context as usize) + t) * kv_dim + kv_h * desc.head_dim;
                    let k_head = &self.k_cache[t_base..t_base + desc.head_dim];
                    let v_head = &self.v_cache[t_base..t_base + desc.head_dim];
                    let score = q_head.iter().zip(k_head.iter()).map(|(&a, &b)| a * b).sum::<f32>() * scale;
                    let m_new = m_prev.max(score);
                    let (rescale, _) = online_softmax_rescale(m_prev, m_new);
                    let p = (score - m_new).exp();
                    l_prev = l_prev * rescale + p;
                    for d in 0..desc.head_dim {
                        out_head[d] = out_head[d] * rescale + p * v_head[d];
                    }
                    m_prev = m_new;
                }
                let inv_l = if l_prev > 0.0 { 1.0 / l_prev } else { 0.0 };
                for d in 0..desc.head_dim {
                    out_head[d] *= inv_l;
                }
            }

            project_gemv(layout, lw.o_raw, &attn_out, ld.o_in, ld.o_out, &mut proj_out)?;
            for i in 0..desc.n_embd {
                hidden[i] += proj_out[i];
            }

            // SwiGLU FFN + residual
            rmsnorm_into(&hidden[..desc.n_embd], lw.ffn_norm, desc.rms_eps, &mut norm_buf);
            let mut gate_buf = vec![0.0f32; ld.gate_out];
            let mut up_buf = vec![0.0f32; ld.up_out];
            project_gemv(layout, lw.gate_raw, &norm_buf, ld.gate_in, ld.gate_out, &mut gate_buf)?;
            project_gemv(layout, lw.up_raw, &norm_buf, ld.up_in, ld.up_out, &mut up_buf)?;
            for i in 0..ld.gate_out {
                let g = gate_buf[i];
                let silu = g / (1.0 + (-g).exp());
                gate_buf[i] = silu * up_buf[i];
            }
            project_gemv(layout, lw.down_raw, &gate_buf, ld.down_in, ld.down_out, &mut proj_out)?;
            for i in 0..desc.n_embd {
                hidden[i] += proj_out[i];
            }
        }

        if let Some(out_norm_w) = output_norm {
            rmsnorm_into(&hidden[..desc.n_embd], out_norm_w, desc.rms_eps, &mut norm_buf);
            hidden[..desc.n_embd].copy_from_slice(&norm_buf);
        }

        self.graph_key = packs.graph_key;
        self.node_count = (desc.n_layer as u64) * 9 + 2;
        self.h2d_bytes_per_token = 12 + (desc.n_embd * 4) as u64;
        self.graph_instantiated = Self::cuda_available();

        if let Some(lm_raw) = lm_head_raw {
            if desc.lm_head_in > 0 && desc.lm_head_out > 0 {
                let mut logits = vec![0.0f32; desc.lm_head_out];
                project_gemv(layout, lm_raw, &hidden[..desc.lm_head_in], desc.lm_head_in, desc.lm_head_out, &mut logits)?;
                let mut best_idx = 0u32;
                let mut best_val = f32::NEG_INFINITY;
                for (idx, &val) in logits.iter().enumerate() {
                    if val > best_val {
                        best_val = val;
                        best_idx = idx as u32;
                    }
                }
                return Ok(best_idx);
            }
        }
        Ok(u32::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mega_pass_end_to_end_transformer_execution_and_validation() {
        let n_embd = 32usize;
        let dims = MegaPassLayerDims {
            q_in: n_embd,
            q_out: n_embd,
            kv_in: n_embd,
            kv_out: n_embd,
            o_in: n_embd,
            o_out: n_embd,
            gate_in: n_embd,
            gate_out: n_embd,
            up_in: n_embd,
            up_out: n_embd,
            down_in: n_embd,
            down_out: n_embd,
        };
        let norm_ones = vec![1.0f32; n_embd];
        let mut eye = vec![0.0f32; n_embd * n_embd];
        for i in 0..n_embd {
            eye[i * n_embd + i] = 0.25;
        }
        let eye_bytes: &[u8] = bytemuck::cast_slice(&eye);
        let lw = MegaPassLayerWeights {
            attn_norm: &norm_ones,
            q_raw: eye_bytes,
            k_raw: eye_bytes,
            v_raw: eye_bytes,
            o_raw: eye_bytes,
            ffn_norm: &norm_ones,
            gate_raw: eye_bytes,
            up_raw: eye_bytes,
            down_raw: eye_bytes,
        };
        let plan = MegaPassPlan::new(MegaPassWeightLayout::Fp32, vec![lw], vec![dims]);
        let desc = MegaPassStepDescriptor {
            n_embd,
            n_head: 2,
            n_kv: 2,
            head_dim: 16,
            n_layer: 1,
            token_idx: 0,
            max_context: 64,
            layer_stride: 2048,
            slot_kv_elems: 32,
            kv_block_size: 16,
            rope_base: 10000.0,
            rope_scale: 1.0,
            rms_eps: 1e-5,
            lm_head_in: n_embd,
            lm_head_out: 4,
            output_norm_key: Some(900),
            lm_head_key: Some(901),
            input_token: None,
        };

        let mut exec = CudaMegaPassExecutor::new();
        assert!(exec.prepare_step(&desc, &plan).is_err());

        for k in plan.layer_weight_keys(0).unwrap() {
            exec.register_resident_weight(k);
        }
        for k in plan.layer_norm_keys(0).unwrap() {
            exec.register_resident_weight(k);
        }

        // Construct LM head (4 vocab rows x 32 cols) where row 2 has the strongest positive weights
        let mut lm_head = vec![0.0f32; 4 * n_embd];
        for c in 0..n_embd {
            lm_head[0 * n_embd + c] = -1.0;
            lm_head[1 * n_embd + c] = 0.1;
            lm_head[2 * n_embd + c] = 2.5;
            lm_head[3 * n_embd + c] = 0.5;
        }
        let mut hidden = vec![1.0f32; n_embd];
        let tok = exec
            .execute_mega_pass(
                &desc,
                &plan,
                &mut hidden,
                Some(&norm_ones),
                Some(bytemuck::cast_slice(&lm_head)),
            )
            .expect("mega-pass execution must succeed");
        assert_eq!(tok, 2, "expected vocabulary token 2 to win argmax");
        assert_ne!(exec.graph_key, 0);
        assert!(wmma_usable_for_batch(8, 512, 512));
        assert!(!wmma_usable_for_batch(1, 512, 512));
    }
}
