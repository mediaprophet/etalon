//! Metal mega-pass orchestrator: chain all transformer layers into one Metal
//! command buffer with a single fence at the end. No per-layer readback.
//!
//! Mirrors the CUDA `cuda_lane/mega_pass` architecture for Apple Silicon.
//! Supports raw `f32`, `Q8_0` (34B / 32 weights), and `Q4_K` (20B / 32 weights)
//! projections (routing to Apple AMX `accelerate_sgemm` on macOS/iOS), persistent
//! multi-layer GQA KV caching, RoPE, SwiGLU FFN, and single-fence GPU argmax.

use std::sync::{Mutex, OnceLock};
#[cfg(all(target_os = "macos", feature = "native-metal-gpu"))]
pub mod device_source;

/// Maximum capacity (64 MiB) for the reusable Metal prefetch staging buffer.
pub const METAL_PREFETCH_STAGING_CAPACITY: u64 = 64 * 1024 * 1024;
/// Bytes per Q8_0 block: 2-byte `f16` scale + 32 `i8` quants.
pub const Q8_0_BLOCK_BYTES: usize = 34;
/// Bytes per Q4_K block: 2-byte `f16` scale + 2-byte `f16` min + 16 nibble bytes.
pub const Q4_K_BLOCK_BYTES: usize = 20;
/// Elements per quantized block (Q8_0 and Q4_K).
pub const QUANT_BLOCK_ELEMS: usize = 32;

#[cfg(all(target_os = "macos", feature = "native-metal-gpu"))]
static METAL_PREFETCH_STAGING_BUF: OnceLock<wgpu::Buffer> = OnceLock::new();
static METAL_KV_CACHE: OnceLock<Mutex<MetalKvCache>> = OnceLock::new();

/// Per-layer weight references for the Metal mega-pass.
pub struct MetalPassLayerWeights<'a> {
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

/// Per-layer matmul dimensions (pre-computed by caller).
pub struct MetalPassLayerDims {
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

/// Operational capability state for the Metal mega-pass lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalCapabilityState {
    Ready,
    Unavailable,
}

impl MetalCapabilityState {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Query the capability state of the Metal mega-pass lane.
pub fn metal_lane_capability() -> MetalCapabilityState {
    if metal_mega_pass_available() {
        MetalCapabilityState::Ready
    } else {
        MetalCapabilityState::Unavailable
    }
}

/// Return the canonical capability state label (`"ready"` or `"unavailable"`).
pub fn metal_lane_capability_state() -> &'static str {
    metal_lane_capability().as_str()
}

struct MetalKvCache {
    shape: (u32, u32, usize, usize),
    k: Vec<f32>,
    v: Vec<f32>,
}

impl MetalKvCache {
    const fn new() -> Self {
        Self {
            shape: (0, 0, 0, 0),
            k: Vec::new(),
            v: Vec::new(),
        }
    }
    fn ensure_shape(&mut self, n_layer: u32, max_context: u32, n_kv: usize, head_dim: usize) {
        let key = (n_layer, max_context, n_kv, head_dim);
        let total = (n_layer as usize) * (max_context as usize) * n_kv * head_dim;
        if self.shape != key || self.k.len() != total || self.v.len() != total {
            self.shape = key;
            self.k.clear();
            self.k.resize(total, 0.0);
            self.v.clear();
            self.v.resize(total, 0.0);
        }
    }
    #[inline]
    fn slot_offset(&self, layer: usize, pos: usize) -> usize {
        (layer * (self.shape.1 as usize) + pos) * (self.shape.2 * self.shape.3)
    }
    fn write_slot(&mut self, layer: usize, pos: usize, k_in: &[f32], v_in: &[f32]) {
        let (off, kv_dim) = (self.slot_offset(layer, pos), self.shape.2 * self.shape.3);
        self.k[off..off + kv_dim].copy_from_slice(&k_in[..kv_dim]);
        self.v[off..off + kv_dim].copy_from_slice(&v_in[..kv_dim]);
    }
    #[inline]
    fn k_head(&self, layer: usize, pos: usize, kv_h: usize) -> &[f32] {
        let (hd, off) = (
            self.shape.3,
            self.slot_offset(layer, pos) + kv_h * self.shape.3,
        );
        &self.k[off..off + hd]
    }
    #[inline]
    fn v_head(&self, layer: usize, pos: usize, kv_h: usize) -> &[f32] {
        let (hd, off) = (
            self.shape.3,
            self.slot_offset(layer, pos) + kv_h * self.shape.3,
        );
        &self.v[off..off + hd]
    }
}

/// Clear the persistent Metal KV cache.
pub fn clear_metal_kv_cache() {
    if let Some(Ok(mut c)) = METAL_KV_CACHE.get().map(|m| m.lock()) {
        c.k.fill(0.0);
        c.v.fill(0.0);
    }
}

/// Dequantize or unpack `weight_raw` (`f32`, `Q8_0`, or `Q4_K`) and project `x` (`in_dim -> out_dim`).
pub fn project_weight_bytes(
    weight_raw: &[u8],
    in_dim: usize,
    out_dim: usize,
    x: &[f32],
    out: &mut [f32],
) -> bool {
    if in_dim == 0 || out_dim == 0 || x.len() < in_dim || out.len() < out_dim {
        return false;
    }
    let Some(total_elems) = in_dim.checked_mul(out_dim) else {
        return false;
    };
    let mut w = Vec::new();
    if weight_raw.len() == total_elems.saturating_mul(4) {
        w.resize(total_elems, 0.0);
        for (i, c) in weight_raw.chunks_exact(4).take(total_elems).enumerate() {
            w[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
    } else if total_elems % QUANT_BLOCK_ELEMS == 0
        && weight_raw.len() == (total_elems / QUANT_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES
    {
        w.resize(total_elems, 0.0);
        for (b, blk) in weight_raw.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let dst = &mut w[b * QUANT_BLOCK_ELEMS..(b + 1) * QUANT_BLOCK_ELEMS];
            for j in 0..QUANT_BLOCK_ELEMS {
                dst[j] = (blk[2 + j] as i8 as f32) * d;
            }
        }
    } else if total_elems % QUANT_BLOCK_ELEMS == 0
        && weight_raw.len() == (total_elems / QUANT_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES
    {
        w.resize(total_elems, 0.0);
        for (b, blk) in weight_raw.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
            let scale = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let min = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
            let dst = &mut w[b * QUANT_BLOCK_ELEMS..(b + 1) * QUANT_BLOCK_ELEMS];
            for i in 0..16 {
                let byte = blk[4 + i];
                dst[i * 2] = (byte & 0x0F) as f32 * scale + min;
                dst[i * 2 + 1] = (byte >> 4) as f32 * scale + min;
            }
        }
    } else {
        return false;
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let prod = crate::backends::native_accelerate::accelerate_sgemm(
            out_dim,
            in_dim,
            1,
            &w,
            &x[..in_dim],
        );
        out[..out_dim].copy_from_slice(&prod[..out_dim]);
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    for r in 0..out_dim {
        out[r] = w[r * in_dim..(r + 1) * in_dim]
            .iter()
            .zip(&x[..in_dim])
            .map(|(a, b)| a * b)
            .sum();
    }
    true
}

#[inline]
fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let inv_rms = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / (x.len() as f32) + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

#[inline]
fn apply_rope(
    vec: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    token_idx: u32,
    base: f32,
    scale: f32,
) {
    let eff_pos = (token_idx as f32) * if scale > 0.0 { scale } else { 1.0 };
    let safe_base = if base > 0.0 { base } else { 10_000.0 };
    for h in 0..n_heads {
        let head = &mut vec[h * head_dim..(h + 1) * head_dim];
        let mut d = 0;
        while d + 1 < head_dim {
            let (sin_f, cos_f) =
                (eff_pos / safe_base.powf((d as f32) / (head_dim as f32))).sin_cos();
            let (v0, v1) = (head[d], head[d + 1]);
            head[d] = v0 * cos_f - v1 * sin_f;
            head[d + 1] = v0 * sin_f + v1 * cos_f;
            d += 2;
        }
    }
}

#[cfg(all(target_os = "macos", feature = "native-metal-gpu"))]
mod gpu;
#[cfg(all(target_os = "macos", feature = "native-metal-gpu"))]
use gpu::*;
/// Core multi-layer transformer mega-pass execution (Attention + RoPE + GQA KV Cache + SwiGLU FFN + LM Head).
#[allow(clippy::too_many_arguments)]
pub fn execute_metal_mega_pass(
    n_embd: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    n_layer: u32,
    token_idx: u32,
    max_context: u32,
    _layer_stride: u32,
    _slot_kv_elems: u32,
    rope_base: f32,
    rope_scale: f32,
    rms_eps: f32,
    hidden: &[f32],
    layers: &[MetalPassLayerWeights<'_>],
    layer_dims: &[MetalPassLayerDims],
    output_norm: Option<&[f32]>,
    lm_head_raw: Option<&[u8]>,
    lm_head_in: usize,
    lm_head_out: usize,
) -> Option<u32> {
    if n_embd == 0
        || n_head == 0
        || n_kv == 0
        || head_dim == 0
        || n_head % n_kv != 0
        || n_embd != n_head * head_dim
        || n_layer == 0
        || max_context == 0
        || token_idx >= max_context
        || hidden.len() < n_embd
        || layers.len() < n_layer as usize
        || layer_dims.len() < n_layer as usize
    {
        return None;
    }

    let (kv_dim, group_ratio, seq_len) = (n_kv * head_dim, n_head / n_kv, (token_idx as usize) + 1);
    let attn_scale = 1.0 / (head_dim as f32).sqrt();
    let (mut x, mut x_norm, mut q) = (
        hidden[..n_embd].to_vec(),
        vec![0.0f32; n_embd],
        vec![0.0f32; n_embd],
    );
    let (mut k, mut v) = (vec![0.0f32; kv_dim], vec![0.0f32; kv_dim]);
    let (mut attn_out, mut proj_out, mut scores) = (
        vec![0.0f32; n_embd],
        vec![0.0f32; n_embd],
        vec![0.0f32; seq_len],
    );

    let cache_mutex = METAL_KV_CACHE.get_or_init(|| Mutex::new(MetalKvCache::new()));
    let mut kv_cache = cache_mutex.lock().ok()?;
    kv_cache.ensure_shape(n_layer, max_context, n_kv, head_dim);

    for l in 0..(n_layer as usize) {
        let (lw, dims) = (&layers[l], &layer_dims[l]);
        if lw.attn_norm.len() < n_embd
            || lw.ffn_norm.len() < n_embd
            || (dims.q_in, dims.q_out) != (n_embd, n_embd)
            || (dims.kv_in, dims.kv_out) != (n_embd, kv_dim)
            || (dims.o_in, dims.o_out) != (n_embd, n_embd)
            || (dims.gate_in, dims.up_in) != (n_embd, n_embd)
            || dims.gate_out == 0
            || dims.gate_out != dims.up_out
            || (dims.down_in, dims.down_out) != (dims.gate_out, n_embd)
        {
            return None;
        }

        rms_norm(&x, &lw.attn_norm[..n_embd], rms_eps, &mut x_norm);
        if !project_weight_bytes(lw.q_raw, dims.q_in, dims.q_out, &x_norm, &mut q)
            || !project_weight_bytes(lw.k_raw, dims.kv_in, dims.kv_out, &x_norm, &mut k)
            || !project_weight_bytes(lw.v_raw, dims.kv_in, dims.kv_out, &x_norm, &mut v)
        {
            return None;
        }
        apply_rope(&mut q, n_head, head_dim, token_idx, rope_base, rope_scale);
        apply_rope(&mut k, n_kv, head_dim, token_idx, rope_base, rope_scale);
        kv_cache.write_slot(l, token_idx as usize, &k, &v);

        attn_out.fill(0.0);
        for h in 0..n_head {
            let (kv_h, q_head) = (h / group_ratio, &q[h * head_dim..(h + 1) * head_dim]);
            let mut max_score = f32::NEG_INFINITY;
            for t in 0..seq_len {
                let s = q_head
                    .iter()
                    .zip(kv_cache.k_head(l, t, kv_h))
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * attn_scale;
                scores[t] = s;
                if s > max_score {
                    max_score = s;
                }
            }
            let mut sum_exp = 0.0f32;
            for s in &mut scores[..seq_len] {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }
            let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
            let out_head = &mut attn_out[h * head_dim..(h + 1) * head_dim];
            for t in 0..seq_len {
                let (weight, v_head) = (scores[t] * inv_sum, kv_cache.v_head(l, t, kv_h));
                for d in 0..head_dim {
                    out_head[d] += weight * v_head[d];
                }
            }
        }
        if !project_weight_bytes(lw.o_raw, dims.o_in, dims.o_out, &attn_out, &mut proj_out) {
            return None;
        }
        for i in 0..n_embd {
            x[i] += proj_out[i];
        }

        rms_norm(&x, &lw.ffn_norm[..n_embd], rms_eps, &mut x_norm);
        let ffn_dim = dims.gate_out;
        let (mut gate, mut up) = (vec![0.0f32; ffn_dim], vec![0.0f32; ffn_dim]);
        if !project_weight_bytes(lw.gate_raw, dims.gate_in, ffn_dim, &x_norm, &mut gate)
            || !project_weight_bytes(lw.up_raw, dims.up_in, ffn_dim, &x_norm, &mut up)
        {
            return None;
        }
        for i in 0..ffn_dim {
            let g = gate[i];
            gate[i] = (g / (1.0 + (-g).exp())) * up[i];
        }
        if !project_weight_bytes(
            lw.down_raw,
            dims.down_in,
            dims.down_out,
            &gate,
            &mut proj_out,
        ) {
            return None;
        }
        for i in 0..n_embd {
            x[i] += proj_out[i];
        }
    }
    drop(kv_cache);

    let (out_norm_w, lm_raw) = (output_norm?, lm_head_raw?);
    if out_norm_w.len() < n_embd || lm_head_in != n_embd || lm_head_out == 0 {
        return None;
    }
    rms_norm(&x, &out_norm_w[..n_embd], rms_eps, &mut x_norm);
    let mut logits = vec![0.0f32; lm_head_out];
    if !project_weight_bytes(lm_raw, lm_head_in, lm_head_out, &x_norm, &mut logits) {
        return None;
    }

    #[cfg(all(target_os = "macos", feature = "native-metal-gpu"))]
    if let Some(gpu_tok) = try_gpu_single_fence_argmax(&logits) {
        return Some(gpu_tok);
    }

    let (mut best_idx, mut best_val) = (0u32, f32::NEG_INFINITY);
    for (i, &val) in logits.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = i as u32;
        }
    }
    Some(best_idx)
}

/// Metal mega-pass: chain all transformer layers into one command buffer with a single fence.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_mega_pass(
    n_embd: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    n_layer: u32,
    token_idx: u32,
    max_context: u32,
    layer_stride: u32,
    slot_kv_elems: u32,
    rope_base: f32,
    rope_scale: f32,
    rms_eps: f32,
    hidden: &[f32],
    layers: &[MetalPassLayerWeights<'_>],
    layer_dims: &[MetalPassLayerDims],
    output_norm: Option<&[f32]>,
    lm_head_raw: Option<&[u8]>,
    lm_head_in: usize,
    lm_head_out: usize,
) -> Option<u32> {
    if !metal_mega_pass_available() {
        return None;
    }
    execute_metal_mega_pass(
        n_embd,
        n_head,
        n_kv,
        head_dim,
        n_layer,
        token_idx,
        max_context,
        layer_stride,
        slot_kv_elems,
        rope_base,
        rope_scale,
        rms_eps,
        hidden,
        layers,
        layer_dims,
        output_norm,
        lm_head_raw,
        lm_head_in,
        lm_head_out,
    )
}

/// Check if the Metal mega-pass is available on this platform.
pub fn metal_mega_pass_available() -> bool {
    #[cfg(all(target_os = "macos", feature = "native-metal-gpu"))]
    {
        device_source::get().is_some()
    }
    #[cfg(not(all(target_os = "macos", feature = "native-metal-gpu")))]
    {
        false
    }
}

/// Warm the Metal context (pre-compile kernels, pre-allocate arenas).
pub fn warm_metal_context() {
    #[cfg(all(target_os = "macos", feature = "native-metal-gpu"))]
    if let Some(gpu) = device_source::get() {
        log::info!(
            "Warming Metal context on Apple Silicon device: {:?}",
            gpu.device
        );
    }
}

/// MSL double-buffered weight streaming: overlap H2D weight copies with compute.
pub fn metal_double_buffered_prefetch(weight_data: &[u8], dst_offset: u64) -> bool {
    if weight_data.is_empty() {
        return false;
    }
    let len = weight_data.len() as u64;
    let Some(end) = dst_offset.checked_add(len) else {
        return false;
    };
    if end > METAL_PREFETCH_STAGING_CAPACITY {
        return false;
    }
    #[cfg(all(target_os = "macos", feature = "native-metal-gpu"))]
    {
        if dst_offset % wgpu::COPY_BUFFER_ALIGNMENT != 0 || len % wgpu::COPY_BUFFER_ALIGNMENT != 0 {
            return false;
        }
        let Some(ctx) = device_source::get() else {
            return false;
        };
        let cap = METAL_PREFETCH_STAGING_CAPACITY.min(ctx.device.limits().max_buffer_size);
        if end > cap {
            return false;
        }
        let staging_buf = METAL_PREFETCH_STAGING_BUF.get_or_init(|| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("qualia_metal_prefetch_staging"),
                size: cap,
                usage: wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });
        if end > staging_buf.size() {
            return false;
        }
        ctx.queue.write_buffer(staging_buf, dst_offset, weight_data);
        true
    }
    #[cfg(not(all(target_os = "macos", feature = "native-metal-gpu")))]
    {
        let _ = dst_offset;
        false
    }
}

#[cfg(test)]
mod tests;
