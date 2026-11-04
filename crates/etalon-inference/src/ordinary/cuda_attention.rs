use super::rope::rope_inplace;
use etalon_model::inference::{ModelHyperparams, TensorInfo as GgufTensorInfo};
const MAX_CONTEXT_WINDOW: u32 = 1024;
pub trait CudaAttentionHost {
    fn with_cpu_kv<R>(&self, run: impl FnOnce(Option<&mut [f32]>) -> R) -> R;
    fn device_attempt(
        &self,
        hit: bool,
        layer: u32,
        token: u32,
        o_in: usize,
        o_out: usize,
        q_dim: usize,
        n_embd: usize,
    );
    fn gemm(
        &self,
        raw: &[u8],
        info: &GgufTensorInfo,
        input: &[f32],
        out: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> bool;
}
#[derive(Clone, Copy)]
pub struct CudaAttentionLayout {
    pub max_context: u32,
    pub n_layer: u32,
    pub n_kv_head: u32,
    pub head_dim: u32,
    pub slot_kv_elems: u32,
    pub layer_stride: u32,
    pub total_f32_elems: usize,
    pub int8: bool,
    pub dict_k: u32,
}
impl CudaAttentionLayout {
    fn ring_slot(&self, token: u32) -> u32 {
        token % self.max_context
    }
    fn k_index(&self, layer: u32, slot: u32, head: u32, dim: u32) -> usize {
        layer as usize * self.layer_stride as usize
            + slot as usize * self.slot_kv_elems as usize * 2
            + head as usize * self.head_dim as usize
            + dim as usize
    }
    fn v_index(&self, layer: u32, slot: u32, head: u32, dim: u32) -> usize {
        self.k_index(layer, slot, head, dim) + self.n_kv_head as usize * self.head_dim as usize
    }
}
pub fn try_cuda_soa_attention_layer<P: CudaAttentionHost>(
    host: &P,
    hidden_input: &[f32],
    n_embd: usize,
    q_dim: usize,
    kv_dim: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    layout: &CudaAttentionLayout,
    layer: u32,
    token_idx: u32,
    h: &ModelHyperparams,
    q_raw: &[u8],
    k_raw: &[u8],
    v_raw: &[u8],
    out_info: Option<&GgufTensorInfo>,
    mmap: &[u8],
    tensor_data_start: u64,
    scratch_a: &mut [f32],
    scratch_b: &mut [f32],
) -> bool {
    if head_dim == 0 || n_head == 0 || n_kv == 0 {
        return false;
    }
    if q_dim + kv_dim * 2 > scratch_a.len() + scratch_b.len() {
        // Need space: use scratch_a for Q and attn_out, scratch_b for K then V sequentially.
    }
    // Layout: scratch_b holds K then we overwrite with V after writing K to cache;
    // scratch_a holds Q for SDPA then O result.
    if q_dim > scratch_a.len() || kv_dim > scratch_b.len() {
        return false;
    }

    // P4: device-side RoPE + KV + SDPA + O-proj — no mid-chain QKV readback.
    // Only f32 (non-int8 / non-dict) layouts match the CUDA KV index formula.
    if !layout.int8 && layout.dict_k == 0 && n_embd <= scratch_a.len() {
        if let Some(out_info) = out_info {
            if out_info.ggml_type == etalon_model::ggml::GGML_TYPE_Q4_K_SOA {
                let (o_in, o_out) = super::matmul_dims(out_info);
                if o_in == q_dim && o_out == n_embd {
                    if let Ok(o_raw) =
                        etalon_model::ggml::fetch_tensor_bytes(mmap, tensor_data_start, out_info)
                    {
                        // Ensure device KV once (no-op if already matching).
                        let _ = crate::backends::cuda::ensure_device_kv_cache(
                            layout.max_context,
                            layout.n_layer,
                            layout.n_kv_head,
                            layout.head_dim,
                            layout.slot_kv_elems,
                            layout.layer_stride,
                            layout.total_f32_elems,
                        );
                        // No host dual-write: device KV is authoritative on this path.
                        // Avoids two extra D2H fences per layer (P4 fence budget).
                        if crate::backends::cuda::try_q4k_soa_attention_device(
                            n_embd,
                            n_head,
                            n_kv,
                            head_dim,
                            layer,
                            token_idx,
                            layout.max_context,
                            layout.layer_stride,
                            layout.slot_kv_elems,
                            h.effective_rope_freq_base(),
                            h.effective_rope_scale(),
                            &hidden_input[..n_embd],
                            q_raw,
                            k_raw,
                            v_raw,
                            o_raw,
                            None,
                            &mut scratch_a[..n_embd],
                        ) {
                            host.device_attempt(true, layer, token_idx, o_in, o_out, q_dim, n_embd);
                            return true;
                        } else {
                            host.device_attempt(
                                false, layer, token_idx, o_in, o_out, q_dim, n_embd,
                            );
                        }
                    }
                }
            }
        }
    }

    // Matches the host's admitted projection scratch ceiling. Keep the CUDA
    // readback fallback caller/stack-buffered rather than allocating per token.
    const MAX_PROJECTION: usize = 10_240;
    if q_dim > MAX_PROJECTION || kv_dim > MAX_PROJECTION {
        return false;
    }
    let mut q_storage = [0f32; MAX_PROJECTION];
    let mut k_storage = [0f32; MAX_PROJECTION];
    let mut v_storage = [0f32; MAX_PROJECTION];
    let mut q_proj = &mut q_storage[..q_dim];
    let mut k_proj = &mut k_storage[..kv_dim];
    let mut v_proj = &mut v_storage[..kv_dim];
    if !crate::backends::cuda::try_q4k_soa_qkv(
        n_embd,
        q_dim,
        kv_dim,
        &hidden_input[..n_embd],
        q_raw,
        k_raw,
        v_raw,
        &mut q_proj,
        &mut k_proj,
        &mut v_proj,
    ) {
        return false;
    }

    let base_freq = h.effective_rope_freq_base();
    let rope_scale = h.effective_rope_scale();
    let q_heads_per_kv = h.q_heads_per_kv() as usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let slot = layout.ring_slot(token_idx);
    let pos = token_idx as usize;
    if pos >= MAX_CONTEXT_WINDOW as usize {
        return false;
    }

    // RoPE K + write KV cache
    rope_inplace(
        &mut k_proj,
        n_kv,
        head_dim,
        token_idx,
        base_freq,
        rope_scale,
    );
    host.with_cpu_kv(|kv| {
        let Some(kv) = kv else {
            return false;
        };
        for kvh in 0..n_kv {
            for d in 0..head_dim {
                let idx = layout.k_index(layer, slot, kvh as u32, d as u32);
                if idx >= kv.len() {
                    return false;
                }
                kv[idx] = k_proj[kvh * head_dim + d];
            }
        }
        for kvh in 0..n_kv {
            for d in 0..head_dim {
                let idx = layout.v_index(layer, slot, kvh as u32, d as u32);
                if idx >= kv.len() {
                    return false;
                }
                kv[idx] = v_proj[kvh * head_dim + d];
            }
        }

        // RoPE Q + SDPA → scratch_b[..q_dim]
        rope_inplace(
            &mut q_proj,
            n_head,
            head_dim,
            token_idx,
            base_freq,
            rope_scale,
        );
        let attn_out = &mut scratch_b[..q_dim];
        attn_out.fill(0.0);
        let mut att_scores = [0f32; MAX_CONTEXT_WINDOW as usize];
        for q_h in 0..n_head {
            let kv_h = q_h / q_heads_per_kv.max(1);
            let q_head = &q_proj[q_h * head_dim..(q_h + 1) * head_dim];
            let out_head = &mut attn_out[q_h * head_dim..(q_h + 1) * head_dim];
            let mut max_score = f32::NEG_INFINITY;
            for past_pos in 0..=pos {
                let past_slot = layout.ring_slot(past_pos as u32);
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let k_idx = layout.k_index(layer, past_slot, kv_h as u32, d as u32);
                    if k_idx >= kv.len() {
                        return false;
                    }
                    dot += q_head[d] * kv[k_idx];
                }
                let score = dot * scale;
                att_scores[past_pos] = score;
                if score > max_score {
                    max_score = score;
                }
            }
            let mut sum_exp = 0.0f32;
            for past_pos in 0..=pos {
                let e = (att_scores[past_pos] - max_score).exp();
                att_scores[past_pos] = e;
                sum_exp += e;
            }
            if sum_exp == 0.0 {
                return false;
            }
            for past_pos in 0..=pos {
                let prob = att_scores[past_pos] / sum_exp;
                let past_slot = layout.ring_slot(past_pos as u32);
                for d in 0..head_dim {
                    let v_idx = layout.v_index(layer, past_slot, kv_h as u32, d as u32);
                    if v_idx >= kv.len() {
                        return false;
                    }
                    out_head[d] += kv[v_idx] * prob;
                }
            }
        }

        // O-projection via CUDA GEMV
        let Some(out_info) = out_info else {
            let n = q_dim.min(n_embd);
            scratch_a[..n].copy_from_slice(&attn_out[..n]);
            return true;
        };
        let (o_in, o_out) = super::matmul_dims(out_info);
        if o_in > q_dim || o_out > scratch_a.len() {
            return false;
        }
        let Ok(o_raw) = etalon_model::ggml::fetch_tensor_bytes(mmap, tensor_data_start, out_info)
        else {
            return false;
        };
        if out_info.ggml_type == etalon_model::ggml::GGML_TYPE_Q4_K_SOA {
            if crate::backends::cuda::try_q4k_soa_gemv(
                o_in,
                o_out,
                &attn_out[..o_in],
                o_raw,
                &mut scratch_a[..o_out],
            ) {
                return true;
            }
        }
        // Fallback: stack gemm for O
        host.gemm(
            o_raw,
            out_info,
            &attn_out[..o_in],
            &mut scratch_a[..o_out],
            o_in,
            o_out,
        )
    })
}
