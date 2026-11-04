use super::cpu::rms_norm_inplace;
use super::rope::rope_inplace;
use super::RMS_NORM_EPS;
use etalon_model::inference::{ModelHyperparams, TensorInfo as GgufTensorInfo};
const MAX_STACK_GEMM_OUT: usize = 10240;
const MAX_HIDDEN_DIM: usize = 4096;
const MAX_CONTEXT_WINDOW: u32 = 1024;
pub trait CpuAttentionHost {
    fn log(&self, message: core::fmt::Arguments<'_>);
    fn gemm(
        &self,
        raw: &[u8],
        info: &GgufTensorInfo,
        input: &[f32],
        out: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> bool;
    fn encode(&self, key: bool, vector: &[f32], k: usize, out: &mut [f32]);
    fn reconstruct(&self, key: bool, words: &[f32], k: usize, out: &mut [f32]);
    fn reconstruct_write(
        &self,
        layer: usize,
        key: bool,
        values: &mut [f32],
        heads: usize,
        dim: usize,
    );
    fn capture(&self, layer: usize, key: bool, values: &[f32], heads: usize, dim: usize);
}
#[derive(Clone, Copy)]
pub struct CpuKvLayout {
    pub max_context: u32,
    pub layer_stride: u32,
    pub slot_kv_elems: u32,
    pub n_kv_head: u32,
    pub head_dim: u32,
    pub dict_k: u32,
}
impl CpuKvLayout {
    fn ring_slot(&self, position: u32) -> u32 {
        position % self.max_context
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
    fn code_index(&self, layer: u32, slot: u32, head: u32, key: bool, i: u32) -> usize {
        let k = self.dict_k as usize;
        layer as usize * self.layer_stride as usize
            + slot as usize * 2 * self.n_kv_head as usize * k
            + (if key { 0 } else { self.n_kv_head as usize * k })
            + head as usize * k
            + i as usize
    }
}
pub fn cpu_attention_pass<H: CpuAttentionHost>(
    host: &H,
    kv: &mut [f32],
    hidden: &[f32],
    n_embd: usize,
    num_tokens_in_batch: u32,
    batch_start_token_idx: u32,
    layout: &CpuKvLayout,
    layer: u32,
    h: &ModelHyperparams,
    info: &GgufTensorInfo,
    raw_weights: &[u8],
    proj_kind: u32,
    norm_weight: Option<&[f32]>,
    mut readback_out: Option<&mut [f32]>,
) -> bool {
    let head_dim = h.head_dim() as usize;
    let n_head = h.n_head as usize;
    let n_kv = h.effective_n_kv_head() as usize;
    if head_dim == 0
        || head_dim > MAX_STACK_GEMM_OUT
        || n_head == 0
        || n_kv == 0
        || (norm_weight.is_some() && n_embd > MAX_HIDDEN_DIM)
        || layout.max_context == 0
    {
        return false;
    }
    let (n_in, out_dim) = super::matmul_dims(info);
    if out_dim == 0 || out_dim > MAX_STACK_GEMM_OUT || head_dim > out_dim {
        host.log(format_args!(
            "[cpu_attn] bad dims out_dim={out_dim} head_dim={head_dim}"
        ));
        return false;
    }
    let proj_heads = out_dim / head_dim;
    let q_dim = n_head * head_dim;
    let q_heads_per_kv = h.q_heads_per_kv() as usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let base_freq = h.effective_rope_freq_base();
    let rope_scale = h.effective_rope_scale();

    let dict_k = layout.dict_k as usize;
    let dict_mode = dict_k > 0;
    let mut proj = [0f32; MAX_STACK_GEMM_OUT];
    let mut norm_tok = [0f32; MAX_HIDDEN_DIM];
    for t in 0..num_tokens_in_batch as usize {
        let pos = batch_start_token_idx + t as u32;
        let slot = layout.ring_slot(pos);
        let tok_start = t * n_embd;
        if tok_start + n_embd > hidden.len() {
            host.log(format_args!(
                "[cpu_attn] hidden OOB t={t} need={}",
                tok_start + n_embd
            ));
            return false;
        }
        let htok = &hidden[tok_start..tok_start + n_embd];
        let gemm_in: &[f32] = if let Some(w) = norm_weight {
            if w.len() < n_embd {
                return false;
            }
            norm_tok[..n_embd].copy_from_slice(htok);
            rms_norm_inplace(&mut norm_tok[..n_embd], &w[..n_embd], RMS_NORM_EPS);
            &norm_tok[..n_embd]
        } else {
            htok
        };
        if !host.gemm(
            raw_weights,
            info,
            gemm_in,
            &mut proj[..out_dim],
            n_in,
            out_dim,
        ) {
            host.log(format_args!(
                    "[cpu_attn] proj failed kind={proj_kind} n_in={n_in} out_dim={out_dim} hidden={n_embd}"
                ));
            return false;
        }

        match proj_kind {
            1 => {
                rope_inplace(
                    &mut proj[..out_dim],
                    proj_heads,
                    head_dim,
                    pos,
                    base_freq,
                    rope_scale,
                );
                if dict_mode {
                    // Phase 4b: encode each post-RoPE K head vector to `dict_k` code words.
                    for kvh in 0..n_kv {
                        let base = layout.code_index(layer, slot, kvh as u32, true, 0);
                        if base + dict_k > kv.len() {
                            host.log(format_args!(
                                "[cpu_attn] K code OOB base={base} len={}",
                                kv.len()
                            ));
                            return false;
                        }
                        let vh = &proj[kvh * head_dim..(kvh + 1) * head_dim];
                        host.encode(true, vh, dict_k, &mut kv[base..base + dict_k]);
                    }
                } else {
                    // Phase 4a certify path: reconstruct-on-write (f32 store); no-op unless the dict
                    // runtime is enabled on the f32 layout.
                    host.reconstruct_write(
                        layer as usize,
                        true,
                        &mut proj[..out_dim],
                        n_kv,
                        head_dim,
                    );
                    for kvh in 0..n_kv {
                        for d in 0..head_dim {
                            let idx = layout.k_index(layer, slot, kvh as u32, d as u32);
                            if idx >= kv.len() {
                                host.log(format_args!(
                                    "[cpu_attn] K idx OOB idx={idx} len={}",
                                    kv.len()
                                ));
                                return false;
                            }
                            kv[idx] = proj[kvh * head_dim + d];
                        }
                    }
                }
                // W5b: tap the post-RoPE K vectors for the go/no-go capture. No-op unless enabled.
                host.capture(layer as usize, true, &proj[..out_dim], n_kv, head_dim);
            }
            2 => {
                if dict_mode {
                    // Phase 4b: encode each V head vector to `dict_k` code words.
                    for kvh in 0..n_kv {
                        let base = layout.code_index(layer, slot, kvh as u32, false, 0);
                        if base + dict_k > kv.len() {
                            host.log(format_args!(
                                "[cpu_attn] V code OOB base={base} len={}",
                                kv.len()
                            ));
                            return false;
                        }
                        let vh = &proj[kvh * head_dim..(kvh + 1) * head_dim];
                        host.encode(false, vh, dict_k, &mut kv[base..base + dict_k]);
                    }
                } else {
                    host.reconstruct_write(
                        layer as usize,
                        false,
                        &mut proj[..out_dim],
                        n_kv,
                        head_dim,
                    );
                    for kvh in 0..n_kv {
                        for d in 0..head_dim {
                            let idx = layout.v_index(layer, slot, kvh as u32, d as u32);
                            if idx >= kv.len() {
                                host.log(format_args!(
                                    "[cpu_attn] V idx OOB idx={idx} len={}",
                                    kv.len()
                                ));
                                return false;
                            }
                            kv[idx] = proj[kvh * head_dim + d];
                        }
                    }
                }
                host.capture(layer as usize, false, &proj[..out_dim], n_kv, head_dim);
            }
            0 => {
                let mut att_scores = [0f32; MAX_CONTEXT_WINDOW as usize];
                rope_inplace(
                    &mut proj[..out_dim],
                    proj_heads,
                    head_dim,
                    pos,
                    base_freq,
                    rope_scale,
                );
                let out_buf = match readback_out.as_mut() {
                    Some(out) => {
                        let out_off = t * q_dim;
                        if out_off + q_dim > out.len() {
                            host.log(format_args!(
                                "[cpu_attn] Q out OOB off={out_off} q_dim={q_dim} len={}",
                                out.len()
                            ));
                            return false;
                        }
                        &mut out[out_off..out_off + q_dim]
                    }
                    None => return false,
                };
                out_buf.fill(0.0);
                let pos_usize = pos as usize;
                if pos_usize >= MAX_CONTEXT_WINDOW as usize {
                    host.log(format_args!("[cpu_attn] pos OOB pos={pos}"));
                    return false;
                }

                let mut recon = [0f32; MAX_STACK_GEMM_OUT];
                for q_h in 0..n_head {
                    let kv_h = q_h / q_heads_per_kv;
                    let q_head_slice = &proj[q_h * head_dim..(q_h + 1) * head_dim];
                    let out_head_slice = &mut out_buf[q_h * head_dim..(q_h + 1) * head_dim];
                    let mut max_score = f32::NEG_INFINITY;
                    for past_pos in 0..=pos {
                        let past_slot = layout.ring_slot(past_pos);
                        if dict_mode {
                            recon[..head_dim].fill(0.0);
                            let base = layout.code_index(layer, past_slot, kv_h as u32, true, 0);
                            if base + dict_k <= kv.len() {
                                host.reconstruct(
                                    true,
                                    &kv[base..base + dict_k],
                                    dict_k,
                                    &mut recon[..head_dim],
                                );
                            }
                        }
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            let kval = if dict_mode {
                                recon[d]
                            } else {
                                let k_idx = layout.k_index(layer, past_slot, kv_h as u32, d as u32);
                                if k_idx >= kv.len() {
                                    host.log(format_args!(
                                        "[cpu_attn] SDPA K idx OOB idx={k_idx} len={}",
                                        kv.len()
                                    ));
                                    return false;
                                }
                                kv[k_idx]
                            };
                            dot += q_head_slice[d] * kval;
                        }
                        let score = dot * scale;
                        att_scores[past_pos as usize] = score;
                        if score > max_score {
                            max_score = score;
                        }
                    }
                    let mut sum_exp = 0.0f32;
                    for past_pos in 0..=pos {
                        let exp_val = (att_scores[past_pos as usize] - max_score).exp();
                        att_scores[past_pos as usize] = exp_val;
                        sum_exp += exp_val;
                    }
                    if sum_exp == 0.0 {
                        host.log(format_args!(
                                "[MC3] softmax sum_exp=0 layer={layer} pos={pos} q_h={q_h} max_score={max_score}"
                            ));
                        return false;
                    }
                    for past_pos in 0..=pos {
                        let prob = att_scores[past_pos as usize] / sum_exp;
                        let past_slot = layout.ring_slot(past_pos);
                        if dict_mode {
                            recon[..head_dim].fill(0.0);
                            let base = layout.code_index(layer, past_slot, kv_h as u32, false, 0);
                            if base + dict_k <= kv.len() {
                                host.reconstruct(
                                    false,
                                    &kv[base..base + dict_k],
                                    dict_k,
                                    &mut recon[..head_dim],
                                );
                            }
                        }
                        for d in 0..head_dim {
                            let vval = if dict_mode {
                                recon[d]
                            } else {
                                let v_idx = layout.v_index(layer, past_slot, kv_h as u32, d as u32);
                                if v_idx >= kv.len() {
                                    host.log(format_args!(
                                        "[cpu_attn] SDPA V idx OOB idx={v_idx} len={}",
                                        kv.len()
                                    ));
                                    return false;
                                }
                                kv[v_idx]
                            };
                            out_head_slice[d] += vval * prob;
                        }
                    }
                }
            }
            _ => return false,
        }
    }
    true
}
