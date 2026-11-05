use super::*;
use crate::ordinary::{
    cpu::prepare_pre_norm_input,
    matmul_dims,
    quant_support::ggml_gpu_attention_shader_supported,
};
const MAX_HIDDEN_DIM: usize = 4096;
impl<P: AttentionProvider> AttentionEngine<'_, P> {
    pub fn dispatch_attention_layer(
        &mut self,
        index: &P::Index,
        layer: u32,
        token_idx: u32,
        hidden: &[f32],
        emb_dim: usize,
        tensors: &LayerTensors,
        scratch_a: &mut [f32],
        scratch_b: &mut [f32],
    ) -> Option<usize> {
        let layout = self.host.kv_layout()?;
        let q_info = tensors.attn_q.as_ref()?;
        let k_info = tensors.attn_k.as_ref()?;
        let v_info = tensors.attn_v.as_ref()?;
        let h = P::hyperparams(index);
        let n_head = h.n_head as usize;
        let n_kv = h.effective_n_kv_head() as usize;
        let head_dim = h.head_dim() as usize;
        {
            use std::sync::atomic::{AtomicBool, Ordering as AO};
            static A_DBG: AtomicBool = AtomicBool::new(false);
            if layer == 0 && self.host.debug_decode() && !A_DBG.swap(true, AO::Relaxed) {
                eprintln!(
                    "[attn-dbg] n_head={} n_kv={} head_dim={} q_dim={} qty={} kty={} vty={} ashader_q={}",
                    n_head,
                    n_kv,
                    head_dim,
                    n_head * head_dim,
                    q_info.ggml_type,
                    k_info.ggml_type,
                    v_info.ggml_type,
                    ggml_gpu_attention_shader_supported(q_info.ggml_type),
                );
            }
        }
        if head_dim == 0 || n_head == 0 || n_kv == 0 {
            return None;
        }
        let q_dim = n_head * head_dim;
        if q_dim > scratch_a.len() || q_dim > scratch_b.len() || emb_dim < h.n_embd as usize {
            return None;
        }
        // Use the attention-shader support set (fused_attention.wgsl handles Q4_0/Q5_0/Q8_0/Q4_K/Q6_K)
        // — NOT the narrower GEMM `ggml_gpu_quant_supported`, which wrongly rejected Q8_0 → no attention.
        if !ggml_gpu_attention_shader_supported(q_info.ggml_type)
            || !ggml_gpu_attention_shader_supported(k_info.ggml_type)
            || !ggml_gpu_attention_shader_supported(v_info.ggml_type)
        {
            return None;
        }

        let mmap = self.host.model_bytes()?;
        let k_raw =
            etalon_model::ggml::fetch_tensor_bytes(mmap, P::tensor_data_start(index), k_info)
                .ok()?;
        let v_raw =
            etalon_model::ggml::fetch_tensor_bytes(mmap, P::tensor_data_start(index), v_info)
                .ok()?;
        let q_raw =
            etalon_model::ggml::fetch_tensor_bytes(mmap, P::tensor_data_start(index), q_info)
                .ok()?;
        let n_embd = h.n_embd as usize;

        // Pre-norm (attn_norm) on the residual stream before Q/K/V — REQUIRED on all targets.
        // (Native previously skipped this → the residual stream exploded layer-over-layer to inf.)
        let mut norm_w_attn = [0f32; MAX_HIDDEN_DIM];
        let mut h_norm_attn = [0f32; MAX_HIDDEN_DIM];
        let hidden_input = prepare_pre_norm_input(
            &hidden[..emb_dim],
            emb_dim,
            tensors.attn_norm.as_ref(),
            Some(&mmap[..]),
            P::tensor_data_start(index),
            &mut h_norm_attn,
            &mut norm_w_attn,
        );

        // CUDA residual path: Q/K/V on sticky multi-weight slab (one x upload for three GEMVs),
        // then host RoPE + KV write + SDPA, then O-proj via CUDA GEMV.
        if self.host.prefer_cuda()
            && q_info.ggml_type == etalon_model::ggml::GGML_TYPE_Q4_K_SOA
            && k_info.ggml_type == etalon_model::ggml::GGML_TYPE_Q4_K_SOA
            && v_info.ggml_type == etalon_model::ggml::GGML_TYPE_Q4_K_SOA
        {
            let kv_dim = n_kv * head_dim;
            if q_dim <= scratch_a.len()
                && kv_dim <= scratch_b.len()
                && self.host.try_cuda_soa_attention_layer(
                    hidden_input,
                    n_embd,
                    q_dim,
                    kv_dim,
                    n_head,
                    n_kv,
                    head_dim,
                    &layout,
                    layer,
                    token_idx,
                    &h,
                    q_raw,
                    k_raw,
                    v_raw,
                    tensors.attn_output.as_ref(),
                    mmap,
                    P::tensor_data_start(index),
                    scratch_a,
                    scratch_b,
                )
            {
                // O-proj lands in scratch_a; size is emb_dim for standard Llama.
                return Some(n_embd.min(emb_dim).min(scratch_a.len()));
            }
        }
        if self.host.attention_preproject()
            && self.host.dispatch_attention_kv_preproject_fused(
                hidden_input,
                n_embd,
                &layout,
                layer,
                token_idx,
                &h,
                k_info,
                k_raw,
                v_info,
                v_raw,
                n_kv * head_dim,
            )
        {
            if self.host.attention_o_fuse() {
                if let Some(out_info) = tensors.attn_output {
                    let (o_in, o_out) = matmul_dims(&out_info);
                    if o_in <= q_dim && o_out <= scratch_a.len() {
                        if let Ok(o_raw) = etalon_model::ggml::fetch_tensor_bytes(
                            mmap,
                            P::tensor_data_start(index),
                            &out_info,
                        ) {
                            if self.host.dispatch_attention_q_o_fused(
                                hidden_input,
                                n_embd,
                                &layout,
                                layer,
                                token_idx,
                                &h,
                                q_info,
                                q_raw,
                                &out_info,
                                o_raw,
                                &mut scratch_a[..o_out],
                                o_in,
                                o_out,
                            ) {
                                return Some(o_out.min(emb_dim));
                            }
                        }
                    }
                }
            }
            if self.host.dispatch_attention_pass(
                hidden_input,
                n_embd,
                1,
                token_idx,
                &layout,
                layer,
                token_idx,
                &h,
                q_info,
                q_raw,
                0,
                n_head as u32,
                None,
                Some(&mut scratch_b[..q_dim]),
            ) {
                if let Some(out_info) = tensors.attn_output {
                    let (o_in, o_out) = matmul_dims(&out_info);
                    if o_in <= q_dim
                        && self.host.dispatch_gemm_into(
                            index,
                            &out_info,
                            &scratch_b[..o_in],
                            &mut scratch_a[..o_out],
                            o_in,
                            o_out,
                        )
                    {
                        return Some(o_out.min(emb_dim));
                    }
                } else {
                    let n = q_dim.min(emb_dim);
                    scratch_a[..n].copy_from_slice(&scratch_b[..n]);
                    return Some(n);
                }
            }
        }

        if !self.host.dispatch_attention_pass(
            hidden_input,
            n_embd,
            1,
            token_idx,
            &layout,
            layer,
            token_idx,
            &h,
            k_info,
            k_raw,
            1,
            n_kv as u32,
            None,
            None,
        ) {
            return None;
        }

        if !self.host.dispatch_attention_pass(
            hidden_input,
            n_embd,
            1,
            token_idx,
            &layout,
            layer,
            token_idx,
            &h,
            v_info,
            v_raw,
            2,
            n_kv as u32,
            None,
            None,
        ) {
            return None;
        }
        if self.host.attention_o_fuse() {
            if let Some(out_info) = tensors.attn_output {
                let (o_in, o_out) = matmul_dims(&out_info);
                if o_in <= q_dim && o_out <= scratch_a.len() {
                    if let Ok(o_raw) = etalon_model::ggml::fetch_tensor_bytes(
                        mmap,
                        P::tensor_data_start(index),
                        &out_info,
                    ) {
                        if self.host.dispatch_attention_q_o_fused(
                            hidden_input,
                            n_embd,
                            &layout,
                            layer,
                            token_idx,
                            &h,
                            q_info,
                            q_raw,
                            &out_info,
                            o_raw,
                            &mut scratch_a[..o_out],
                            o_in,
                            o_out,
                        ) {
                            return Some(o_out.min(emb_dim));
                        }
                    }
                }
            }
        }
        if !self.host.dispatch_attention_pass(
            hidden_input,
            n_embd,
            1,
            token_idx,
            &layout,
            layer,
            token_idx,
            &h,
            q_info,
            q_raw,
            0,
            n_head as u32,
            None,
            Some(&mut scratch_b[..q_dim]),
        ) {
            return None;
        }

        if let Some(out_info) = tensors.attn_output {
            let (o_in, o_out) = matmul_dims(&out_info);
            if o_in <= q_dim
                && self.host.dispatch_gemm_into(
                    index,
                    &out_info,
                    &scratch_b[..o_in],
                    &mut scratch_a[..o_out],
                    o_in,
                    o_out,
                )
            {
                return Some(o_out.min(emb_dim));
            }
        }
        let n = q_dim.min(emb_dim);
        scratch_a[..n].copy_from_slice(&scratch_b[..n]);
        Some(n)
    }
}

