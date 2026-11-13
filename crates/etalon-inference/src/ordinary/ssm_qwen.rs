use super::{cpu::*, ssm_layout::*, HybridEngine, LayerProvider, RMS_NORM_EPS};
use crate::qwen::gated_deltanet::{step_causal_conv1d, step_gated_deltanet};
use etalon_inference_kernel::recurrent::{
    gated_rms_norm, normalize_qwen_qk, qwen_gates, silu_into, step_qwen_delta, QwenHeadLayout,
};
use etalon_model::inference::LayerTensors;
impl<P: LayerProvider> HybridEngine<'_, P> {
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_qwen_ssm_layer(
        &mut self,
        index: &P::Index,
        layer: u32,
        hidden: &mut [f32],
        emb_dim: usize,
        tensors: &LayerTensors,
        scratch_a: &mut [f32],
        scratch_b: &mut [f32],
        state_offset: usize,
    ) -> bool {
        if self.host.model_bytes().is_none() {
            self.host.event(super::LayerEvent::QwenModelMissing(layer));
            return false;
        }
        let h = P::hyperparams(index);
        let n_embd = h.n_embd as usize;
        let eps = if h.rms_norm_eps > 0.0 {
            h.rms_norm_eps
        } else {
            RMS_NORM_EPS
        };

        let qkv_info = match tensors.attn_qkv.as_ref() {
            Some(i) => *i,
            None => return false,
        };
        let alpha_info = match tensors.ssm_alpha.as_ref() {
            Some(i) => *i,
            None => return false,
        };
        let beta_info = match tensors.ssm_beta.as_ref() {
            Some(i) => *i,
            None => return false,
        };
        let conv_info = match tensors.ssm_conv1d.as_ref() {
            Some(i) => *i,
            None => return false,
        };
        let out_info = match tensors.ssm_out.as_ref() {
            Some(i) => *i,
            None => return false,
        };

        // Qwen 3.5 GatedDeltaNet linear-attention layer path when ssm_a, ssm_dt_bias, ssm_norm, and attn_gate are present.
        if let (Some(gate_info), Some(a_info), Some(dt_bias_info), Some(ssm_norm_info)) = (
            tensors.attn_gate.as_ref(),
            tensors.ssm_a.as_ref(),
            tensors.ssm_dt_bias.as_ref(),
            tensors.ssm_norm.as_ref(),
        ) {
            let (qkv_in, qkv_out) = super::matmul_dims(&qkv_info);
            let (gate_in, value_width) = super::matmul_dims(gate_info);
            let (alpha_in, value_heads) = super::matmul_dims(&alpha_info);
            let (beta_in, beta_heads) = super::matmul_dims(&beta_info);
            let key_dim = if h.ssm_state_size > 0 {
                h.ssm_state_size as usize
            } else {
                128
            };
            if value_heads == 0
                || value_heads != beta_heads
                || value_heads > MAX_SSM_HEAD
                || value_width == 0
                || value_width % value_heads != 0
                || value_width > MAX_SSM_INNER
                || qkv_out <= value_width
                || qkv_out > MAX_SSM_INNER
                || qkv_out > scratch_a.len()
                || qkv_out > scratch_b.len()
            {
                return false;
            }
            let value_dim = value_width / value_heads;
            let qk_total = qkv_out - value_width;
            if key_dim == 0 || qk_total % (2 * key_dim) != 0 {
                return false;
            }
            let key_heads = qk_total / (2 * key_dim);
            if key_heads == 0 || !value_heads.is_multiple_of(key_heads) {
                return false;
            }
            let qk_width = key_heads * key_dim;
            let expanded_qk = value_heads * key_dim;
            let head_layout = QwenHeadLayout {
                key_heads,
                value_heads,
                key_dim,
                value_dim,
            };
            let Ok(layout) = checked_ssm_layout_for(tensors, &h, n_embd) else {
                return false;
            };
            if state_offset
                .checked_add(layout.stride)
                .is_none_or(|end| end > self.host.state().len())
            {
                return false;
            }

            // 1. Pre-norm (attn_norm.weight)
            let mut qkv_input_snap = [0.0f32; MAX_SSM_INNER];
            qkv_input_snap[..n_embd].copy_from_slice(&hidden[..n_embd]);
            if let Some(ni) = tensors.attn_norm.as_ref() {
                let mut norm_w = [0.0f32; MAX_SSM_INNER];
                if self.host.norm(ni, &mut norm_w[..n_embd]) >= n_embd {
                    rms_norm_inplace(&mut qkv_input_snap[..n_embd], &norm_w[..n_embd], eps);
                }
            }

            // 2. Linear projections: attn_qkv, attn_gate (z), ssm_alpha, ssm_beta
            let mut z_buf = [0.0f32; MAX_SSM_INNER];
            let mut alpha_proj = [0.0f32; MAX_SSM_HEAD];
            let mut beta_proj = [0.0f32; MAX_SSM_HEAD];
            if !self.host.gemm(
                index,
                &qkv_info,
                &qkv_input_snap[..qkv_in.min(n_embd)],
                &mut scratch_a[..qkv_out],
                qkv_in,
                qkv_out,
            ) || !self.host.gemm(
                index,
                gate_info,
                &qkv_input_snap[..gate_in.min(n_embd)],
                &mut z_buf[..value_width],
                gate_in,
                value_width,
            ) || !self.host.gemm(
                index,
                &alpha_info,
                &qkv_input_snap[..alpha_in.min(n_embd)],
                &mut alpha_proj[..value_heads],
                alpha_in,
                value_heads,
            ) || !self.host.gemm(
                index,
                &beta_info,
                &qkv_input_snap[..beta_in.min(n_embd)],
                &mut beta_proj[..value_heads],
                beta_in,
                value_heads,
            ) {
                self.host.event(super::LayerEvent::QwenQkvProjection(layer));
                return false;
            }

            // 3. Causal Conv1D over all qkv_out (8192) channels + SiLU
            let mut conv_w = [0.0f32; MAX_SSM_INNER * 4];
            let conv_elems = qkv_out * 4;
            {
                let Some(conv_raw) = self.host.raw(&conv_info) else {
                    return false;
                };
                if etalon_model::ggml::dequantize_row_into(
                    conv_raw,
                    conv_info.ggml_type,
                    conv_elems,
                    &mut conv_w[..conv_elems],
                ) != Ok(conv_elems)
                {
                    return false;
                }
            }
            {
                let layer_state =
                    &mut self.host.state_mut()[state_offset..state_offset + layout.stride];
                let (conv_region, _) = layer_state.split_at_mut(layout.conv_len);
                if step_causal_conv1d(
                    conv_region,
                    &conv_w[..conv_elems],
                    &scratch_a[..qkv_out],
                    &mut scratch_b[..qkv_out],
                )
                .is_err()
                {
                    return false;
                }
            }
            if silu_into(&scratch_b[..qkv_out], &mut scratch_a[..qkv_out]).is_err() {
                return false;
            }

            // 4. L2-normalize Q and K per head, expand to value_heads, and scale Q by key_dim^-0.5
            let mut q_exp = [0.0f32; MAX_SSM_INNER];
            let mut k_exp = [0.0f32; MAX_SSM_INNER];
            if normalize_qwen_qk(
                head_layout,
                &scratch_a[..qkv_out],
                &mut q_exp[..expanded_qk],
                &mut k_exp[..expanded_qk],
            )
            .is_err()
            {
                return false;
            }

            // 5. Compute decay and beta gates from alpha_proj, beta_proj, ssm_a, ssm_dt.bias
            let mut a_raw = [0.0f32; MAX_SSM_HEAD];
            let mut dt_bias = [0.0f32; MAX_SSM_HEAD];
            if self.host.norm(a_info, &mut a_raw[..value_heads]) < value_heads
                || self.host.norm(dt_bias_info, &mut dt_bias[..value_heads]) < value_heads
            {
                return false;
            }
            let mut a_log = [0.0f32; MAX_SSM_HEAD];
            for h_idx in 0..value_heads {
                // GGUF `convert_hf_to_gguf.py` stores `-exp(A_log)` in `blk.N.ssm_a` (negative).
                let val = a_raw[h_idx];
                a_log[h_idx] = if val < 0.0 { (-val).ln() } else { val };
            }
            let mut decay = [0.0f32; MAX_SSM_HEAD];
            let mut beta = [0.0f32; MAX_SSM_HEAD];
            if qwen_gates(
                &alpha_proj[..value_heads],
                &beta_proj[..value_heads],
                &a_log[..value_heads],
                &dt_bias[..value_heads],
                &mut decay[..value_heads],
                &mut beta[..value_heads],
            )
            .is_err()
            {
                return false;
            }

            // 6. GatedDeltaNet recurrence per value head
            let mut ssm_out_buf = [0.0f32; MAX_SSM_INNER];
            {
                let layer_state =
                    &mut self.host.state_mut()[state_offset..state_offset + layout.stride];
                let delta_state =
                    &mut layer_state[layout.delta_offset..layout.delta_offset + layout.delta_len];
                let state_elems = key_dim * value_dim;
                let ratio = value_heads / key_heads;
                let v_slice = &scratch_a[2 * qk_width..2 * qk_width + value_width];
                for head in 0..value_heads {
                    // GGUF `convert_hf_to_gguf.py` transposes value heads to `[ratio, key_heads]`
                    // so `ggml_repeat_4d` maps value head `head` to key head `head % key_heads`.
                    let kh = head % key_heads;
                    let qs = (kh * ratio) * key_dim;
                    let vs = head * value_dim;
                    let ss = head * state_elems;
                    if step_qwen_delta(
                        &mut delta_state[ss..ss + state_elems],
                        &q_exp[qs..qs + key_dim],
                        &k_exp[qs..qs + key_dim],
                        &v_slice[vs..vs + value_dim],
                        decay[head],
                        beta[head],
                        key_dim,
                        value_dim,
                        &mut ssm_out_buf[vs..vs + value_dim],
                    )
                    .is_err()
                    {
                        return false;
                    }
                }
            }

            // 7. Per-head gated RMSNorm (`ssm_norm.weight` + SiLU(`z_buf`))
            let mut ssm_norm_w = [0.0f32; MAX_SSM_HEAD];
            if self.host.norm(ssm_norm_info, &mut ssm_norm_w[..value_dim]) < value_dim {
                return false;
            }
            for head in 0..value_heads {
                let vs = head * value_dim;
                if gated_rms_norm(
                    &ssm_out_buf[vs..vs + value_dim],
                    &z_buf[vs..vs + value_dim],
                    &ssm_norm_w[..value_dim],
                    eps,
                    &mut scratch_b[vs..vs + value_dim],
                )
                .is_err()
                {
                    return false;
                }
            }

            // 8. Output projection (`ssm_out.weight`) + residual add
            let (out_in, out_out) = super::matmul_dims(&out_info);
            if out_in > value_width || out_out == 0 || out_out > scratch_a.len() {
                return false;
            }
            if !self.host.gemm(
                index,
                &out_info,
                &scratch_b[..out_in],
                &mut scratch_a[..out_out],
                out_in,
                out_out,
            ) {
                return false;
            }
            add_residual_inplace(
                &mut hidden[..emb_dim],
                &scratch_a[..out_out],
                emb_dim.min(out_out),
            );
            return true;
        }

        let alpha_dim = alpha_info.dims[0] as usize;
        let d_head_ssm = if h.head_dim() > 0 {
            h.head_dim() as usize
        } else {
            MAX_SSM_HEAD
        };
        let n_heads_ssm = (if d_head_ssm > 0 {
            alpha_dim / d_head_ssm
        } else {
            1
        })
        .max(1);

        let (qkv_in, qkv_out) = super::matmul_dims(&qkv_info);
        let inner_dim = if let Some(ci) = tensors.ssm_conv1d.as_ref() {
            if ci.n_dims >= 3 && ci.dims[2] > 0 {
                ci.dims[2] as usize
            } else {
                ci.dims[0] as usize
            }
        } else {
            qkv_out / 3
        }
        .min(MAX_SSM_INNER);
        if inner_dim == 0 || qkv_out > scratch_a.len() {
            return false;
        }

        // 1. Pre-norm: snap normalised input into a stack buffer.
        let qkv_input_snap: [f32; MAX_SSM_INNER] = {
            let mut buf = [0f32; MAX_SSM_INNER];
            if let Some(ni) = tensors.attn_norm.as_ref() {
                let mut norm_w = [0f32; MAX_SSM_INNER];
                let n = self.host.norm(ni, &mut norm_w);
                if n >= n_embd {
                    buf[..n_embd].copy_from_slice(&hidden[..n_embd]);
                    rms_norm_inplace(&mut buf[..n_embd], &norm_w[..n_embd], RMS_NORM_EPS);
                } else {
                    buf[..n_embd].copy_from_slice(&hidden[..n_embd]);
                }
            } else {
                buf[..n_embd].copy_from_slice(&hidden[..n_embd]);
            }
            buf
        };

        // 2. Fused QKV projection → scratch_a[0..qkv_out].
        if !self.host.gemm(
            index,
            &qkv_info,
            &qkv_input_snap[..qkv_in.min(n_embd)],
            scratch_a,
            qkv_in,
            qkv_out,
        ) {
            self.host.event(super::LayerEvent::QwenQkvProjection(layer));
            return false;
        }

        // 3. Causal conv on channels (up to inner_dim); decode conv weights + snap input.
        let mut conv_w = [0f32; 32768];
        let conv_elems = (inner_dim * 4).min(conv_w.len());
        {
            let conv_raw = match self.host.raw(&conv_info) {
                Some(s) => s,
                None => {
                    return false;
                }
            };
            let _ = etalon_model::ggml::dequantize_row_into(
                conv_raw,
                conv_info.ggml_type,
                conv_elems,
                &mut conv_w[..conv_elems],
            );
        } // mmap borrow released
        {
            let layout = SsmLayerStateLayout::new(inner_dim, d_head_ssm, d_head_ssm, n_heads_ssm);
            let avail = self.host.state().len().saturating_sub(state_offset);
            if avail < layout.stride {
                return false;
            }
            let layer_state =
                &mut self.host.state_mut()[state_offset..state_offset + layout.stride];
            let (conv_region, _) = layer_state.split_at_mut(layout.conv_len);
            let _ = step_causal_conv1d(
                conv_region,
                &conv_w[..conv_elems],
                &scratch_a[..inner_dim],
                &mut scratch_b[..inner_dim],
            );
        }

        // 4. Alpha / beta gates.
        let mut alpha_buf = [1.0f32; MAX_SSM_HEAD];
        let mut beta_buf = [1.0f32; MAX_SSM_HEAD];
        {
            let _ = self
                .host
                .norm(&alpha_info, &mut alpha_buf[..d_head_ssm.min(MAX_SSM_HEAD)]);
            let _ = self
                .host
                .norm(&beta_info, &mut beta_buf[..d_head_ssm.min(MAX_SSM_HEAD)]);
        }

        // 5. GatedDeltaNet recurrence: scoped state borrow.
        let mut ssm_out_buf = [0f32; MAX_SSM_INNER];
        let ssm_len = n_heads_ssm * d_head_ssm;
        {
            let layout2 = SsmLayerStateLayout::new(inner_dim, d_head_ssm, d_head_ssm, n_heads_ssm);
            let layer_state2 =
                &mut self.host.state_mut()[state_offset..state_offset + layout2.stride];
            let delta_state =
                &mut layer_state2[layout2.delta_offset..layout2.delta_offset + layout2.delta_len];
            let d_state = d_head_ssm;
            for head in 0..n_heads_ssm {
                let hs = head * d_head_ssm;
                let he = hs + d_head_ssm;
                let vs = 2 * inner_dim + hs;
                let ve = 2 * inner_dim + he;
                let ss = head * d_state * d_head_ssm;
                let se = ss + d_state * d_head_ssm;
                if ve > scratch_a.len() || se > delta_state.len() {
                    break;
                }
                let _ = step_gated_deltanet(
                    &mut delta_state[ss..se],
                    &scratch_a[hs..he],
                    &scratch_b[hs..he],
                    &scratch_a[vs..ve],
                    &alpha_buf[..1],
                    &beta_buf[..1],
                    d_state,
                    d_head_ssm,
                    &mut ssm_out_buf[hs..he],
                );
            }
        }

        // 6. Sigmoid gate from q.
        for i in 0..ssm_len {
            let gate = 1.0f32 / (1.0 + (-scratch_a[i]).exp());
            ssm_out_buf[i] *= gate;
        }

        // 7. Output projection.
        let (out_in, out_out) = super::matmul_dims(&out_info);
        if out_in > ssm_out_buf.len() || out_out == 0 {
            return false;
        }
        if !self.host.gemm(
            index,
            &out_info,
            &ssm_out_buf[..out_in],
            scratch_a,
            out_in,
            out_out,
        ) {
            return false;
        }

        // 8. Residual add.
        add_residual_inplace(
            &mut hidden[..emb_dim],
            &scratch_a[..out_out],
            emb_dim.min(out_out),
        );
        true
    }
}
