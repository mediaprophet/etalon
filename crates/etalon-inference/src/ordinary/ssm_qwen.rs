use super::{cpu::*, ssm_layout::*, HybridEngine, LayerProvider, RMS_NORM_EPS};
use crate::qwen::gated_deltanet::{step_causal_conv1d, step_gated_deltanet};
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
        let _tds = P::data_start(index);

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
