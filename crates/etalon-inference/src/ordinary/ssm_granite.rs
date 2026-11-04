use super::{cpu::*, ssm_layout::*, HybridEngine, LayerProvider, RMS_NORM_EPS};
use crate::qwen::gated_deltanet::step_causal_conv1d;
use etalon_model::inference::LayerTensors;
impl<P: LayerProvider> HybridEngine<'_, P> {
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_granite_ssm_layer(
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
            self.host
                .event(super::LayerEvent::GraniteModelMissing(layer));
            return false;
        }
        let h = P::hyperparams(index);
        let n_embd = h.n_embd as usize;
        let tds = P::data_start(index);

        let ssm_in_info = match tensors.ssm_in.as_ref() {
            Some(i) => *i,
            None => return false,
        };
        let ssm_conv1d_info = match tensors.ssm_conv1d.as_ref() {
            Some(i) => *i,
            None => return false,
        };
        let ssm_out_info = match tensors.ssm_out.as_ref() {
            Some(i) => *i,
            None => return false,
        };

        let (inner_dim, proj_in) = super::matmul_dims(&ssm_in_info);
        if proj_in == 0 || inner_dim == 0 || inner_dim > MAX_SSM_INNER {
            return false;
        }

        // 1. Pre-norm: inline mmap access; borrow released after block.
        let ssm_input_is_norm = {
            if let Some(ni) = tensors.attn_norm.as_ref() {
                let mmap = self.host.model_bytes().unwrap();
                let mut norm_w = [0f32; MAX_SSM_INNER];
                let n = dequant_norm_row_into(mmap, tds, ni, &mut norm_w);
                if n >= n_embd {
                    scratch_b[..n_embd].copy_from_slice(&hidden[..n_embd]);
                    rms_norm_inplace(&mut scratch_b[..n_embd], &norm_w[..n_embd], RMS_NORM_EPS);
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };
        // scratch_b[..n_embd] holds the normalised input if ssm_input_is_norm.

        // 2. Input projection: scratch_b → scratch_a via ssm_in weight.
        //    dispatch_gemm_into takes &self — only works if no &mut self field is live.
        if ssm_input_is_norm {
            let snap_in: [f32; MAX_SSM_INNER] = {
                let mut buf = [0f32; MAX_SSM_INNER];
                buf[..n_embd].copy_from_slice(&scratch_b[..n_embd]);
                buf
            };
            if !self.host.gemm(
                index,
                &ssm_in_info,
                &snap_in[..proj_in],
                scratch_a,
                proj_in,
                inner_dim,
            ) {
                self.host
                    .event(super::LayerEvent::GraniteInputProjection(layer));
                return false;
            }
        } else {
            let snap_in: [f32; MAX_SSM_INNER] = {
                let mut buf = [0f32; MAX_SSM_INNER];
                buf[..n_embd.min(proj_in)].copy_from_slice(&hidden[..n_embd.min(proj_in)]);
                buf
            };
            if !self.host.gemm(
                index,
                &ssm_in_info,
                &snap_in[..proj_in],
                scratch_a,
                proj_in,
                inner_dim,
            ) {
                self.host
                    .event(super::LayerEvent::GraniteInputProjection(layer));
                return false;
            }
        }
        // scratch_a[..inner_dim] = projected input x

        // 3. Causal conv1d on projected input.
        //    mmap borrow ends after its last use (dequantize_row_into) — NLL allows
        //    ssm_recurrent_state mutable borrow to start in the same block afterwards.
        {
            let Ok(layout) = checked_ssm_layout_for(tensors, &h, n_embd) else {
                return false;
            };
            if layout.conv_len != CONV_HISTORY_DEPTH * inner_dim {
                return false;
            }
            let avail = self.host.state().len().saturating_sub(state_offset);
            if avail < layout.stride {
                return false;
            }
            let mut conv_w = [0f32; MAX_SSM_INNER * 4];
            {
                let mmap = self.host.model_bytes().unwrap();
                let conv_raw =
                    match etalon_model::ggml::fetch_tensor_bytes(mmap, tds, &ssm_conv1d_info) {
                        Ok(s) => s,
                        Err(_) => {
                            self.host
                                .event(super::LayerEvent::GraniteConvolutionFetch(layer));
                            return false;
                        }
                    };
                if etalon_model::ggml::dequantize_row_into(
                    conv_raw,
                    ssm_conv1d_info.ggml_type,
                    inner_dim * 4,
                    &mut conv_w[..inner_dim * 4],
                )
                .is_err()
                {
                    return false;
                }
            } // mmap and conv_raw dropped here
            let mut conv_inp = [0f32; MAX_SSM_INNER];
            conv_inp[..inner_dim].copy_from_slice(&scratch_a[..inner_dim]);
            let layer_state =
                &mut self.host.state_mut()[state_offset..state_offset + layout.stride];
            let (conv_region, _) = layer_state.split_at_mut(layout.conv_len);
            if step_causal_conv1d(
                conv_region,
                &conv_w[..inner_dim * 4],
                &conv_inp[..inner_dim],
                &mut scratch_b[..inner_dim],
            )
            .is_err()
            {
                return false;
            }
        }

        // 4. dt_bias softplus gate.
        if let Some(dt_info) = tensors.ssm_dt_bias.as_ref() {
            let mmap = self.host.model_bytes().unwrap();
            let mut dt_w = [0f32; MAX_SSM_INNER];
            let dt_n = dequant_norm_row_into(mmap, tds, dt_info, &mut dt_w);
            for i in 0..dt_n.min(inner_dim) {
                let v = scratch_b[i] + dt_w[i];
                scratch_b[i] = if v >= 20.0 {
                    v
                } else {
                    (1.0f32 + v.exp()).ln()
                };
            }
        }

        // 5. SSM norm.
        if let Some(norm_info) = tensors.ssm_norm.as_ref() {
            let mmap = self.host.model_bytes().unwrap();
            let mut ssm_norm_w = [0f32; MAX_SSM_INNER];
            let n = dequant_norm_row_into(mmap, tds, norm_info, &mut ssm_norm_w);
            if n >= inner_dim {
                rms_norm_inplace(
                    &mut scratch_b[..inner_dim],
                    &ssm_norm_w[..inner_dim],
                    RMS_NORM_EPS,
                );
            }
        }

        // 6. Output projection (snap scratch_b to avoid conflict with dispatch_gemm_into).
        let (out_in, out_out) = super::matmul_dims(&ssm_out_info);
        if out_in > inner_dim || out_out == 0 {
            return false;
        }
        let snap_b: [f32; MAX_SSM_INNER] = {
            let mut buf = [0f32; MAX_SSM_INNER];
            buf[..out_in].copy_from_slice(&scratch_b[..out_in]);
            buf
        };
        if !self.host.gemm(
            index,
            &ssm_out_info,
            &snap_b[..out_in],
            scratch_a,
            out_in,
            out_out,
        ) {
            return false;
        }

        // 7. Residual add.
        add_residual_inplace(
            &mut hidden[..emb_dim],
            &scratch_a[..out_out],
            emb_dim.min(out_out),
        );
        true
    }
}
