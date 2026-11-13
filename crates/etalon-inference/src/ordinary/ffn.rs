use super::{cpu::*, HybridEngine, LayerProvider};
use etalon_model::inference::LayerTensors;
const MAX_HIDDEN_DIM: usize = 4096;
const MAX_STACK_GEMM_DIM: usize = 20480;
impl<P: LayerProvider> HybridEngine<'_, P> {
    pub fn ffn(
        &mut self,
        index: &P::Index,
        hidden: &mut [f32],
        emb_dim: usize,
        tensors: &LayerTensors,
        scratch_a: &mut [f32],
    ) -> bool {
        // SwiGLU FFN with ffn_norm pre-norm — REQUIRED on all targets (native previously ran a
        // norm-less ReLU chain on the raw residual → exponential blow-up to inf).
        {
            let mut norm_w_ffn = [0f32; MAX_HIDDEN_DIM];
            let mut h_norm_ffn = [0f32; MAX_HIDDEN_DIM];
            let ffn_input = prepare_pre_norm_input(
                &hidden[..emb_dim],
                emb_dim,
                tensors.ffn_norm.as_ref(),
                self.host.model_bytes(),
                P::data_start(index),
                &mut h_norm_ffn,
                &mut norm_w_ffn,
            );
            let gate_info = match tensors.ffn_gate.as_ref() {
                Some(i) => i,
                None => {
                    if tensors.moe_router.is_some() {
                        let moe_ok = self.host.moe(index, emb_dim, tensors, scratch_a, ffn_input);
                        if moe_ok {
                            add_residual_inplace(
                                &mut hidden[..emb_dim],
                                &scratch_a[..emb_dim],
                                emb_dim,
                            );
                            return true;
                        }
                    }
                    return false;
                }
            };
            let up_info = match tensors.ffn_up.as_ref() {
                Some(i) => i,
                None => return false,
            };
            let down_info = match tensors.ffn_down.as_ref() {
                Some(i) => i,
                None => return false,
            };
            let (gate_in, n_ffn) = super::matmul_dims(gate_info);
            let (up_in, up_out) = super::matmul_dims(up_info);
            let (dn_in, dn_out) = super::matmul_dims(down_info);
            if gate_in > emb_dim
                || up_in != gate_in
                || up_out != n_ffn
                || dn_in != n_ffn
                || n_ffn > MAX_STACK_GEMM_DIM
                || dn_out > scratch_a.len()
            {
                return false;
            }
            // AWQ step 1 (calibration only; no-op in production): record per-input-channel salience
            // at the post-ffn_norm FFN input — the activation that scales the gate/up projections.
            self.host.record_ffn_input(&ffn_input[..gate_in]);
            let mut gate_buf = [0f32; MAX_STACK_GEMM_DIM];
            let mut up_buf = [0f32; MAX_STACK_GEMM_DIM];
            if !self.host.gemm(
                index,
                gate_info,
                &ffn_input[..gate_in],
                &mut gate_buf[..n_ffn],
                gate_in,
                n_ffn,
            ) {
                return false;
            }
            if !self.host.gemm(
                index,
                up_info,
                &ffn_input[..up_in],
                &mut up_buf[..n_ffn],
                up_in,
                n_ffn,
            ) {
                return false;
            }
            silu_inplace(&mut gate_buf[..n_ffn], n_ffn);
            for i in 0..n_ffn {
                gate_buf[i] *= up_buf[i];
            }
            if !self.host.gemm(
                index,
                down_info,
                &gate_buf[..dn_in],
                scratch_a,
                dn_in,
                dn_out,
            ) {
                return false;
            }
            add_residual_inplace(
                &mut hidden[..emb_dim],
                &scratch_a[..dn_out],
                emb_dim.min(dn_out),
            );
            return true;
        }
    }
}
