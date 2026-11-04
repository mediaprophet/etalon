use super::{cpu::add_residual_inplace, HybridEngine, LayerProvider};
impl<P: LayerProvider> HybridEngine<'_, P> {
    pub fn layer(
        &mut self,
        index: &P::Index,
        layer: u32,
        token_idx: u32,
        hidden: &mut [f32],
        emb_dim: usize,
        scratch_a: &mut [f32],
        scratch_b: &mut [f32],
    ) -> bool {
        let tensors = P::tensors(index, layer);
        let mut attn_ok = false;
        // Decode-profiler: intra-layer split — attention (this block) vs FFN (below).
        // Native-only: `llm_bench` is `#[cfg(not(wasm32))]`; the wasm-full LLM bundle skips profiling.
        #[cfg(not(target_arch = "wasm32"))]
        let t_attn = std::time::Instant::now();

        if tensors.is_hybrid_ssm_layer() {
            // Hybrid SSM path (Granite / Qwen GatedDeltaNet): replaces the attention block.
            // The FFN block runs below as normal (SSM + FFN is the standard hybrid layer layout).
            attn_ok = self.dispatch(
                index, layer, hidden, emb_dim, &tensors, scratch_a, scratch_b,
            );
            if !attn_ok {
                return false;
            }
        } else if tensors.attn_q.is_some() && tensors.attn_k.is_some() && tensors.attn_v.is_some() {
            if let Some(n) = self.host.attention(
                index,
                layer,
                token_idx,
                &hidden[..emb_dim],
                emb_dim,
                &tensors,
                scratch_a,
                scratch_b,
            ) {
                add_residual_inplace(&mut hidden[..emb_dim], &scratch_a[..n], n);
                attn_ok = true;
            }
        } else if let Some(qkv_info) = tensors.attn_qkv {
            let (n_in, n_out) = super::matmul_dims(&qkv_info);
            if n_in <= emb_dim
                && self
                    .host
                    .gemm(index, &qkv_info, &hidden[..n_in], scratch_a, n_in, n_out)
            {
                if let Some(out_info) = tensors.attn_output {
                    let (out_in, out_out) = super::matmul_dims(&out_info);
                    if self.host.gemm(
                        index,
                        &out_info,
                        &scratch_a[..out_in.min(n_out)],
                        scratch_b,
                        out_in,
                        out_out,
                    ) {
                        add_residual_inplace(
                            &mut hidden[..emb_dim],
                            &scratch_b[..out_out],
                            emb_dim.min(out_out),
                        );
                        attn_ok = true;
                    }
                } else {
                    add_residual_inplace(
                        &mut hidden[..emb_dim],
                        &scratch_a[..n_out],
                        emb_dim.min(n_out),
                    );
                    attn_ok = true;
                }
            }
        } else if let Some(info) = tensors.attn_output {
            let (n_in, n_out) = super::matmul_dims(&info);
            if n_in <= emb_dim
                && self
                    .host
                    .gemm(index, &info, &hidden[..n_in], scratch_a, n_in, n_out)
            {
                add_residual_inplace(
                    &mut hidden[..emb_dim],
                    &scratch_a[..n_out],
                    emb_dim.min(n_out),
                );
                attn_ok = true;
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        self.host
            .profile_attention(t_attn.elapsed().as_nanos() as u64);

        if !attn_ok
            && tensors.attn_output.is_none()
            && tensors.ffn_gate.is_none()
            && tensors.moe_router.is_none()
            && !tensors.is_hybrid_ssm_layer()
        {
            return false;
        }

        self.host.layer_diagnostic(
            layer,
            attn_ok,
            &tensors,
            &scratch_a[..emb_dim],
            &hidden[..emb_dim],
        );

        #[cfg(not(target_arch = "wasm32"))]
        let t_ffn = std::time::Instant::now();
        let ffn_ok = self
            .host
            .ffn_dispatch(index, hidden, emb_dim, &tensors, scratch_a, scratch_b);
        #[cfg(not(target_arch = "wasm32"))]
        self.host.profile_ffn(t_ffn.elapsed().as_nanos() as u64);
        ffn_ok
    }
}
