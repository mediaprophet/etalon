use super::async_attention::AsyncAttentionProvider;
use super::cpu::{add_residual_inplace, prepare_pre_norm_input};
use etalon_model::inference::LayerTensors;
const MAX_HIDDEN_DIM: usize = 4096;
pub async fn dispatch_attention_q_ffn_token_async<P: AsyncAttentionProvider>(
    host: &mut P,
    index: &P::Index,
    layer: u32,
    token_idx: u32,
    hidden: &mut [f32],
    emb_dim: usize,
    tensors: &LayerTensors,
    scratch_a: &mut [f32],
    scratch_b: &mut [f32],
) -> bool {
    let layout = match host.kv_layout() {
        Some(l) => l,
        None => return false,
    };
    let q_info = match tensors.attn_q.as_ref() {
        Some(i) => i,
        None => return false,
    };
    let h = P::hyperparams(index);
    let n_head = h.n_head as usize;
    let head_dim = h.head_dim() as usize;
    let q_dim = n_head * head_dim;
    if q_dim > scratch_a.len() || q_dim > scratch_b.len() || emb_dim < h.n_embd as usize {
        return false;
    }
    let mmap = match host.model_bytes() {
        Some(m) => m,
        None => return false,
    };
    let q_raw =
        match etalon_model::ggml::fetch_tensor_bytes(mmap, P::tensor_data_start(index), q_info) {
            Ok(s) => s,
            Err(_) => return false,
        };
    let n_embd = h.n_embd as usize;
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
    if !host
        .dispatch_attention_pass_async(
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
            Some(&mut scratch_b[..q_dim]),
        )
        .await
    {
        return false;
    }
    let mut attn_ok = false;
    if let Some(out_info) = tensors.attn_output.as_ref() {
        let (o_in, o_out) = super::matmul_dims(out_info);
        if o_in <= q_dim
            && host
                .dispatch_gemm_into_async(
                    index,
                    out_info,
                    &scratch_b[..o_in],
                    &mut scratch_a[..o_out],
                    o_in,
                    o_out,
                )
                .await
        {
            add_residual_inplace(
                &mut hidden[..emb_dim],
                &scratch_a[..o_out],
                emb_dim.min(o_out),
            );
            attn_ok = true;
        }
    } else {
        let n = q_dim.min(emb_dim);
        add_residual_inplace(&mut hidden[..emb_dim], &scratch_b[..n], n);
        attn_ok = true;
    }
    if !attn_ok {
        return false;
    }
    host.dispatch_ffn_block_pre_norm_async(index, hidden, emb_dim, tensors, scratch_a, scratch_b)
        .await
}
