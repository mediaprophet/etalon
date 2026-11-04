use super::attention::AttentionProvider;
use super::cpu::prepare_pre_norm_input;
use etalon_model::inference::{LayerTensors, ModelHyperparams, TensorInfo};
const MAX_HIDDEN_DIM: usize = 4096;
#[allow(async_fn_in_trait)]
pub trait AsyncAttentionProvider: AttentionProvider {
    async fn dispatch_ffn_block_pre_norm_async(
        &mut self,
        index: &Self::Index,
        hidden: &mut [f32],
        emb_dim: usize,
        tensors: &LayerTensors,
        scratch_a: &mut [f32],
        scratch_b: &mut [f32],
    ) -> bool;

    async fn dispatch_gemm_into_async(
        &self,
        index: &Self::Index,
        info: &TensorInfo,
        input: &[f32],
        out: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> bool;
    async fn dispatch_attention_pass_async(
        &self,
        hidden: &[f32],
        n_embd: usize,
        batch: u32,
        start: u32,
        layout: &Self::Layout,
        layer: u32,
        token: u32,
        h: &ModelHyperparams,
        info: &TensorInfo,
        weights: &[u8],
        kind: u32,
        groups: u32,
        out: Option<&mut [f32]>,
    ) -> bool;
}
pub async fn dispatch_attention_layer_async<P: AsyncAttentionProvider>(
    host: &mut P,
    index: &P::Index,
    layer: u32,
    token_idx: u32,
    hidden: &[f32],
    emb_dim: usize,
    tensors: &LayerTensors,
    scratch_a: &mut [f32],
    scratch_b: &mut [f32],
) -> Option<usize> {
    let layout = host.kv_layout()?;
    let q_info = tensors.attn_q.as_ref()?;
    let k_info = tensors.attn_k.as_ref()?;
    let v_info = tensors.attn_v.as_ref()?;
    let h = P::hyperparams(index);
    let n_head = h.n_head as usize;
    let n_kv = h.effective_n_kv_head() as usize;
    let head_dim = h.head_dim() as usize;
    if head_dim == 0 || n_head == 0 || n_kv == 0 {
        return None;
    }
    let q_dim = n_head * head_dim;
    if q_dim > scratch_a.len() || q_dim > scratch_b.len() || emb_dim < h.n_embd as usize {
        return None;
    }
    if !super::quant_support::ggml_gpu_quant_supported(q_info.ggml_type)
        || !super::quant_support::ggml_gpu_quant_supported(k_info.ggml_type)
        || !super::quant_support::ggml_gpu_quant_supported(v_info.ggml_type)
    {
        return None;
    }

    let mmap = host.model_bytes()?;
    let k_raw =
        etalon_model::ggml::fetch_tensor_bytes(mmap, P::tensor_data_start(index), k_info).ok()?;
    let v_raw =
        etalon_model::ggml::fetch_tensor_bytes(mmap, P::tensor_data_start(index), v_info).ok()?;
    let q_raw =
        etalon_model::ggml::fetch_tensor_bytes(mmap, P::tensor_data_start(index), q_info).ok()?;
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
            k_info,
            k_raw,
            1,
            n_kv as u32,
            None,
        )
        .await
    {
        return None;
    }
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
            v_info,
            v_raw,
            2,
            n_kv as u32,
            None,
        )
        .await
    {
        return None;
    }
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
        return None;
    }

    if let Some(out_info) = tensors.attn_output {
        let (o_in, o_out) = super::matmul_dims(&out_info);
        if o_in <= q_dim
            && host
                .dispatch_gemm_into_async(
                    index,
                    &out_info,
                    &scratch_b[..o_in],
                    &mut scratch_a[..o_out],
                    o_in,
                    o_out,
                )
                .await
        {
            return Some(o_out.min(emb_dim));
        }
    }
    let n = q_dim.min(emb_dim);
    scratch_a[..n].copy_from_slice(&scratch_b[..n]);
    Some(n)
}
