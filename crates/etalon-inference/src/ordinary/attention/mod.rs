use etalon_model::inference::{LayerTensors, ModelHyperparams, TensorInfo as GgufTensorInfo};
pub struct AttentionEngine<'a, P> {
    pub host: &'a mut P,
}
pub trait AttentionProvider {
    type Index;
    type Layout: Copy;
    fn kv_layout(&self) -> Option<Self::Layout>;
    fn hyperparams(index: &Self::Index) -> ModelHyperparams;
    fn tensor_data_start(index: &Self::Index) -> u64;
    fn model_bytes(&self) -> Option<&[u8]>;
    fn debug_decode(&self) -> bool;
    fn prefer_cuda(&self) -> bool;
    fn attention_preproject(&self) -> bool;
    fn attention_o_fuse(&self) -> bool;
    fn dispatch_gemm_into(
        &self,
        index: &Self::Index,
        info: &GgufTensorInfo,
        input: &[f32],
        out: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> bool;
    fn dispatch_ffn_block_pre_norm(
        &mut self,
        index: &Self::Index,
        hidden: &mut [f32],
        emb_dim: usize,
        tensors: &LayerTensors,
        scratch_a: &mut [f32],
        scratch_b: &mut [f32],
    ) -> bool;
    fn dispatch_attention_pass(
        &self,
        hidden: &[f32],
        n_embd: usize,
        num_tokens_in_batch: u32,
        batch_start_token_idx: u32,
        layout: &Self::Layout,
        layer: u32,
        token_idx: u32,
        h: &ModelHyperparams,
        info: &GgufTensorInfo,
        raw_weights: &[u8],
        proj_kind: u32,
        n_workgroups: u32,
        norm_weight: Option<&[f32]>,
        readback_out: Option<&mut [f32]>,
    ) -> bool;
    fn try_cuda_soa_attention_layer(
        &self,
        hidden_input: &[f32],
        n_embd: usize,
        q_dim: usize,
        kv_dim: usize,
        n_head: usize,
        n_kv: usize,
        head_dim: usize,
        layout: &Self::Layout,
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
    ) -> bool;
    fn dispatch_attention_kv_preproject_fused(
        &self,
        hidden: &[f32],
        n_embd: usize,
        layout: &Self::Layout,
        layer: u32,
        token_idx: u32,
        h: &ModelHyperparams,
        k_info: &GgufTensorInfo,
        k_raw: &[u8],
        v_info: &GgufTensorInfo,
        v_raw: &[u8],
        kv_dim: usize,
    ) -> bool;
    fn dispatch_attention_q_o_fused(
        &self,
        hidden: &[f32],
        n_embd: usize,
        layout: &Self::Layout,
        layer: u32,
        token_idx: u32,
        h: &ModelHyperparams,
        q_info: &GgufTensorInfo,
        q_raw: &[u8],
        o_info: &GgufTensorInfo,
        o_raw: &[u8],
        out: &mut [f32],
        o_in: usize,
        o_out: usize,
    ) -> bool;
}
mod layer;
mod token;
