use super::types::*;
use crate::ordinary::attention::AttentionProvider;
use crate::ordinary::gpu_params::AttentionGpuParams;
use etalon_model::inference::{LayerTensors, ModelHyperparams, TensorInfo as GgufTensorInfo};
pub struct PrefillResources<'a> {
    pub attention_params_buf: Option<&'a wgpu::Buffer>,
    pub elem_params_buf: Option<&'a wgpu::Buffer>,
    pub gemm_params_buf: Option<&'a wgpu::Buffer>,
    pub prefill_scratch_buf: Option<&'a wgpu::Buffer>,
    pub gemm_input_buf: Option<&'a wgpu::Buffer>,
    pub gemm_output_buf: Option<&'a wgpu::Buffer>,
    pub norm_weight_buf: Option<&'a wgpu::Buffer>,
    pub mc8_k_proj_buf: Option<&'a wgpu::Buffer>,
    pub mc8_v_proj_buf: Option<&'a wgpu::Buffer>,
    pub prefill_work_buf_a: Option<&'a wgpu::Buffer>,
    pub prefill_work_buf_b: Option<&'a wgpu::Buffer>,
    pub gemm_aux_buf: Option<&'a wgpu::Buffer>,
    pub mc8_q_proj_buf: Option<&'a wgpu::Buffer>,
    pub gemm_max_out_dim: u32,
    pub gemm_max_input_floats: usize,
    pub mc8_weights_resident: bool,
}
pub trait PrefillProvider: AttentionProvider {
    type Pipeline;
    fn weight_role_buffer(&self, role: Mc8WeightRole) -> &wgpu::Buffer;
    fn encoding_resources(&self) -> super::encoding::EncodingResources<'_>;
    fn layer_stride(layout: &Self::Layout) -> u32;
    fn shader_supported(ggml_type: u32) -> bool;
    fn write_weight_words(&self, raw: &[u8], max_bytes: usize);
    fn attention_kv_mask_for_dispatch(
        layout: &Self::Layout,
        token: u32,
        kind: u32,
    ) -> ([u32; KV_ATTENTION_MASK_WORDS], u32, u32);

    fn prefill_resources(&self) -> PrefillResources<'_>;
    fn gpu_queue(&self) -> &wgpu::Queue;
    fn mc8_buffers_ready(&self) -> bool;
    fn begin(&self) -> Self::Pipeline;
    fn encoder(pipeline: &mut Self::Pipeline) -> &mut wgpu::CommandEncoder;
    fn max_context(layout: &Self::Layout) -> u32;
    fn mask_word_count() -> u32;
    fn tensors(index: &Self::Index, layer: u32) -> LayerTensors;
    fn log(&self, message: &str);
    fn attention_gpu_params(
        h: &ModelHyperparams,
        layout: &Self::Layout,
        layer: u32,
        token: u32,
        info: &GgufTensorInfo,
        raw: usize,
        kind: u32,
        batch: u32,
        start: u32,
        active: u32,
        words: u32,
        stride: u32,
    ) -> AttentionGpuParams;
    fn encode_gemm_bufs_offset(
        &self,
        pipeline: &mut Self::Pipeline,
        info: &GgufTensorInfo,
        raw: &[u8],
        n_in: usize,
        n_out: usize,
        input: &wgpu::Buffer,
        in_off: wgpu::BufferAddress,
        in_bytes: wgpu::BufferAddress,
        output: &wgpu::Buffer,
        out_off: wgpu::BufferAddress,
        out_bytes: wgpu::BufferAddress,
        n_batch: u32,
        in_row_stride: u32,
        out_row_stride: u32,
        gemm_dyn_offset: u32,
        layer: u32,
        weight_role: Mc8WeightRole,
    ) -> bool;
    fn encode_attention_batched_q_prefill(
        &self,
        pipeline: &mut Self::Pipeline,
        input: &wgpu::Buffer,
        in_off: wgpu::BufferAddress,
        in_bytes: wgpu::BufferAddress,
        output: &wgpu::Buffer,
        out_off: wgpu::BufferAddress,
        out_bytes: wgpu::BufferAddress,
        n_embd: usize,
        n_tokens: u32,
        layout: &Self::Layout,
        layer: u32,
        h: &ModelHyperparams,
        raw_weights: &[u8],
        attn_dyn_offset: u32,
    ) -> bool;
    fn encode_elem_offset(
        &self,
        pipeline: &mut Self::Pipeline,
        op: u32,
        n: u32,
        batch: u32,
        a: &wgpu::Buffer,
        a_off: wgpu::BufferAddress,
        a_bytes: wgpu::BufferAddress,
        b: &wgpu::Buffer,
        b_off: wgpu::BufferAddress,
        b_bytes: wgpu::BufferAddress,
        out: &wgpu::Buffer,
        out_off: wgpu::BufferAddress,
        out_bytes: wgpu::BufferAddress,
        a_row_stride: u32,
        b_row_stride: u32,
        out_row_stride: u32,
        a_slot: u32,
        b_slot: u32,
        out_slot: u32,
        elem_dyn_offset: u32,
    );
    fn encode_fused_ffn_expansion(
        &self,
        pipeline: &mut Self::Pipeline,
        layer: u32,
        input: &wgpu::Buffer,
        in_off: wgpu::BufferAddress,
        in_bytes: wgpu::BufferAddress,
        output: &wgpu::Buffer,
        out_off: wgpu::BufferAddress,
        out_bytes: wgpu::BufferAddress,
        n_out: u32,
        n_batch: u32,
        gemm_dyn_offset: u32,
    ) -> bool;
    fn mc8_norm_source(
        &self,
        mmap: &[u8],
        tensor_data_start: u64,
        info: &GgufTensorInfo,
        n_embd: usize,
        layer: u32,
        is_ffn: bool,
    ) -> Option<(&wgpu::Buffer, wgpu::BufferAddress)>;
    fn encode_attention_pass_gpu(
        &self,
        pipeline: &mut Self::Pipeline,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
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
        attn_dyn_offset: u32,
        weight_role: Mc8WeightRole,
    ) -> bool;
    fn mc8_flush(&self, pipeline: &mut Self::Pipeline);
    fn encode_attn_ffn_tail_gpu(
        &self,
        pipeline: &mut Self::Pipeline,
        index: &Self::Index,
        layer: u32,
        token_idx: u32,
        emb_dim: usize,
        tensors: &LayerTensors,
        token_hidden: &wgpu::Buffer,
        attn_input: Option<&wgpu::Buffer>,
        work_aliases_hidden: bool,
    ) -> bool;
}
