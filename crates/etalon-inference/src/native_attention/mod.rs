use crate::ordinary::{
    gpu_params::*,
    matmul_dims,
    quant_support::{ggml_gpu_attention_shader_supported, ggml_gpu_gemm_supported},
};
use etalon_model::inference::{ModelHyperparams, TensorInfo as GgufTensorInfo};
mod resources;
pub use resources::AttentionResources;
pub struct AttentionGpuEngine<'a, P> {
    pub host: &'a P,
}
pub trait GpuAttentionProvider {
    type Layout;
    type Mask: AsRef<[u32]>;
    fn attention_resources(&self) -> AttentionResources<'_>;
    fn resident_weights_enabled(&self) -> bool;
    fn coop_gemv_enabled(&self) -> bool;
    fn gpu_device(&self) -> &wgpu::Device;
    fn gpu_queue(&self) -> &wgpu::Queue;
    fn device(&self) -> &wgpu::Device {
        self.gpu_device()
    }
    fn native_gemm_bind_layout(&self, coop: bool) -> &wgpu::BindGroupLayout;
    fn resident_weight_buffer(&self, key: u64, raw: &[u8]) -> Option<wgpu::Buffer>;
    fn submit(&self, command: wgpu::CommandBuffer) -> bool;
    fn readback(&self, staging: &wgpu::Buffer, out_bytes: u64, out: &mut [f32]) -> bool;
    fn pass_writes_begin(&self) -> Option<wgpu::ComputePassTimestampWrites<'static>>;
    fn pass_writes_end(&self) -> Option<wgpu::ComputePassTimestampWrites<'static>>;
    fn resolve(&self, encoder: &mut wgpu::CommandEncoder);
    fn accumulate_attention(&self);
    fn layer_stride(layout: &Self::Layout) -> u32;
    fn attention_kv_mask_for_dispatch(
        layout: &Self::Layout,
        token_idx: u32,
        proj_kind: u32,
    ) -> (Self::Mask, u32, u32);
    fn attention_gpu_params(
        h: &ModelHyperparams,
        layout: &Self::Layout,
        layer: u32,
        token_idx: u32,
        info: &GgufTensorInfo,
        raw_len: usize,
        proj_kind: u32,
        num_tokens_in_batch: u32,
        batch_start_token_idx: u32,
        mask_active: u32,
        mask_word_count: u32,
        out_stride_elems: u32,
    ) -> AttentionGpuParams;
}
mod fused_tail;
mod preproject;
