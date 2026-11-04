use super::*;
#[derive(Clone, Copy)]
pub enum Profile {
    Both,
    Begin,
    End,
}
#[derive(Clone, Copy)]
pub enum Phase {
    Gemm,
    OutputTopk,
    Embedding,
    FusedBlock,
}
pub struct ProjectionResources<'a> {
    pub gemm_input_buf: &'a Option<wgpu::Buffer>,
    pub gemm_weight_buf: &'a Option<wgpu::Buffer>,
    pub gemm_output_buf: &'a Option<wgpu::Buffer>,
    pub gemm_params_buf: &'a Option<wgpu::Buffer>,
    pub gemm_output_staging: &'a Option<wgpu::Buffer>,
    pub topk_params_buf: &'a Option<wgpu::Buffer>,
    pub topk_cand_val_buf: &'a Option<wgpu::Buffer>,
    pub topk_cand_idx_buf: &'a Option<wgpu::Buffer>,
    pub topk_cand_staging: &'a Option<wgpu::Buffer>,
    pub mc8_logits_resident_buf: &'a Option<wgpu::Buffer>,
    pub coop_gemv_pipeline: &'a wgpu::ComputePipeline,
    pub coop_gemv_mr_pipeline: &'a wgpu::ComputePipeline,
    pub pipeline: &'a wgpu::ComputePipeline,
    pub embedding_pipeline: &'a wgpu::ComputePipeline,
    pub mock_pipeline: &'a wgpu::ComputePipeline,
    pub embedding_bind_layout: &'a wgpu::BindGroupLayout,
    pub pipeline_bind_layout: &'a wgpu::BindGroupLayout,
    pub output_topk_pipeline: &'a Option<wgpu::ComputePipeline>,
    pub output_topk_bind_layout: &'a Option<wgpu::BindGroupLayout>,
    pub gemm_max_out_dim: u32,
    pub max_tensor_bytes: usize,
    pub mc8_logits_row_bytes: u32,
}
pub struct ProjectionTensor<'a> {
    pub shape: &'a [usize],
    pub byte_offset: u64,
    pub is_quantized_q4_k: bool,
}
pub trait ProjectionGpuProvider: ProjectionProvider {
    fn tensor_data_offset(&self) -> u64;
    fn backend_fused_block(&self, tensor: &ProjectionTensor<'_>, input: &[f32])
        -> Option<Vec<f32>>;
    fn readback_owned(
        &self,
        buffer: &wgpu::Buffer,
        bytes: u64,
        panic_on_error: bool,
    ) -> Option<Vec<f32>>;
    fn record_operations(&self, count: usize);
    fn projection_resources(&self) -> ProjectionResources<'_>;
    fn gpu_device(&self) -> &wgpu::Device;
    fn gpu_queue(&self) -> &wgpu::Queue;
    fn device(&self) -> &wgpu::Device {
        self.gpu_device()
    }
    fn coop_gemv_enabled(&self) -> bool;
    fn coop_gemv_workgroups(&self, rows: u32) -> u32;
    fn resident_weights_enabled(&self) -> bool;
    fn native_gemm_bind_layout(&self, coop: bool) -> &wgpu::BindGroupLayout;
    fn resident_weight_buffer(&self, key: u64, raw: &[u8]) -> Option<wgpu::Buffer>;
    fn write_weight_words(&self, raw: &[u8], max_bytes: usize);
    fn submit(&self, commands: wgpu::CommandBuffer);
    fn map_read(&self, buffer: &wgpu::Buffer, bytes: u64) -> bool;
    fn readback(&self, buffer: &wgpu::Buffer, bytes: u64, out: &mut [f32]) -> bool;
    fn pass_writes(&self, profile: Profile) -> Option<wgpu::ComputePassTimestampWrites<'static>>;
    fn resolve(&self, encoder: &mut wgpu::CommandEncoder);
    fn accumulate(&self, phase: Phase);
    fn cpu_gemm(
        &self,
        raw: &[u8],
        info: &GgufTensorInfo,
        input: &[f32],
        out: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> bool;
    fn cuda_gemm(
        &self,
        info: &GgufTensorInfo,
        raw: &[u8],
        input: &[f32],
        out: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> bool;
}
pub struct ProjectionGpuEngine<'a, P> {
    pub host: &'a P,
    resources: ProjectionResources<'a>,
}
impl<'a, P: ProjectionGpuProvider> ProjectionGpuEngine<'a, P> {
    pub fn new(host: &'a P) -> Self {
        Self {
            host,
            resources: host.projection_resources(),
        }
    }
}
impl<'a, P> core::ops::Deref for ProjectionGpuEngine<'a, P> {
    type Target = ProjectionResources<'a>;
    fn deref(&self) -> &Self::Target {
        &self.resources
    }
}
