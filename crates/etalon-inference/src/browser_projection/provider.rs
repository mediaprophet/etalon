use crate::projections::ProjectionProvider;
pub struct BrowserProjectionResources<'a> {
    pub gemm_input_buf: Option<&'a wgpu::Buffer>,
    pub gemm_weight_buf: Option<&'a wgpu::Buffer>,
    pub gemm_output_buf: Option<&'a wgpu::Buffer>,
    pub gemm_params_buf: Option<&'a wgpu::Buffer>,
    pub gemm_output_staging: Option<&'a wgpu::Buffer>,
    pub mc8_logits_resident_buf: Option<&'a wgpu::Buffer>,
    pub pipeline: &'a wgpu::ComputePipeline,
    pub mmv_q8_0_pipeline: &'a wgpu::ComputePipeline,
    pub gemm_max_out_dim: u32,
    pub max_tensor_bytes: usize,
    pub mc8_logits_row_bytes: u32,
}
#[allow(async_fn_in_trait)]
pub trait BrowserProjectionProvider: ProjectionProvider {
    type Top1Plan: Copy;
    fn top1_plan(vocab_size: usize, chunks: u32) -> Option<Self::Top1Plan>;
    fn prepare_browser_top1(&self, plan: Self::Top1Plan) -> bool;
    fn encode_browser_top1_chunk(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        logits: &wgpu::Buffer,
        staging: &wgpu::Buffer,
        plan: Self::Top1Plan,
        chunk: usize,
    ) -> bool;
    async fn read_browser_top1(
        &self,
        staging: &wgpu::Buffer,
        plan: Self::Top1Plan,
    ) -> Option<crate::projections::StreamingArgmaxResult>;
    fn tensor_data_offset(&self) -> u64;
    fn accumulate_fused_block(&self);
    fn record_fused_ops(&self, ops: usize);

    fn resources(&self) -> BrowserProjectionResources<'_>;
    fn gpu_device(&self) -> &wgpu::Device;
    fn gpu_queue(&self) -> &wgpu::Queue;
    fn write_weight_words(&self, raw: &[u8], max: usize);
    fn submit(&self, command: wgpu::CommandBuffer) -> bool;
    fn timestamps(&self) -> Option<wgpu::ComputePassTimestampWrites<'static>>;
    fn resolve(&self, encoder: &mut wgpu::CommandEncoder);
    fn accumulate_gemm(&self);
    fn buffers_ready(&self) -> bool;
    fn log(&self, message: &str);
    async fn await_map(&self, slice: wgpu::BufferSlice<'_>) -> bool;
}
