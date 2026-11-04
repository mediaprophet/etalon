use crate::gpu_resources::{ResourceBudget, ResourceError, ResourcePlan};
pub struct TopkResources {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
    pub values: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub staging: wgpu::Buffer,
    pub params: wgpu::Buffer,
}
impl TopkResources {
    pub fn build(
        device: &wgpu::Device,
        cache: Option<&wgpu::PipelineCache>,
        budget: ResourceBudget,
    ) -> Result<Self, ResourceError> {
        const CAND_BYTES: u64 = 262144 / 1024 * 64 * 4;
        budget.admit(ResourcePlan {
            buffer_bytes: CAND_BYTES * 4 + 16384,
            shader_source_bytes: super::topk::TOPK_REDUCTION_WGSL.len() as u64,
            compute_pipelines: 1,
        })?;
        if CAND_BYTES * 2 > budget.max_single_buffer_bytes
            || CAND_BYTES * 2 > device.limits().max_buffer_size
        {
            return Err(ResourceError::DeviceBufferLimit);
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("output_topk"),
            source: wgpu::ShaderSource::Wgsl(super::topk::TOPK_REDUCTION_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("output_topk_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("topk_block"),
            compilation_options: Default::default(),
            cache,
        });
        let pipeline_layout = pipeline.get_bind_group_layout(0);
        // Multi-chunk mega-pass: hold candidates for a full large-vocab sweep (≤256k ids),
        // k=1 → one cand per block; oversize for TOPK_MAX_K headroom on smaller vocabs.
        const MAX_VOCAB_RESIDENT: usize = 262_144;
        let max_blocks = (MAX_VOCAB_RESIDENT / super::topk::TOPK_BLOCK_SIZE).max(1);
        let cand_bytes = ((max_blocks * super::topk::TOPK_MAX_K).max(1) * 4) as wgpu::BufferAddress;
        let topk_cand_val_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TopkCandVal"),
            size: cand_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let topk_cand_idx_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TopkCandIdx"),
            size: cand_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let topk_cand_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TopkCandStaging"),
            size: cand_bytes * 2, // packed: [val .. | idx ..]
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let topk_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TopkParams"),
            // Browser top-1 uses one 256-byte-aligned slot per vocabulary
            // chunk; native paths continue to bind the first 16 bytes.
            size: 64 * 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            pipeline,
            layout: pipeline_layout,
            values: topk_cand_val_buf,
            indices: topk_cand_idx_buf,
            staging: topk_cand_staging,
            params: topk_params_buf,
        })
    }
}
