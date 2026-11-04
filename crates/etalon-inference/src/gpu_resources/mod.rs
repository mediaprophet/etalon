//! Device-supplied reusable inference resources; admission and device lifetime belong to callers.
mod budget;
mod buffers;
mod common;
mod ffn;
mod gemv;
mod projections;
pub mod shaders;

pub use budget::{ResourceBudget, ResourceError, ResourcePlan};
#[cfg(target_arch = "wasm32")]
pub use buffers::BrowserGemmBuffers;
pub use buffers::{GemmBuffers, GemmGeometry, KvBuffers, KvGeometry};
pub use common::CommonPipelines;
pub use ffn::FfnPipelines;
pub use gemv::GemvPipelines;
pub use projections::ProjectionsPipelines;

#[derive(Clone, Copy, Debug)]
pub struct PipelineOptions {
    pub subgroups: bool,
}

pub struct EngineResources {
    pub gemv: GemvPipelines,
    pub common: CommonPipelines,
    pub ffn: FfnPipelines,
    pub projections: ProjectionsPipelines,
    #[cfg(not(target_arch = "wasm32"))]
    pub attention_kv_gemm_params: wgpu::Buffer,
    #[cfg(not(target_arch = "wasm32"))]
    pub attention_kv_params: wgpu::Buffer,
}

impl EngineResources {
    pub fn plan(_options: PipelineOptions) -> ResourcePlan {
        #[cfg(not(target_arch = "wasm32"))]
        let sources = [
            shaders::FUSED_TRANSFORMER_WGSL,
            if _options.subgroups {
                shaders::COOP_GEMV_SUBGROUP_WGSL
            } else {
                ""
            },
            shaders::FUSED_TENSOR_CONTRACTION_WGSL,
            shaders::QUANTIZED_EMBEDDING_WGSL,
            shaders::FUSED_ATTENTION_WGSL,
            shaders::WASM_ELEMENTWISE_WGSL,
            shaders::DEQUANT_TEMPLATE_WGSL,
            shaders::FUSED_FFN_WGSL,
            shaders::DUAL_GEMV_WGSL,
            shaders::TRIPLE_GEMV_WGSL,
        ];
        #[cfg(target_arch = "wasm32")]
        let sources = [
            shaders::WASM_FUSED_TRANSFORMER_WGSL,
            shaders::QUANTIZED_EMBEDDING_WGSL,
            shaders::FUSED_ATTENTION_WGSL,
            shaders::WASM_ELEMENTWISE_WGSL,
            shaders::WASM_DEQUANT_TEMPLATE_WGSL,
            shaders::WASM_FUSED_FFN_WGSL,
        ];
        // Covers source modules, subgroup concatenation and both dequant-template expansions.
        let shader_source_bytes = sources.iter().map(|s| s.len() as u64).sum::<u64>() * 8;
        ResourcePlan {
            buffer_bytes: if cfg!(target_arch = "wasm32") {
                0
            } else {
                1024
            },
            shader_source_bytes,
            compute_pipelines: if cfg!(target_arch = "wasm32") { 8 } else { 20 },
        }
    }

    pub fn build(
        device: &wgpu::Device,
        cache: &Option<wgpu::PipelineCache>,
        options: PipelineOptions,
        budget: ResourceBudget,
        admit: impl FnOnce(ResourcePlan) -> Result<(), ResourceError>,
    ) -> Result<Self, ResourceError> {
        if options.subgroups && !device.features().contains(wgpu::Features::SUBGROUP) {
            return Err(ResourceError::UnsupportedSubgroups);
        }
        let plan = Self::plan(options);
        budget.admit(plan)?;
        #[cfg(not(target_arch = "wasm32"))]
        budget.check_buffer(device, 512, wgpu::BufferUsages::UNIFORM)?;
        admit(plan)?;
        let gemv = gemv::build(device, cache, options);
        let common = common::build(device, cache, options);
        let ffn = ffn::build(device, cache, options);
        let projections = projections::build(device, cache, options);
        Ok(Self {
            gemv,
            common,
            ffn,
            projections,
            #[cfg(not(target_arch = "wasm32"))]
            attention_kv_gemm_params: buffers::uniform(device, "AttentionKvGemmParams", 512),
            #[cfg(not(target_arch = "wasm32"))]
            attention_kv_params: buffers::uniform(device, "AttentionKvWriteParams", 512),
        })
    }
}
