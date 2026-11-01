//! In-shader SieveStateMask and GPU block top-1 / top-k reduction.

use bytemuck::{Pod, Zeroable};

/// Raw WGSL shader source for the GPU block top-k reduction kernel.
pub const TOPK_REDUCTION_WGSL: &str = include_str!("shaders/topk_reduction.wgsl");

/// Uniform parameters matching the WGSL `Params` layout in `topk_reduction.wgsl`.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct TopkParams {
    pub n: u32,
    pub k: u32,
    pub block_size: u32,
    pub cand_base: u32,
}

const _: () = {
    assert!(core::mem::size_of::<TopkParams>() == 16);
    assert!(core::mem::align_of::<TopkParams>() == 16);
};

/// 80-byte std140-aligned logit mask header containing the 512-bit sieve bitmask
/// and autoregressive sampling hyper-parameters.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct LogitMaskHeader {
    pub mask_words: [u32; 16],
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub presence_penalty: f32,
}

const _: () = {
    assert!(core::mem::size_of::<LogitMaskHeader>() == 80);
    assert!(core::mem::align_of::<LogitMaskHeader>() == 16);
};

impl Default for LogitMaskHeader {
    fn default() -> Self {
        Self {
            mask_words: [u32::MAX; 16],
            temperature: 1.0,
            top_p: 1.0,
            min_p: 0.0,
            presence_penalty: 0.0,
        }
    }
}

/// Complete GPU compute pipeline for block top-k / argmax reduction.
pub struct TopkReductionPipeline {
    pub pipeline: etalon::ComputePipeline,
    pub layout: etalon::BindGroupLayout,
}

impl TopkReductionPipeline {
    pub fn new(device: &etalon::Device) -> Self {
        Self::try_new(device).expect("top-k reduction exceeds device compute limits")
    }

    pub fn try_new(device: &etalon::Device) -> Result<Self, crate::TensorPipelineError> {
        crate::kernel_limits::check(&device.limits(), 3, 8_704)?;
        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("etalon-topk-reduction-shader"),
            source: etalon::ShaderSource::Wgsl(TOPK_REDUCTION_WGSL.into()),
        });

        let layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("etalon-topk-reduction-layout"),
            entries: &[
                etalon::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: core::num::NonZeroU64::new(
                            core::mem::size_of::<TopkParams>() as u64,
                        ),
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("etalon-topk-reduction-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("etalon-topk-reduction-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("topk_block"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self { pipeline, layout })
    }

    pub fn dispatch<'a>(
        &'a self,
        cpass: &mut etalon::ComputePass<'a>,
        bind_group: &'a etalon::BindGroup,
        num_blocks: u32,
    ) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(num_blocks, 1, 1);
    }
}
