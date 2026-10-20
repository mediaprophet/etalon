//! Online-softmax FlashAttention with guarded all-masked row handling.
//!
//! Provides the complete GPU compute pipeline for fused multi-head and grouped-query
//! attention (GQA) with in-shader RoPE, KV cache updates, and online softmax.

use bytemuck::{Pod, Zeroable};

/// Raw WGSL shader source for the fused FlashAttention kernel with guarded all-masked row handling.
pub const FUSED_ATTENTION_WGSL: &str = include_str!("shaders/fused_attention.wgsl");

/// Uniform parameters matching the WGSL `AttentionParams` layout.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct AttentionParams {
    pub n_embd: u32,
    pub n_head: u32,
    pub n_kv: u32,
    pub head_dim: u32,
    pub pos: u32,
    pub max_context: u32,
    pub layer_stride: u32,
    pub slot_kv_elems: u32,
    pub scale: f32,
    pub rope_theta: f32,
    pub rope_scale: f32,
    pub kv_quant: u32, // 0 = FP32, 1 = INT8
}

const _: () = {
    assert!(core::mem::size_of::<AttentionParams>() == 48);
    assert!(core::mem::align_of::<AttentionParams>() == 16);
};

impl Default for AttentionParams {
    fn default() -> Self {
        Self {
            n_embd: 4096,
            n_head: 32,
            n_kv: 8,
            head_dim: 128,
            pos: 0,
            max_context: 4096,
            layer_stride: 0,
            slot_kv_elems: 0,
            scale: 1.0 / (128.0f32).sqrt(),
            rope_theta: 10000.0,
            rope_scale: 1.0,
            kv_quant: 0,
        }
    }
}

/// Numerical online-softmax rescale calculation with guarded all-masked row handling.
#[inline(always)]
pub fn online_softmax_rescale(m_old: f32, m_new: f32) -> (f32, f32) {
    const MASK_FLOOR: f32 = -3.0e38;
    if m_new <= MASK_FLOOR {
        (0.0, 0.0)
    } else if m_old <= MASK_FLOOR {
        (0.0, 1.0)
    } else {
        let safe_diff = m_old - m_new;
        let rescale = safe_diff.exp();
        (rescale, 1.0)
    }
}

/// GPU compute pipeline for fused FlashAttention dispatch.
pub struct FusedAttentionPipeline {
    pub pipeline: etalon::ComputePipeline,
    pub layout: etalon::BindGroupLayout,
}

impl FusedAttentionPipeline {
    pub fn new(device: &etalon::Device) -> Self {
        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("etalon-fused-attention-shader"),
            source: etalon::ShaderSource::Wgsl(FUSED_ATTENTION_WGSL.into()),
        });

        let layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("etalon-fused-attention-layout"),
            entries: &[
                etalon::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: core::num::NonZeroU64::new(
                            core::mem::size_of::<AttentionParams>() as u64,
                        ),
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
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
            label: Some("etalon-fused-attention-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("etalon-fused-attention-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, layout }
    }

    pub fn dispatch<'a>(
        &'a self,
        cpass: &mut etalon::ComputePass<'a>,
        bind_group: &'a etalon::BindGroup,
        num_heads: u32,
    ) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(num_heads, 1, 1);
    }
}
