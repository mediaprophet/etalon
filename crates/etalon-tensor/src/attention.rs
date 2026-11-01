//! Online-softmax FlashAttention with guarded all-masked row handling.
//!
//! Provides the complete GPU compute pipeline for fused multi-head and grouped-query
//! attention (GQA) with in-shader RoPE, KV cache updates, and online softmax.

use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;

mod packed;
pub use packed::{
    pack_attention_weights_and_mask, packed_fused_attention_wgsl, PackedAttentionError,
};

/// Raw WGSL shader source for the fused FlashAttention kernel with guarded all-masked row handling.
pub const FUSED_ATTENTION_WGSL: &str = include_str!("shaders/fused_attention.wgsl");

/// Legacy attention configuration. Use `FusedAttentionParams` for GPU uniform bindings.
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
        Self::try_new(device).expect("fused attention exceeds device compute limits")
    }

    pub fn try_new(device: &etalon::Device) -> Result<Self, crate::TensorPipelineError> {
        Self::build(device, false)
    }

    /// Four storage bindings; binding 1 contains weights followed by packed mask pages.
    pub fn try_new_packed(device: &etalon::Device) -> Result<Self, crate::TensorPipelineError> {
        Self::build(device, true)
    }

    pub fn new_packed(device: &etalon::Device) -> Self {
        Self::try_new_packed(device).expect("packed fused attention exceeds device compute limits")
    }

    fn build(device: &etalon::Device, packed: bool) -> Result<Self, crate::TensorPipelineError> {
        crate::kernel_limits::check(&device.limits(), if packed { 4 } else { 5 }, 11_104)?;
        let source = if packed {
            Cow::Owned(packed_fused_attention_wgsl())
        } else {
            Cow::Borrowed(FUSED_ATTENTION_WGSL)
        };
        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("etalon-fused-attention-shader"),
            source: etalon::ShaderSource::Wgsl(source),
        });

        let entries = [
            storage_entry(0, true),
            storage_entry(1, true),
            etalon::BindGroupLayoutEntry {
                binding: 2,
                visibility: etalon::ShaderStages::COMPUTE,
                ty: etalon::BindingType::Buffer {
                    ty: etalon::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: core::num::NonZeroU64::new(core::mem::size_of::<
                        FusedAttentionParams,
                    >() as u64),
                },
                count: None,
            },
            storage_entry(3, false),
            storage_entry(4, false),
            storage_entry(5, true),
        ];
        let layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("etalon-fused-attention-layout"),
            entries: &entries[..if packed { 5 } else { 6 }],
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

        Ok(Self { pipeline, layout })
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

    /// Dispatch Q+attention (`proj_kind == 0`) over per-token mask pages in a batch.
    pub fn dispatch_batched<'a>(
        &'a self,
        cpass: &mut etalon::ComputePass<'a>,
        bind_group: &'a etalon::BindGroup,
        num_heads: u32,
        batch_len: u32,
    ) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(num_heads, batch_len, 1);
    }
}

fn storage_entry(binding: u32, read_only: bool) -> etalon::BindGroupLayoutEntry {
    etalon::BindGroupLayoutEntry {
        binding,
        visibility: etalon::ShaderStages::COMPUTE,
        ty: etalon::BindingType::Buffer {
            ty: etalon::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Exact 96-byte uniform ABI for `FUSED_ATTENTION_WGSL` (binding 2).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct FusedAttentionParams {
    pub n_embd: u32,
    pub n_head: u32,
    pub n_kv_head: u32,
    pub head_dim: u32,
    pub q_heads_per_kv: u32,
    pub token_idx: u32,
    pub max_context: u32,
    pub layer_idx: u32,
    pub layer_stride: u32,
    pub slot_kv_elems: u32,
    pub weight_ggml_type: u32,
    pub weight_row_elems: u32,
    pub weight_byte_len: u32,
    pub proj_kind: u32,
    pub rope_theta_base: f32,
    pub rope_scale: f32,
    pub num_tokens_in_batch: u32,
    pub batch_start_token_idx: u32,
    pub mask_active: u32,
    pub mask_word_count: u32,
    pub out_stride_elems: u32,
    pub proj_row_stride: u32,
    pub kv_quant: u32,
    pub dict_pack: u32,
}

const _: () = assert!(core::mem::size_of::<FusedAttentionParams>() == 96);

impl Default for FusedAttentionParams {
    fn default() -> Self {
        Self {
            n_embd: 4096,
            n_head: 32,
            n_kv_head: 8,
            head_dim: 128,
            q_heads_per_kv: 4,
            max_context: 4096,
            slot_kv_elems: 1024,
            rope_theta_base: 10000.0,
            rope_scale: 1.0,
            num_tokens_in_batch: 1,
            ..Self::zeroed()
        }
    }
}
