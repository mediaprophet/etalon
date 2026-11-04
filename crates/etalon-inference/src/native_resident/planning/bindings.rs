use super::*;
use std::num::NonZeroU64;

pub(super) fn ubind(
    buf: &wgpu::Buffer,
    off: u64,
    size: Option<NonZeroU64>,
) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: buf,
        offset: off,
        size,
    })
}

pub(super) struct Bindings<'a> {
    pub device: &'a wgpu::Device,
    pub params: &'a wgpu::Buffer,
    pub attention_params: &'a wgpu::Buffer,
    pub gemm_layout: &'a wgpu::BindGroupLayout,
    pub attention_layout: &'a wgpu::BindGroupLayout,
    pub kv: &'a wgpu::Buffer,
    pub mask: &'a wgpu::Buffer,
    pub layer_f32s: u64,
}

impl Bindings<'_> {
    pub fn elem(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        a: wgpu::BindingResource<'_>,
        b: wgpu::BindingResource<'_>,
        out: &wgpu::Buffer,
        off: u64,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b,
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ubind(
                        self.params,
                        off,
                        NonZeroU64::new(std::mem::size_of::<ElemGpuParams>() as u64),
                    ),
                },
            ],
        })
    }

    pub fn gemm(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        input: &wgpu::Buffer,
        weight: wgpu::BindingResource<'_>,
        off: u64,
        out: wgpu::BindingResource<'_>,
        residual: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight,
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ubind(
                        self.params,
                        off,
                        NonZeroU64::new(std::mem::size_of::<GemmGpuParams>() as u64),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out,
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: residual.as_entire_binding(),
                },
            ],
        })
    }

    pub fn attention(
        &self,
        label: &str,
        input: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        off: u64,
        layer: u32,
        out: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: self.attention_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ubind(
                        self.attention_params,
                        off,
                        NonZeroU64::new(std::mem::size_of::<AttentionGpuParams>() as u64),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ubind(
                        self.kv,
                        layer as u64 * self.layer_f32s * 4,
                        NonZeroU64::new((self.layer_f32s * 4).max(4)),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.mask.as_entire_binding(),
                },
            ],
        })
    }
}
