//! Portable GPU material storage and fixed fourteen-binding construction.
use etalon as wgpu;
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;
pub const MATERIAL_UNIFORM_STRIDE: u64 = 512;
pub struct MaterialGpu<D> {
    pub _uniform_buffer: wgpu::Buffer,
    pub environment_uniform_buffer: wgpu::Buffer,
    pub material_bind_groups: Vec<wgpu::BindGroup>,
    pub draws: Vec<D>,
    pub transparent_draw_order: Vec<usize>,
}
pub struct MaterialSlots<'a> {
    pub views: [&'a wgpu::TextureView; 6],
    pub samplers: [&'a wgpu::Sampler; 6],
}
impl<D> MaterialGpu<D> {
    pub fn update_environment<E: bytemuck::Pod>(&self, queue: &wgpu::Queue, environment: &E) {
        queue.write_buffer(
            &self.environment_uniform_buffer,
            0,
            bytemuck::bytes_of(environment),
        );
    }
}
pub fn create<'a, U: bytemuck::Pod, E: bytemuck::Pod, D>(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniforms: &[U],
    environment: &E,
    draws: Vec<D>,
    transparent_draw_order: Vec<usize>,
    mut slots: impl FnMut(usize) -> Result<MaterialSlots<'a>, String>,
) -> Result<MaterialGpu<D>, String> {
    if std::mem::size_of::<U>() != MATERIAL_UNIFORM_STRIDE as usize || uniforms.is_empty() {
        return Err("Material uniform ABI mismatch".into());
    }
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("portal-mesh-material-uniforms"),
        contents: bytemuck::cast_slice(uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let environment_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("portal-mesh-environment-lighting"),
        contents: bytemuck::bytes_of(environment),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let mut material_bind_groups = Vec::with_capacity(uniforms.len());
    for index in 0..uniforms.len() {
        let slots = slots(index)?;
        let entries: [wgpu::BindGroupEntry<'_>; 14] =
            std::array::from_fn(|binding| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: match binding {
                    0 => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buffer,
                        offset: 0,
                        size: NonZeroU64::new(MATERIAL_UNIFORM_STRIDE),
                    }),
                    1..=6 => wgpu::BindingResource::TextureView(slots.views[binding - 1]),
                    7..=12 => wgpu::BindingResource::Sampler(slots.samplers[binding - 7]),
                    _ => environment_uniform_buffer.as_entire_binding(),
                },
            });
        material_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("portal-mesh-material-textures-and-factors"),
            layout,
            entries: &entries,
        }));
    }
    Ok(MaterialGpu {
        _uniform_buffer: uniform_buffer,
        environment_uniform_buffer,
        material_bind_groups,
        draws,
        transparent_draw_order,
    })
}
pub fn layout_entries(binding_offset: u32) -> [wgpu::BindGroupLayoutEntry; 12] {
    std::array::from_fn(|slot| wgpu::BindGroupLayoutEntry {
        binding: binding_offset + slot as u32,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: if slot < 6 {
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            }
        } else {
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
        },
        count: None,
    })
}
pub fn create_layout(device: &wgpu::Device, environment_bytes: u64) -> wgpu::BindGroupLayout {
    let texture = layout_entries(1);
    let entries: [wgpu::BindGroupLayoutEntry; 14] = std::array::from_fn(|binding| match binding {
        0 => wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: NonZeroU64::new(MATERIAL_UNIFORM_STRIDE),
            },
            count: None,
        },
        1..=12 => texture[binding - 1].clone(),
        _ => wgpu::BindGroupLayoutEntry {
            binding: 13,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(environment_bytes),
            },
            count: None,
        },
    });
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("portal-mesh-material-textures-layout"),
        entries: &entries,
    })
}
