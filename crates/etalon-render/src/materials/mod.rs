//! Portable GPU material storage and fixed fourteen-binding construction.
use std::num::NonZeroU64;
use etalon::util::DeviceExt;
pub const MATERIAL_UNIFORM_STRIDE: u64 = 512;
pub struct MaterialGpu<D> {
    pub _uniform_buffer: etalon::Buffer,
    pub environment_uniform_buffer: etalon::Buffer,
    pub material_bind_groups: Vec<etalon::BindGroup>,
    pub draws: Vec<D>,
    pub transparent_draw_order: Vec<usize>,
}
pub struct MaterialSlots<'a> {
    pub views: [&'a etalon::TextureView; 6],
    pub samplers: [&'a etalon::Sampler; 6],
}
impl<D> MaterialGpu<D> {
    pub fn update_environment<E: bytemuck::Pod>(&self, queue: &etalon::Queue, environment: &E) {
        queue.write_buffer(
            &self.environment_uniform_buffer,
            0,
            bytemuck::bytes_of(environment),
        );
    }
}
pub fn create<'a, U: bytemuck::Pod, E: bytemuck::Pod, D>(
    device: &etalon::Device,
    layout: &etalon::BindGroupLayout,
    uniforms: &[U],
    environment: &E,
    draws: Vec<D>,
    transparent_draw_order: Vec<usize>,
    mut slots: impl FnMut(usize) -> Result<MaterialSlots<'a>, String>,
) -> Result<MaterialGpu<D>, String> {
    if std::mem::size_of::<U>() != MATERIAL_UNIFORM_STRIDE as usize || uniforms.is_empty() {
        return Err("Material uniform ABI mismatch".into());
    }
    let uniform_buffer = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
        label: Some("portal-mesh-material-uniforms"),
        contents: bytemuck::cast_slice(uniforms),
        usage: etalon::BufferUsages::UNIFORM,
    });
    let environment_uniform_buffer = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
        label: Some("portal-mesh-environment-lighting"),
        contents: bytemuck::bytes_of(environment),
        usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
    });
    let mut material_bind_groups = Vec::with_capacity(uniforms.len());
    for index in 0..uniforms.len() {
        let slots = slots(index)?;
        let entries: [etalon::BindGroupEntry<'_>; 14] =
            std::array::from_fn(|binding| etalon::BindGroupEntry {
                binding: binding as u32,
                resource: match binding {
                    0 => etalon::BindingResource::Buffer(etalon::BufferBinding {
                        buffer: &uniform_buffer,
                        offset: 0,
                        size: NonZeroU64::new(MATERIAL_UNIFORM_STRIDE),
                    }),
                    1..=6 => etalon::BindingResource::TextureView(slots.views[binding - 1]),
                    7..=12 => etalon::BindingResource::Sampler(slots.samplers[binding - 7]),
                    _ => environment_uniform_buffer.as_entire_binding(),
                },
            });
        material_bind_groups.push(device.create_bind_group(&etalon::BindGroupDescriptor {
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
pub fn layout_entries(binding_offset: u32) -> [etalon::BindGroupLayoutEntry; 12] {
    std::array::from_fn(|slot| etalon::BindGroupLayoutEntry {
        binding: binding_offset + slot as u32,
        visibility: etalon::ShaderStages::FRAGMENT,
        ty: if slot < 6 {
            etalon::BindingType::Texture {
                sample_type: etalon::TextureSampleType::Float { filterable: true },
                view_dimension: etalon::TextureViewDimension::D2,
                multisampled: false,
            }
        } else {
            etalon::BindingType::Sampler(etalon::SamplerBindingType::Filtering)
        },
        count: None,
    })
}
pub fn create_layout(device: &etalon::Device, environment_bytes: u64) -> etalon::BindGroupLayout {
    let texture = layout_entries(1);
    let entries: [etalon::BindGroupLayoutEntry; 14] = std::array::from_fn(|binding| match binding {
        0 => etalon::BindGroupLayoutEntry {
            binding: 0,
            visibility: etalon::ShaderStages::VERTEX_FRAGMENT,
            ty: etalon::BindingType::Buffer {
                ty: etalon::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: NonZeroU64::new(MATERIAL_UNIFORM_STRIDE),
            },
            count: None,
        },
        1..=12 => texture[binding - 1].clone(),
        _ => etalon::BindGroupLayoutEntry {
            binding: 13,
            visibility: etalon::ShaderStages::FRAGMENT,
            ty: etalon::BindingType::Buffer {
                ty: etalon::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(environment_bytes),
            },
            count: None,
        },
    });
    device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
        label: Some("portal-mesh-material-textures-layout"),
        entries: &entries,
    })
}
