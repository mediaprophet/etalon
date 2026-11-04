use super::*;

pub fn target_bytes(width: u32, height: u32, scene_format: etalon::TextureFormat) -> Option<u64> {
    let scene_bytes = u64::from(scene_format.block_copy_size(None)?);
    u64::from(width)
        .checked_mul(u64::from(height))?
        .checked_mul(scene_bytes.checked_mul(2)?.checked_add(16)?)
        .and_then(|bytes| {
            bytes.checked_add(PARAMS_BYTES + OUTPUT_PARAMS_BYTES + DEPTH_PRODUCER_PARAMS_BYTES)
        })
}

pub(super) fn create_texture(
    device: &etalon::Device,
    label: &str,
    width: u32,
    height: u32,
    format: etalon::TextureFormat,
    usage: etalon::TextureUsages,
) -> etalon::Texture {
    device.create_texture(&etalon::TextureDescriptor {
        label: Some(label),
        size: etalon::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

pub(super) fn float_texture_entry(binding: u32) -> etalon::BindGroupLayoutEntry {
    etalon::BindGroupLayoutEntry {
        binding,
        visibility: etalon::ShaderStages::FRAGMENT,
        ty: etalon::BindingType::Texture {
            sample_type: etalon::TextureSampleType::Float { filterable: false },
            view_dimension: etalon::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

pub(super) fn depth_texture_entry(binding: u32) -> etalon::BindGroupLayoutEntry {
    etalon::BindGroupLayoutEntry {
        binding,
        visibility: etalon::ShaderStages::FRAGMENT,
        ty: etalon::BindingType::Texture {
            sample_type: etalon::TextureSampleType::Depth,
            view_dimension: etalon::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

pub(super) fn sampler_entry(binding: u32) -> etalon::BindGroupLayoutEntry {
    etalon::BindGroupLayoutEntry {
        binding,
        visibility: etalon::ShaderStages::FRAGMENT,
        ty: etalon::BindingType::Sampler(etalon::SamplerBindingType::NonFiltering),
        count: None,
    }
}

pub(super) fn uniform_entry(binding: u32, size: u64) -> etalon::BindGroupLayoutEntry {
    etalon::BindGroupLayoutEntry {
        binding,
        visibility: etalon::ShaderStages::FRAGMENT,
        ty: etalon::BindingType::Buffer {
            ty: etalon::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size),
        },
        count: None,
    }
}

pub(super) fn texture_entry<'a>(
    binding: u32,
    view: &'a etalon::TextureView,
) -> etalon::BindGroupEntry<'a> {
    etalon::BindGroupEntry {
        binding,
        resource: etalon::BindingResource::TextureView(view),
    }
}

pub(super) fn buffer_entry<'a>(
    binding: u32,
    buffer: &'a etalon::Buffer,
) -> etalon::BindGroupEntry<'a> {
    etalon::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub(super) fn create_pipeline(
    device: &etalon::Device,
    label: &str,
    shader: &etalon::ShaderModule,
    layout: &etalon::BindGroupLayout,
    vertex_entry: &str,
    fragment_entry: &str,
    format: etalon::TextureFormat,
) -> etalon::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: etalon::VertexState {
            module: shader,
            entry_point: Some(vertex_entry),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(etalon::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            compilation_options: Default::default(),
            targets: &[Some(etalon::ColorTargetState {
                format,
                blend: None,
                write_mask: etalon::ColorWrites::ALL,
            })],
        }),
        primitive: etalon::PrimitiveState::default(),
        depth_stencil: None,
        multisample: etalon::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}
