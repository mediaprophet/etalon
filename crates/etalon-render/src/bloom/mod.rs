//! Persistent HDR bloom targets and exact Kawase pipeline.
use etalon as wgpu;
use wgpu::util::DeviceExt;
pub const BLOOM_WGSL: &str = concat!(
    include_str!("../shaders/output_transform.wgsl"),
    include_str!("../shaders/bloom.wgsl")
);
const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const BLOOM_THRESHOLD: f32 = 1.0;
const BLOOM_INTENSITY: f32 = 1.15;
const BLOOM_STRENGTH: f32 = 0.85;
const BLOOM_EXPOSURE: f32 = 1.05;
const KAWASE_OFFSETS: [f32; 4] = [1.0, 2.0, 4.0, 8.0];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BloomParamsGpu {
    pub threshold: f32,
    pub intensity: f32,
    pub offset: f32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompositeParamsGpu {
    pub exposure: f32,
    pub bloom_strength: f32,
    pub surface_is_srgb: f32,
    pub _pad1: f32,
    pub white_balance_gains: [f32; 3],
    pub _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BloomUniformBlock {
    pub bloom: BloomParamsGpu,
    pub composite: CompositeParamsGpu,
}

/// HDR scene target + half-res Kawase ping-pong (allocated only in `OperationalMode::Full`).
pub struct BloomChain<R> {
    pub hdr_texture: wgpu::Texture,
    pub hdr_view: wgpu::TextureView,
    pub blur_a: wgpu::Texture,
    pub blur_a_view: wgpu::TextureView,
    pub blur_b: wgpu::Texture,
    pub blur_b_view: wgpu::TextureView,
    pub dummy_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub uniform_buf: wgpu::Buffer,
    pub bind_layout: wgpu::BindGroupLayout,
    pub extract_pipeline: wgpu::RenderPipeline,
    pub kawase_pipeline: wgpu::RenderPipeline,
    pub composite_pipeline: wgpu::RenderPipeline,
    pub half_width: u32,
    pub half_height: u32,
    pub exposure_scale: f32,
    pub surface_is_srgb: bool,
    pub white_balance_gains: [f32; 3],
    pub _frame_target_reservation: R,
}

impl<R> BloomChain<R> {
    /// HDR scene target extent in pixels.
    pub fn hdr_extent(&self) -> (u32, u32) {
        (self.hdr_texture.width(), self.hdr_texture.height())
    }

    /// Half-resolution Kawase ping-pong extent.
    pub fn blur_extent(&self) -> (u32, u32) {
        (self.half_width, self.half_height)
    }

    /// Keep texture handles alive for resize validation / VRAM accounting.
    pub fn texture_handles(&self) -> (&wgpu::Texture, &wgpu::Texture, &wgpu::Texture) {
        (&self.hdr_texture, &self.blur_a, &self.blur_b)
    }
}

/// Probes the HDR format (`Rgba16Float` render attachment + texture binding) on both native and `wasm32` WebGPU adapters.
pub fn probe_hdr_format(device: &wgpu::Device) -> bool {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("portal-hdr-probe"),
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default());
    true
}

/// HDR color-target state.
pub fn hdr_color_target_state() -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format: HDR_FORMAT,
        blend: Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        write_mask: wgpu::ColorWrites::ALL,
    }
}

/// Bloom VRAM bytes.
pub fn bloom_vram_bytes(width: u32, height: u32) -> Option<u64> {
    let w = width.max(1) as u64;
    let h = height.max(1) as u64;
    let hdr = w.checked_mul(h)?.checked_mul(8)?;
    let half_w = (w / 2).max(1);
    let half_h = (h / 2).max(1);
    let half = half_w.checked_mul(half_h)?.checked_mul(8)?.checked_mul(2)?;
    hdr.checked_add(half)?
        .checked_add(8)? // one 1x1 dummy target
        .checked_add(std::mem::size_of::<BloomUniformBlock>() as u64)
}

/// Creates a float target.
pub fn create_float_target(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Bloom bind layout.
pub fn bloom_bind_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("portal-bloom-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<BloomUniformBlock>() as u64,
                    ),
                },
                count: None,
            },
        ],
    })
}

/// Makes a bloom bind group.
pub fn make_bloom_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    tex_a: &wgpu::TextureView,
    tex_b: &wgpu::TextureView,
    uniform_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("portal-bloom-bind"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(tex_a),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(tex_b),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform_buf.as_entire_binding(),
            },
        ],
    })
}

/// Creates the bloom chain.
pub fn create_bloom_chain<R>(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    surface_format: wgpu::TextureFormat,
    exposure_scale: f32,
    reserve: impl FnOnce(u64) -> Option<R>,
) -> Option<BloomChain<R>> {
    let frame_target_reservation = reserve(bloom_vram_bytes(width, height)?)?;

    let half_width = (width / 2).max(1);
    let half_height = (height / 2).max(1);
    let (hdr_texture, hdr_view) = create_float_target(device, "portal-hdr", width, height);
    let (blur_a, blur_a_view) =
        create_float_target(device, "portal-bloom-a", half_width, half_height);
    let (blur_b, blur_b_view) =
        create_float_target(device, "portal-bloom-b", half_width, half_height);
    let (dummy, dummy_view) = create_float_target(device, "portal-bloom-dummy", 1, 1);

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("portal-bloom-sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("portal-bloom-uniforms"),
        contents: bytemuck::bytes_of(&BloomUniformBlock {
            bloom: BloomParamsGpu {
                threshold: BLOOM_THRESHOLD,
                intensity: BLOOM_INTENSITY,
                offset: 1.0,
                _pad: 0.0,
            },
            composite: CompositeParamsGpu {
                exposure: BLOOM_EXPOSURE,
                bloom_strength: BLOOM_STRENGTH,
                surface_is_srgb: if surface_format.has_srgb_suffix() {
                    1.0
                } else {
                    0.0
                },
                _pad1: 0.0,
                white_balance_gains: [1.0; 3],
                _pad2: 0.0,
            },
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_layout = bloom_bind_layout(device);
    let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("portal-bloom"),
        source: wgpu::ShaderSource::Wgsl(BLOOM_WGSL.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("portal-bloom-pipeline-layout"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });

    let extract_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("portal-bloom-extract"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &bloom_shader,
            entry_point: Some("extract_vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &bloom_shader,
            entry_point: Some("extract_fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let kawase_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("portal-bloom-kawase"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &bloom_shader,
            entry_point: Some("kawase_vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &bloom_shader,
            entry_point: Some("kawase_fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("portal-bloom-composite"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &bloom_shader,
            entry_point: Some("composite_vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &bloom_shader,
            entry_point: Some("composite_fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let _ = dummy;

    Some(BloomChain {
        hdr_texture,
        hdr_view,
        blur_a,
        blur_a_view,
        blur_b,
        blur_b_view,
        dummy_view,
        sampler,
        uniform_buf,
        bind_layout,
        extract_pipeline,
        kawase_pipeline,
        composite_pipeline,
        half_width,
        half_height,
        exposure_scale: exposure_scale,
        surface_is_srgb: surface_format.has_srgb_suffix(),
        white_balance_gains: [1.0; 3],
        _frame_target_reservation: frame_target_reservation,
    })
}

/// Writes the bloom uniform.
pub fn write_bloom_uniform(
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    offset: f32,
    exposure_scale: f32,
    surface_is_srgb: bool,
    white_balance_gains: [f32; 3],
) {
    let block = BloomUniformBlock {
        bloom: BloomParamsGpu {
            threshold: BLOOM_THRESHOLD,
            intensity: BLOOM_INTENSITY,
            offset,
            _pad: 0.0,
        },
        composite: CompositeParamsGpu {
            exposure: exposure_scale,
            bloom_strength: BLOOM_STRENGTH,
            surface_is_srgb: if surface_is_srgb { 1.0 } else { 0.0 },
            _pad1: 0.0,
            white_balance_gains,
            _pad2: 0.0,
        },
    };
    queue.write_buffer(buf, 0, bytemuck::bytes_of(&block));
}

mod record;
pub use record::run_bloom_passes;

#[cfg(test)]
mod tests;

#[cfg(test)]
fn hdr_composite_available(a: bool, b: bool, c: bool, d: bool) -> bool {
    a && b && c && d
}
