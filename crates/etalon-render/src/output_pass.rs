//! Final scene-linear-to-SDR pass used when the HDR bloom chain is unavailable.
use etalon::util::DeviceExt;
pub const OUTPUT_WGSL: &str = concat!(
    include_str!("shaders/output_transform.wgsl"),
    include_str!("shaders/output_composite.wgsl")
);

const OUTPUT_PARAMS_BYTES: u64 = 32;

fn output_target_vram_bytes(width: u32, height: u32, bytes_per_pixel: u32) -> Option<u64> {
    u64::from(width.max(1))
        .checked_mul(u64::from(height.max(1)))?
        .checked_mul(u64::from(bytes_per_pixel))?
        .checked_add(OUTPUT_PARAMS_BYTES)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OutputParams {
    exposure_scale: f32,
    surface_is_srgb: f32,
    _padding: [f32; 2],
    white_balance_gains: [f32; 3],
    _padding2: f32,
}

/// Budgeted scene target and final SDR transform for the portable non-bloom path.
pub struct OutputChain<R> {
    _scene_texture: etalon::Texture,
    scene_view: etalon::TextureView,
    _sampler: etalon::Sampler,
    bind_group: etalon::BindGroup,
    uniform_buf: etalon::Buffer,
    composite_pipeline: etalon::RenderPipeline,
    width: u32,
    height: u32,
    exposure_scale: f32,
    surface_is_srgb: bool,
    hdr_scene: bool,
    white_balance_gains: [f32; 3],
    _frame_target_reservation: R,
}

impl<R> OutputChain<R> {
    /// Tries to construct.
    pub fn try_new(
        device: &etalon::Device,
        width: u32,
        height: u32,
        scene_format: etalon::TextureFormat,
        surface_format: etalon::TextureFormat,
        exposure_scale: f32,
        hdr_scene: bool,
        white_balance_gains: [f32; 3],
        reserve: impl FnOnce(u64) -> Option<R>,
    ) -> Option<Self> {
        let width = width.max(1);
        let height = height.max(1);
        let reservation_bytes =
            output_target_vram_bytes(width, height, scene_format.block_copy_size(None)?)?;
        let reservation = reserve(reservation_bytes)?;

        let scene_texture = device.create_texture(&etalon::TextureDescriptor {
            label: Some("qualia-sdr-scene-linear"),
            size: etalon::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: scene_format,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let scene_view = scene_texture.create_view(&etalon::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&etalon::SamplerDescriptor {
            label: Some("qualia-sdr-output-sampler"),
            mag_filter: etalon::FilterMode::Linear,
            min_filter: etalon::FilterMode::Linear,
            ..Default::default()
        });
        let params = OutputParams {
            exposure_scale,
            surface_is_srgb: if surface_format.has_srgb_suffix() {
                1.0
            } else {
                0.0
            },
            _padding: [0.0; 2],
            white_balance_gains,
            _padding2: 0.0,
        };
        let uniform_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("qualia-sdr-output-params"),
            contents: bytemuck::bytes_of(&params),
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
        });
        let bind_group_layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("qualia-sdr-output-layout"),
            entries: &[
                etalon::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: etalon::ShaderStages::FRAGMENT,
                    ty: etalon::BindingType::Sampler(etalon::SamplerBindingType::Filtering),
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: etalon::ShaderStages::FRAGMENT,
                    ty: etalon::BindingType::Texture {
                        sample_type: etalon::TextureSampleType::Float { filterable: true },
                        view_dimension: etalon::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: etalon::ShaderStages::FRAGMENT,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(OUTPUT_PARAMS_BYTES),
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("qualia-sdr-output-bind"),
            layout: &bind_group_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: etalon::BindingResource::Sampler(&sampler),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: etalon::BindingResource::TextureView(&scene_view),
                },
                etalon::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("qualia-sdr-output"),
            source: etalon::ShaderSource::Wgsl(OUTPUT_WGSL.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("qualia-sdr-output-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let composite_pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("qualia-sdr-output-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: etalon::VertexState {
                module: &shader,
                entry_point: Some("output_vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(etalon::FragmentState {
                module: &shader,
                entry_point: Some("output_fs"),
                compilation_options: Default::default(),
                targets: &[Some(etalon::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: etalon::ColorWrites::ALL,
                })],
            }),
            primitive: etalon::PrimitiveState::default(),
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Some(Self {
            _scene_texture: scene_texture,
            scene_view,
            _sampler: sampler,
            bind_group,
            uniform_buf,
            composite_pipeline,
            width,
            height,
            exposure_scale,
            surface_is_srgb: surface_format.has_srgb_suffix(),
            hdr_scene,
            white_balance_gains,
            _frame_target_reservation: reservation,
        })
    }

    /// `scene_view`.
    pub fn scene_view(&self) -> &etalon::TextureView {
        &self.scene_view
    }

    /// `extent`.
    pub fn extent(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// `uses_hdr_scene`.
    pub fn uses_hdr_scene(&self) -> bool {
        self.hdr_scene
    }

    /// Sets exposure.
    pub fn set_exposure(&mut self, queue: &etalon::Queue, exposure_scale: f32) -> bool {
        if !exposure_scale.is_finite() || exposure_scale <= 0.0 {
            return false;
        }
        self.exposure_scale = exposure_scale;
        let params = OutputParams {
            exposure_scale,
            surface_is_srgb: if self.surface_is_srgb { 1.0 } else { 0.0 },
            _padding: [0.0; 2],
            white_balance_gains: self.white_balance_gains,
            _padding2: 0.0,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&params));
        true
    }

    /// Sets white balance.
    pub fn set_white_balance(&mut self, queue: &etalon::Queue, white_balance_gains: [f32; 3]) {
        self.white_balance_gains = white_balance_gains;
        let params = OutputParams {
            exposure_scale: self.exposure_scale,
            surface_is_srgb: if self.surface_is_srgb { 1.0 } else { 0.0 },
            _padding: [0.0; 2],
            white_balance_gains,
            _padding2: 0.0,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&params));
    }

    /// `composite`.
    pub fn composite(&self, encoder: &mut etalon::CommandEncoder, target: &etalon::TextureView) {
        let mut pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: Some("qualia-sdr-output"),
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: etalon::Operations {
                    load: etalon::LoadOp::Clear(etalon::Color::BLACK),
                    store: etalon::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
#[path = "output_pass_tests.rs"]
mod tests;
