//! Half-resolution, depth/normal screen-space ambient visibility.
//!
//! This portable baseline borrows CACAO's adaptive-resolution strategy but keeps a bounded
//! single-ring kernel suitable for WebGPU/WASM. It is an indirect-light approximation, never a
//! replacement for sun shadows. All targets are explicitly budgeted and optional.

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AoUniform {
    /// Half-resolution width, height, and their reciprocals.
    pub viewport: [f32; 4],
    pub eye: [f32; 4],
    /// XYZ forward basis; W is internal AODEPTH_LOG1P_BA16 encoding version.
    pub camera_forward: [f32; 4],
    pub camera_right: [f32; 4],
    pub camera_up: [f32; 4],
    /// World-space radius, strength, world-space bias, enabled.
    pub params: [f32; 4],
    /// Far distance, log2(1 + depth) per BA16 code, tan(vertical FOV / 2), aspect.
    pub depth: [f32; 4],
}

pub struct AoTargets<R> {
    _depth_texture: etalon::Texture,
    depth_view: etalon::TextureView,
    _normal_texture: etalon::Texture,
    normal_view: etalon::TextureView,
    _ao_texture: etalon::Texture,
    ao_view: etalon::TextureView,
    _reservation: R,
    depth_prepass: etalon::RenderPipeline,
    ao_pipeline: etalon::RenderPipeline,
    ao_generate_bind: etalon::BindGroup,
    width: u32,
    height: u32,
}

impl<R> AoTargets<R> {
    /// Allocates AO/normal targets at `extent`; `Err` on device failure.
    pub fn try_new(
        device: &etalon::Device,
        width: u32,
        height: u32,
        camera_layout: &etalon::BindGroupLayout,
        model_layout: &etalon::BindGroupLayout,
        instance_layout: &etalon::BindGroupLayout,
        material_layout: &etalon::BindGroupLayout,
        _texture_layout: &etalon::BindGroupLayout,
        uniform_buf: &etalon::Buffer,
        copy_source: bool,
        reserve: impl FnOnce(u64) -> Option<R>,
    ) -> Option<Self> {
        let bytes = target_bytes(width, height)?; // Depth32Float + packed RGBA8 surface + R8 AO.
        let (width, height) = half_extent(width, height);
        let reservation = reserve(bytes)?;

        let depth_texture = device.create_texture(&etalon::TextureDescriptor {
            label: Some("qualia-ao-depth-half"),
            size: etalon::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Depth32Float,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&etalon::TextureViewDescriptor::default());
        let normal_usage = etalon::TextureUsages::RENDER_ATTACHMENT
            | etalon::TextureUsages::TEXTURE_BINDING
            | if copy_source {
                etalon::TextureUsages::COPY_SRC
            } else {
                etalon::TextureUsages::empty()
            };
        let normal_texture = device.create_texture(&etalon::TextureDescriptor {
            label: Some("qualia-ao-normal-log1p-ba16-v1-half"),
            size: etalon::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rgba8Unorm,
            usage: normal_usage,
            view_formats: &[],
        });
        let normal_view = normal_texture.create_view(&etalon::TextureViewDescriptor::default());
        let ao_texture = device.create_texture(&etalon::TextureDescriptor {
            label: Some("qualia-ao-visibility-half"),
            size: etalon::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::R8Unorm,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ao_view = ao_texture.create_view(&etalon::TextureViewDescriptor::default());

        let depth_prepass_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("qualia-ao-depth-normal-pipeline-layout"),
            bind_group_layouts: &[
                Some(camera_layout),
                Some(model_layout),
                Some(material_layout),
                Some(instance_layout),
            ],
            immediate_size: 0,
        });
        let prepass_shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("qualia-ao-depth-normal-wgsl"),
            source: etalon::ShaderSource::Wgsl(include_str!("../shaders/ao_prepass.wgsl").into()),
        });
        let position_layout = etalon::VertexBufferLayout {
            array_stride: 12,
            step_mode: etalon::VertexStepMode::Vertex,
            attributes: &[etalon::VertexAttribute {
                format: etalon::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            }],
        };
        let normal_layout = etalon::VertexBufferLayout {
            array_stride: 12,
            step_mode: etalon::VertexStepMode::Vertex,
            attributes: &[etalon::VertexAttribute {
                format: etalon::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 1,
            }],
        };
        let color_layout = etalon::VertexBufferLayout {
            array_stride: 16,
            step_mode: etalon::VertexStepMode::Vertex,
            attributes: &[etalon::VertexAttribute {
                format: etalon::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 2,
            }],
        };
        let uv_layout = etalon::VertexBufferLayout {
            array_stride: 8,
            step_mode: etalon::VertexStepMode::Vertex,
            attributes: &[etalon::VertexAttribute {
                format: etalon::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 3,
            }],
        };
        let depth_prepass = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("qualia-ao-depth-normal-prepass"),
            layout: Some(&depth_prepass_layout),
            vertex: etalon::VertexState {
                module: &prepass_shader,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &[
                    Some(position_layout),
                    Some(normal_layout),
                    Some(color_layout),
                    Some(uv_layout),
                ],
            },
            fragment: Some(etalon::FragmentState {
                module: &prepass_shader,
                entry_point: Some("fragment_main"),
                compilation_options: Default::default(),
                targets: &[Some(etalon::ColorTargetState {
                    format: etalon::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: etalon::ColorWrites::ALL,
                })],
            }),
            primitive: etalon::PrimitiveState {
                topology: etalon::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(etalon::DepthStencilState {
                format: etalon::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(etalon::CompareFunction::Less),
                stencil: etalon::StencilState::default(),
                bias: etalon::DepthBiasState::default(),
            }),
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let depth_normal_layout =
            device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                label: Some("qualia-ao-input-layout"),
                entries: &[
                    etalon::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: etalon::ShaderStages::FRAGMENT,
                        ty: etalon::BindingType::Texture {
                            sample_type: etalon::TextureSampleType::Float { filterable: false },
                            view_dimension: etalon::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    etalon::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: etalon::ShaderStages::FRAGMENT,
                        ty: etalon::BindingType::Buffer {
                            ty: etalon::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(
                                std::mem::size_of::<AoUniform>() as u64,
                            ),
                        },
                        count: None,
                    },
                ],
            });
        let ao_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("qualia-ao-pipeline-layout"),
            bind_group_layouts: &[Some(&depth_normal_layout)],
            immediate_size: 0,
        });
        let ao_shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("qualia-ao-generate-wgsl"),
            source: etalon::ShaderSource::Wgsl(include_str!("../shaders/ao_generate.wgsl").into()),
        });
        let ao_pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("qualia-ao-half-resolution"),
            layout: Some(&ao_layout),
            vertex: etalon::VertexState {
                module: &ao_shader,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(etalon::FragmentState {
                module: &ao_shader,
                entry_point: Some("fragment_main"),
                compilation_options: Default::default(),
                targets: &[Some(etalon::ColorTargetState {
                    format: etalon::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: etalon::ColorWrites::ALL,
                })],
            }),
            primitive: etalon::PrimitiveState {
                topology: etalon::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let ao_generate_bind = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("qualia-ao-generate-inputs"),
            layout: &depth_normal_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: etalon::BindingResource::TextureView(&normal_view),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        Some(Self {
            _depth_texture: depth_texture,
            depth_view,
            _normal_texture: normal_texture,
            normal_view,
            _ao_texture: ao_texture,
            ao_view,
            _reservation: reservation,
            depth_prepass,
            ao_pipeline,
            ao_generate_bind,
            width,
            height,
        })
    }

    /// Records the AO pass into `encoder`.
    pub fn record(
        &self,
        encoder: &mut etalon::CommandEncoder,
        draw: impl FnOnce(&mut etalon::RenderPass<'_>),
    ) {
        let mut prepass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: Some("qualia-ao-depth-normal-prepass"),
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                view: &self.normal_view,
                depth_slice: None,
                resolve_target: None,
                ops: etalon::Operations {
                    load: etalon::LoadOp::Clear(etalon::Color {
                        r: 0.5,
                        g: 0.5,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: etalon::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(etalon::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(etalon::Operations {
                    load: etalon::LoadOp::Clear(1.0),
                    store: etalon::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });
        prepass.set_pipeline(&self.depth_prepass);
        draw(&mut prepass);
        drop(prepass);
        let mut ao = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: Some("qualia-ao-evaluate-half-resolution"),
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                view: &self.ao_view,
                depth_slice: None,
                resolve_target: None,
                ops: etalon::Operations {
                    load: etalon::LoadOp::Clear(etalon::Color::WHITE),
                    store: etalon::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });
        ao.set_pipeline(&self.ao_pipeline);
        ao.set_bind_group(0, &self.ao_generate_bind, &[]);
        ao.draw(0..3, 0..1);
    }

    /// AO texture view.
    pub fn ao_view(&self) -> &etalon::TextureView {
        &self.ao_view
    }
    /// Internal transient RG8 oct normal / BA16 AODEPTH_LOG1P_BA16 depth view.
    pub fn normal_view(&self) -> &etalon::TextureView {
        &self.normal_view
    }
    /// Current target extent.
    pub fn extent(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Test-only normal texture.
    pub fn normal_texture_for_test(&self) -> &etalon::Texture {
        &self._normal_texture
    }
}

pub fn half_extent(width: u32, height: u32) -> (u32, u32) {
    (width.div_ceil(2).max(1), height.div_ceil(2).max(1))
}

/// Target byte size.
pub fn target_bytes(width: u32, height: u32) -> Option<u64> {
    let (width, height) = half_extent(width, height);
    target_bytes_at_half(width, height)
}

/// Target bytes at half res.
pub fn target_bytes_at_half(width: u32, height: u32) -> Option<u64> {
    u64::from(width)
        .checked_mul(u64::from(height))?
        .checked_mul(9)
}
