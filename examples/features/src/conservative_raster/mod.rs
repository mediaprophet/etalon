const RENDER_TARGET_FORMAT: etalon::TextureFormat = etalon::TextureFormat::Rgba8UnormSrgb;

struct Example {
    low_res_target: etalon::TextureView,
    bind_group_upscale: etalon::BindGroup,

    pipeline_triangle_conservative: etalon::RenderPipeline,
    pipeline_triangle_regular: etalon::RenderPipeline,
    pipeline_upscale: etalon::RenderPipeline,
    pipeline_lines: Option<etalon::RenderPipeline>,
    bind_group_layout_upscale: etalon::BindGroupLayout,
}

impl Example {
    fn create_low_res_target(
        config: &etalon::SurfaceConfiguration,
        device: &etalon::Device,
        bind_group_layout_upscale: &etalon::BindGroupLayout,
    ) -> (etalon::TextureView, etalon::BindGroup) {
        let texture_view = device
            .create_texture(&etalon::TextureDescriptor {
                label: Some("Low Resolution Target"),
                size: etalon::Extent3d {
                    width: (config.width / 16).max(1),
                    height: (config.height / 16).max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: etalon::TextureDimension::D2,
                format: RENDER_TARGET_FORMAT,
                usage: etalon::TextureUsages::TEXTURE_BINDING
                    | etalon::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default());

        let sampler = device.create_sampler(&etalon::SamplerDescriptor {
            label: Some("Nearest Neighbor Sampler"),
            mag_filter: etalon::FilterMode::Nearest,
            min_filter: etalon::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("upscale bind group"),
            layout: bind_group_layout_upscale,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: etalon::BindingResource::TextureView(&texture_view),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: etalon::BindingResource::Sampler(&sampler),
                },
            ],
        });

        (texture_view, bind_group)
    }
}

impl crate::framework::Example for Example {
    fn required_features() -> etalon::Features {
        etalon::Features::CONSERVATIVE_RASTERIZATION
    }
    fn optional_features() -> etalon::Features {
        etalon::Features::POLYGON_MODE_LINE
    }
    fn init(
        config: &etalon::SurfaceConfiguration,
        _adapter: &etalon::Adapter,
        device: &etalon::Device,
        _queue: &etalon::Queue,
    ) -> Self {
        let pipeline_layout_empty =
            device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[],
                immediate_size: 0,
            });

        let shader_triangle_and_lines =
            device.create_shader_module(etalon::include_wgsl!("triangle_and_lines.wgsl"));

        let pipeline_triangle_conservative =
            device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
                label: Some("Conservative Rasterization"),
                layout: Some(&pipeline_layout_empty),
                vertex: etalon::VertexState {
                    module: &shader_triangle_and_lines,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(etalon::FragmentState {
                    module: &shader_triangle_and_lines,
                    entry_point: Some("fs_main_red"),
                    compilation_options: Default::default(),
                    targets: &[Some(RENDER_TARGET_FORMAT.into())],
                }),
                primitive: etalon::PrimitiveState {
                    conservative: true,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: etalon::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let pipeline_triangle_regular =
            device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
                label: Some("Regular Rasterization"),
                layout: Some(&pipeline_layout_empty),
                vertex: etalon::VertexState {
                    module: &shader_triangle_and_lines,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(etalon::FragmentState {
                    module: &shader_triangle_and_lines,
                    entry_point: Some("fs_main_blue"),
                    compilation_options: Default::default(),
                    targets: &[Some(RENDER_TARGET_FORMAT.into())],
                }),
                primitive: etalon::PrimitiveState::default(),
                depth_stencil: None,
                multisample: etalon::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let pipeline_lines = if device
            .features()
            .contains(etalon::Features::POLYGON_MODE_LINE)
        {
            Some(
                device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
                    label: Some("Lines"),
                    layout: Some(&pipeline_layout_empty),
                    vertex: etalon::VertexState {
                        module: &shader_triangle_and_lines,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: Some(etalon::FragmentState {
                        module: &shader_triangle_and_lines,
                        entry_point: Some("fs_main_white"),
                        compilation_options: Default::default(),
                        targets: &[Some(config.view_formats[0].into())],
                    }),
                    primitive: etalon::PrimitiveState {
                        polygon_mode: etalon::PolygonMode::Line,
                        topology: etalon::PrimitiveTopology::LineStrip,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: etalon::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                }),
            )
        } else {
            None
        };

        let (pipeline_upscale, bind_group_layout_upscale) = {
            let bind_group_layout =
                device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                    label: Some("upscale bindgroup"),
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
                            ty: etalon::BindingType::Sampler(etalon::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                    ],
                });

            let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
            let shader = device.create_shader_module(etalon::include_wgsl!("upscale.wgsl"));
            (
                device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
                    label: Some("Upscale"),
                    layout: Some(&pipeline_layout),
                    vertex: etalon::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: Some(etalon::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(config.view_formats[0].into())],
                    }),
                    primitive: etalon::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: etalon::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                }),
                bind_group_layout,
            )
        };

        let (low_res_target, bind_group_upscale) =
            Self::create_low_res_target(config, device, &bind_group_layout_upscale);

        Self {
            low_res_target,
            bind_group_upscale,

            pipeline_triangle_conservative,
            pipeline_triangle_regular,
            pipeline_upscale,
            pipeline_lines,
            bind_group_layout_upscale,
        }
    }

    fn resize(
        &mut self,
        config: &etalon::SurfaceConfiguration,
        device: &etalon::Device,
        _queue: &etalon::Queue,
    ) {
        let (low_res_target, bind_group_upscale) =
            Self::create_low_res_target(config, device, &self.bind_group_layout_upscale);
        self.low_res_target = low_res_target;
        self.bind_group_upscale = bind_group_upscale;
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {}

    fn render(&mut self, view: &etalon::TextureView, device: &etalon::Device, queue: &etalon::Queue) {
        let mut encoder = device.create_command_encoder(&etalon::CommandEncoderDescriptor {
            label: Some("primary"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                label: Some("low resolution"),
                color_attachments: &[Some(etalon::RenderPassColorAttachment {
                    view: &self.low_res_target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(etalon::Color::BLACK),
                        store: etalon::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&self.pipeline_triangle_conservative);
            rpass.draw(0..3, 0..1);
            rpass.set_pipeline(&self.pipeline_triangle_regular);
            rpass.draw(0..3, 0..1);
        }
        {
            let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                label: Some("full resolution"),
                color_attachments: &[Some(etalon::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(etalon::Color::BLACK),
                        store: etalon::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&self.pipeline_upscale);
            rpass.set_bind_group(0, &self.bind_group_upscale, &[]);
            rpass.draw(0..3, 0..1);

            if let Some(pipeline_lines) = &self.pipeline_lines {
                rpass.set_pipeline(pipeline_lines);
                rpass.draw(0..4, 0..1);
            }
        }

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("conservative-raster");
}

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "conservative-raster",
    image_path: "/examples/features/src/conservative_raster/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::default(),
    base_test_parameters: etalon_test::TestParameters::default(),
    comparisons: &[etalon_test::ComparisonType::Mean(0.0)],
    _phantom: std::marker::PhantomData::<Example>,
};
