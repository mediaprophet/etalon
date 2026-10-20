use bytemuck::{Pod, Zeroable};
use std::f32::consts;
use etalon::{util::DeviceExt, AstcBlock, AstcChannel};

const IMAGE_SIZE: u32 = 256;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
}

struct Entity {
    vertex_count: u32,
    vertex_buf: etalon::Buffer,
}

// Note: we use the Y=up coordinate space in this example.
struct Camera {
    screen_size: (u32, u32),
    angle_y: f32,
    angle_xz: f32,
    dist: f32,
}

const MODEL_CENTER_Y: f32 = 2.0;

impl Camera {
    fn to_uniform_data(&self) -> [f32; 16 * 3 + 4] {
        let aspect = self.screen_size.0 as f32 / self.screen_size.1 as f32;
        let proj =
            glam::camera::rh::proj::directx::perspective(consts::FRAC_PI_4, aspect, 1.0, 50.0);
        let cam_pos = glam::Vec3::new(
            self.angle_xz.cos() * self.angle_y.sin() * self.dist,
            self.angle_xz.sin() * self.dist + MODEL_CENTER_Y,
            self.angle_xz.cos() * self.angle_y.cos() * self.dist,
        );
        let view = glam::camera::rh::view::look_at_mat4(
            cam_pos,
            glam::Vec3::new(0f32, MODEL_CENTER_Y, 0.0),
            glam::Vec3::Y,
        );
        let proj_inv = proj.inverse();

        let mut raw = [0f32; 16 * 3 + 4];
        raw[..16].copy_from_slice(&AsRef::<[f32; 16]>::as_ref(&proj)[..]);
        raw[16..32].copy_from_slice(&AsRef::<[f32; 16]>::as_ref(&proj_inv)[..]);
        raw[32..48].copy_from_slice(&AsRef::<[f32; 16]>::as_ref(&view)[..]);
        raw[48..51].copy_from_slice(AsRef::<[f32; 3]>::as_ref(&cam_pos));
        raw[51] = 1.0;
        raw
    }
}

pub struct Example {
    camera: Camera,
    sky_pipeline: etalon::RenderPipeline,
    entity_pipeline: etalon::RenderPipeline,
    bind_group: etalon::BindGroup,
    uniform_buf: etalon::Buffer,
    entities: Vec<Entity>,
    depth_view: etalon::TextureView,
    staging_belt: etalon::util::StagingBelt,
}

impl Example {
    const DEPTH_FORMAT: etalon::TextureFormat = etalon::TextureFormat::Depth24Plus;

    fn create_depth_texture(
        config: &etalon::SurfaceConfiguration,
        device: &etalon::Device,
    ) -> etalon::TextureView {
        let depth_texture = device.create_texture(&etalon::TextureDescriptor {
            size: etalon::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT
                | etalon::TextureUsages::TRANSIENT_ATTACHMENT,
            label: None,
            view_formats: &[],
        });

        depth_texture.create_view(&etalon::TextureViewDescriptor::default())
    }
}

impl crate::framework::Example for Example {
    fn optional_features() -> etalon::Features {
        etalon::Features::TEXTURE_COMPRESSION_ASTC
            | etalon::Features::TEXTURE_COMPRESSION_ETC2
            | etalon::Features::TEXTURE_COMPRESSION_BC
    }

    fn init(
        config: &etalon::SurfaceConfiguration,
        _adapter: &etalon::Adapter,
        device: &etalon::Device,
        queue: &etalon::Queue,
    ) -> Self {
        let mut entities = Vec::new();
        {
            let source = include_bytes!("models/rustacean-3d.obj");
            let data = obj::ObjData::load_buf(&source[..]).unwrap();
            let mut vertices = Vec::new();
            for object in data.objects {
                for group in object.groups {
                    vertices.clear();
                    for poly in group.polys {
                        for end_index in 2..poly.0.len() {
                            for &index in &[0, end_index - 1, end_index] {
                                let obj::IndexTuple(position_id, _texture_id, normal_id) =
                                    poly.0[index];
                                let [x, y, z] = data.position[position_id];
                                vertices.push(Vertex {
                                    pos: [y, z, x], // model is rotated to face down, so need to rotate it
                                    normal: data.normal[normal_id.unwrap()],
                                })
                            }
                        }
                    }
                    let vertex_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
                        label: Some("Vertex"),
                        contents: bytemuck::cast_slice(&vertices),
                        usage: etalon::BufferUsages::VERTEX,
                    });
                    entities.push(Entity {
                        vertex_count: vertices.len() as u32,
                        vertex_buf,
                    });
                }
            }
        }

        let bind_group_layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                etalon::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: etalon::ShaderStages::VERTEX | etalon::ShaderStages::FRAGMENT,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: etalon::ShaderStages::FRAGMENT,
                    ty: etalon::BindingType::Texture {
                        sample_type: etalon::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: etalon::TextureViewDimension::Cube,
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: etalon::ShaderStages::FRAGMENT,
                    ty: etalon::BindingType::Sampler(etalon::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create the render pipeline
        let shader = device.create_shader_module(etalon::include_wgsl!("shader.wgsl"));

        let camera = Camera {
            screen_size: (config.width, config.height),
            angle_xz: 0.2,
            angle_y: 0.2,
            dist: 20.0,
        };
        let raw_uniforms = camera.to_uniform_data();
        let uniform_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Buffer"),
            contents: bytemuck::cast_slice(&raw_uniforms),
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
        });

        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Create the render pipelines
        let sky_pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("Sky"),
            layout: Some(&pipeline_layout),
            vertex: etalon::VertexState {
                module: &shader,
                entry_point: Some("vs_sky"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(etalon::FragmentState {
                module: &shader,
                entry_point: Some("fs_sky"),
                compilation_options: Default::default(),
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: etalon::PrimitiveState {
                front_face: etalon::FrontFace::Cw,
                ..Default::default()
            },
            depth_stencil: Some(etalon::DepthStencilState {
                format: Self::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(etalon::CompareFunction::LessEqual),
                stencil: etalon::StencilState::default(),
                bias: etalon::DepthBiasState::default(),
            }),
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let entity_pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("Entity"),
            layout: Some(&pipeline_layout),
            vertex: etalon::VertexState {
                module: &shader,
                entry_point: Some("vs_entity"),
                compilation_options: Default::default(),
                buffers: &[Some(etalon::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as etalon::BufferAddress,
                    step_mode: etalon::VertexStepMode::Vertex,
                    attributes: &etalon::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                })],
            },
            fragment: Some(etalon::FragmentState {
                module: &shader,
                entry_point: Some("fs_entity"),
                compilation_options: Default::default(),
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: etalon::PrimitiveState {
                front_face: etalon::FrontFace::Cw,
                ..Default::default()
            },
            depth_stencil: Some(etalon::DepthStencilState {
                format: Self::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(etalon::CompareFunction::LessEqual),
                stencil: etalon::StencilState::default(),
                bias: etalon::DepthBiasState::default(),
            }),
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&etalon::SamplerDescriptor {
            label: None,
            address_mode_u: etalon::AddressMode::ClampToEdge,
            address_mode_v: etalon::AddressMode::ClampToEdge,
            address_mode_w: etalon::AddressMode::ClampToEdge,
            mag_filter: etalon::FilterMode::Linear,
            min_filter: etalon::FilterMode::Linear,
            mipmap_filter: etalon::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let device_features = device.features();

        let skybox_format = if device_features.contains(etalon::Features::TEXTURE_COMPRESSION_ASTC) {
            log::info!("Using astc");
            etalon::TextureFormat::Astc {
                block: AstcBlock::B4x4,
                channel: AstcChannel::UnormSrgb,
            }
        } else if device_features.contains(etalon::Features::TEXTURE_COMPRESSION_ETC2) {
            log::info!("Using etc2");
            etalon::TextureFormat::Etc2Rgb8A1UnormSrgb
        } else if device_features.contains(etalon::Features::TEXTURE_COMPRESSION_BC) {
            log::info!("Using bc7");
            etalon::TextureFormat::Bc7RgbaUnormSrgb
        } else {
            log::info!("Using rgba8");
            etalon::TextureFormat::Rgba8UnormSrgb
        };

        let size = etalon::Extent3d {
            width: IMAGE_SIZE,
            height: IMAGE_SIZE,
            depth_or_array_layers: 6,
        };

        let layer_size = etalon::Extent3d {
            depth_or_array_layers: 1,
            ..size
        };
        let max_mips = layer_size.max_mips(etalon::TextureDimension::D2);

        log::debug!(
            "Copying {skybox_format:?} skybox images of size {IMAGE_SIZE}, {IMAGE_SIZE}, 6 with {max_mips} mips to gpu",
        );

        let bytes = match skybox_format {
            etalon::TextureFormat::Astc {
                block: AstcBlock::B4x4,
                channel: AstcChannel::UnormSrgb,
            } => &include_bytes!("images/astc.ktx2")[..],
            etalon::TextureFormat::Etc2Rgb8A1UnormSrgb => &include_bytes!("images/etc2.ktx2")[..],
            etalon::TextureFormat::Bc7RgbaUnormSrgb => &include_bytes!("images/bc7.ktx2")[..],
            etalon::TextureFormat::Rgba8UnormSrgb => &include_bytes!("images/rgba8.ktx2")[..],
            _ => unreachable!(),
        };

        let reader = ktx2::Reader::new(bytes).unwrap();
        let header = reader.header();

        let mut image = Vec::with_capacity(reader.data().len());
        for level in reader.levels() {
            image.extend_from_slice(level.data);
        }

        let texture = device.create_texture_with_data(
            queue,
            &etalon::TextureDescriptor {
                size,
                mip_level_count: header.level_count,
                sample_count: 1,
                dimension: etalon::TextureDimension::D2,
                format: skybox_format,
                usage: etalon::TextureUsages::TEXTURE_BINDING | etalon::TextureUsages::COPY_DST,
                label: None,
                view_formats: &[],
            },
            // KTX2 stores mip levels in mip major order.
            etalon::util::TextureDataOrder::MipMajor,
            &image,
        );

        let texture_view = texture.create_view(&etalon::TextureViewDescriptor {
            label: None,
            dimension: Some(etalon::TextureViewDimension::Cube),
            ..etalon::TextureViewDescriptor::default()
        });
        let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: etalon::BindingResource::TextureView(&texture_view),
                },
                etalon::BindGroupEntry {
                    binding: 2,
                    resource: etalon::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });

        let depth_view = Self::create_depth_texture(config, device);

        Example {
            camera,
            sky_pipeline,
            entity_pipeline,
            bind_group,
            uniform_buf,
            entities,
            depth_view,
            staging_belt: etalon::util::StagingBelt::new(device.clone(), 0x100),
        }
    }

    #[expect(clippy::single_match)]
    fn update(&mut self, event: winit::event::WindowEvent) {
        match event {
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                let norm_x = position.x as f32 / self.camera.screen_size.0 as f32 - 0.5;
                let norm_y = position.y as f32 / self.camera.screen_size.1 as f32 - 0.5;
                self.camera.angle_y = norm_x * 5.0;
                self.camera.angle_xz = norm_y;
            }
            _ => {}
        }
    }

    fn resize(
        &mut self,
        config: &etalon::SurfaceConfiguration,
        device: &etalon::Device,
        _queue: &etalon::Queue,
    ) {
        self.depth_view = Self::create_depth_texture(config, device);
        self.camera.screen_size = (config.width, config.height);
    }

    fn render(&mut self, view: &etalon::TextureView, device: &etalon::Device, queue: &etalon::Queue) {
        let mut encoder =
            device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });

        // update rotation
        let raw_uniforms = self.camera.to_uniform_data();
        self.staging_belt
            .write_buffer(
                &mut encoder,
                &self.uniform_buf,
                0,
                etalon::BufferSize::new((raw_uniforms.len() * 4) as etalon::BufferAddress).unwrap(),
            )
            .copy_from_slice(bytemuck::cast_slice(&raw_uniforms));

        self.staging_belt.finish();

        {
            let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(etalon::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(etalon::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: etalon::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(etalon::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(etalon::Operations {
                        load: etalon::LoadOp::Clear(1.0),
                        store: etalon::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_pipeline(&self.entity_pipeline);

            for entity in self.entities.iter() {
                rpass.set_vertex_buffer(0, entity.vertex_buf.slice(..));
                rpass.draw(0..entity.vertex_count, 0..1);
            }

            rpass.set_pipeline(&self.sky_pipeline);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        self.staging_belt.recall();
    }
}

pub fn main() {
    crate::framework::run::<Example>("skybox");
}

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "skybox",
    image_path: "/examples/features/src/skybox/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::default(),
    base_test_parameters: etalon_test::TestParameters::default().expect_fail(
        etalon_test::FailureCase::backend_adapter(etalon::Backends::GL, "ANGLE"),
    ),
    comparisons: &[etalon_test::ComparisonType::Mean(0.02)],
    _phantom: std::marker::PhantomData::<Example>,
};

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST_BCN: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "skybox-bc7",
    image_path: "/examples/features/src/skybox/screenshot_bc7.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::TEXTURE_COMPRESSION_BC,
    base_test_parameters: etalon_test::TestParameters::default(),
    comparisons: &[etalon_test::ComparisonType::Mean(0.02)],
    _phantom: std::marker::PhantomData::<Example>,
};

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST_ETC2: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "skybox-etc2",
    image_path: "/examples/features/src/skybox/screenshot_etc2.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::TEXTURE_COMPRESSION_ETC2,
    base_test_parameters: etalon_test::TestParameters::default(),
    comparisons: &[etalon_test::ComparisonType::Mean(0.016)], // Bounded by Apple A9
    _phantom: std::marker::PhantomData::<Example>,
};

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST_ASTC: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "skybox-astc",
    image_path: "/examples/features/src/skybox/screenshot_astc.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::TEXTURE_COMPRESSION_ASTC,
    base_test_parameters: etalon_test::TestParameters::default(),
    comparisons: &[etalon_test::ComparisonType::Mean(0.017)], // Bounded by Apple A9
    _phantom: std::marker::PhantomData::<Example>,
};
