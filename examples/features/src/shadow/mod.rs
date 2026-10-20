use std::{f32::consts, iter, ops::Range};

use bytemuck::{Pod, Zeroable};
use etalon::util::{align_to, DeviceExt};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [i8; 4],
    _normal: [i8; 4],
}

fn vertex(pos: [i8; 3], nor: [i8; 3]) -> Vertex {
    Vertex {
        _pos: [pos[0], pos[1], pos[2], 1],
        _normal: [nor[0], nor[1], nor[2], 0],
    }
}

fn create_cube() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // top (0, 0, 1)
        vertex([-1, -1, 1], [0, 0, 1]),
        vertex([1, -1, 1], [0, 0, 1]),
        vertex([1, 1, 1], [0, 0, 1]),
        vertex([-1, 1, 1], [0, 0, 1]),
        // bottom (0, 0, -1)
        vertex([-1, 1, -1], [0, 0, -1]),
        vertex([1, 1, -1], [0, 0, -1]),
        vertex([1, -1, -1], [0, 0, -1]),
        vertex([-1, -1, -1], [0, 0, -1]),
        // right (1, 0, 0)
        vertex([1, -1, -1], [1, 0, 0]),
        vertex([1, 1, -1], [1, 0, 0]),
        vertex([1, 1, 1], [1, 0, 0]),
        vertex([1, -1, 1], [1, 0, 0]),
        // left (-1, 0, 0)
        vertex([-1, -1, 1], [-1, 0, 0]),
        vertex([-1, 1, 1], [-1, 0, 0]),
        vertex([-1, 1, -1], [-1, 0, 0]),
        vertex([-1, -1, -1], [-1, 0, 0]),
        // front (0, 1, 0)
        vertex([1, 1, -1], [0, 1, 0]),
        vertex([-1, 1, -1], [0, 1, 0]),
        vertex([-1, 1, 1], [0, 1, 0]),
        vertex([1, 1, 1], [0, 1, 0]),
        // back (0, -1, 0)
        vertex([1, -1, 1], [0, -1, 0]),
        vertex([-1, -1, 1], [0, -1, 0]),
        vertex([-1, -1, -1], [0, -1, 0]),
        vertex([1, -1, -1], [0, -1, 0]),
    ];

    let index_data: &[u16] = &[
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}

fn create_plane(size: i8) -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        vertex([size, -size, 0], [0, 0, 1]),
        vertex([size, size, 0], [0, 0, 1]),
        vertex([-size, -size, 0], [0, 0, 1]),
        vertex([-size, size, 0], [0, 0, 1]),
    ];

    let index_data: &[u16] = &[0, 1, 2, 2, 1, 3];

    (vertex_data.to_vec(), index_data.to_vec())
}

struct Entity {
    mx_world: glam::Mat4,
    rotation_speed: f32,
    color: etalon::Color,
    vertex_buf: etalon::Buffer,
    index_buf: etalon::Buffer,
    index_format: etalon::IndexFormat,
    index_count: usize,
    uniform_offset: etalon::DynamicOffset,
}

struct Light {
    pos: glam::Vec3,
    color: etalon::Color,
    fov: f32,
    depth: Range<f32>,
    target_view: etalon::TextureView,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightRaw {
    proj: [[f32; 4]; 4],
    pos: [f32; 4],
    color: [f32; 4],
}

impl Light {
    fn to_raw(&self) -> LightRaw {
        let view = glam::camera::rh::view::look_at_mat4(self.pos, glam::Vec3::ZERO, glam::Vec3::Z);
        let projection = glam::camera::rh::proj::directx::perspective(
            self.fov * consts::PI / 180.,
            1.0,
            self.depth.start,
            self.depth.end,
        );
        let view_proj = projection * view;
        LightRaw {
            proj: view_proj.to_cols_array_2d(),
            pos: [self.pos.x, self.pos.y, self.pos.z, 1.0],
            color: [
                self.color.r as f32,
                self.color.g as f32,
                self.color.b as f32,
                1.0,
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalUniforms {
    proj: [[f32; 4]; 4],
    num_lights: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EntityUniforms {
    model: [[f32; 4]; 4],
    color: [f32; 4],
}

struct Pass {
    pipeline: etalon::RenderPipeline,
    bind_group: etalon::BindGroup,
    uniform_buf: etalon::Buffer,
}

struct Example {
    entities: Vec<Entity>,
    lights: Vec<Light>,
    lights_are_dirty: bool,
    shadow_pass: Pass,
    forward_pass: Pass,
    forward_depth: etalon::TextureView,
    entity_bind_group: etalon::BindGroup,
    light_storage_buf: etalon::Buffer,
    entity_uniform_buf: etalon::Buffer,
}

impl Example {
    const MAX_LIGHTS: usize = 10;
    const SHADOW_FORMAT: etalon::TextureFormat = etalon::TextureFormat::Depth32Float;
    const SHADOW_SIZE: etalon::Extent3d = etalon::Extent3d {
        width: 512,
        height: 512,
        depth_or_array_layers: Self::MAX_LIGHTS as u32,
    };
    const DEPTH_FORMAT: etalon::TextureFormat = etalon::TextureFormat::Depth32Float;

    fn generate_matrix(aspect_ratio: f32) -> glam::Mat4 {
        let projection = glam::camera::rh::proj::directx::perspective(
            consts::FRAC_PI_4,
            aspect_ratio,
            1.0,
            20.0,
        );
        let view = glam::camera::rh::view::look_at_mat4(
            glam::Vec3::new(3.0f32, -10.0, 6.0),
            glam::Vec3::new(0f32, 0.0, 0.0),
            glam::Vec3::Z,
        );
        projection * view
    }

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
        etalon::Features::DEPTH_CLIP_CONTROL
    }

    fn init(
        config: &etalon::SurfaceConfiguration,
        adapter: &etalon::Adapter,
        device: &etalon::Device,
        _queue: &etalon::Queue,
    ) -> Self {
        let supports_storage_resources = adapter
            .get_downlevel_capabilities()
            .flags
            .contains(etalon::DownlevelFlags::VERTEX_STORAGE)
            && device.limits().max_storage_buffers_per_shader_stage > 0;

        // Create the vertex and index buffers
        let vertex_size = size_of::<Vertex>();
        let (cube_vertex_data, cube_index_data) = create_cube();
        let cube_vertex_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Cubes Vertex Buffer"),
            contents: bytemuck::cast_slice(&cube_vertex_data),
            usage: etalon::BufferUsages::VERTEX,
        });

        let cube_index_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Cubes Index Buffer"),
            contents: bytemuck::cast_slice(&cube_index_data),
            usage: etalon::BufferUsages::INDEX,
        });

        let (plane_vertex_data, plane_index_data) = create_plane(7);
        let plane_vertex_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Plane Vertex Buffer"),
            contents: bytemuck::cast_slice(&plane_vertex_data),
            usage: etalon::BufferUsages::VERTEX,
        });

        let plane_index_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Plane Index Buffer"),
            contents: bytemuck::cast_slice(&plane_index_data),
            usage: etalon::BufferUsages::INDEX,
        });

        struct CubeDesc {
            offset: glam::Vec3,
            angle: f32,
            scale: f32,
            rotation: f32,
        }
        let cube_descs = [
            CubeDesc {
                offset: glam::Vec3::new(-2.0, -2.0, 2.0),
                angle: 10.0,
                scale: 0.7,
                rotation: 0.1,
            },
            CubeDesc {
                offset: glam::Vec3::new(2.0, -2.0, 2.0),
                angle: 50.0,
                scale: 1.3,
                rotation: 0.2,
            },
            CubeDesc {
                offset: glam::Vec3::new(-2.0, 2.0, 2.0),
                angle: 140.0,
                scale: 1.1,
                rotation: 0.3,
            },
            CubeDesc {
                offset: glam::Vec3::new(2.0, 2.0, 2.0),
                angle: 210.0,
                scale: 0.9,
                rotation: 0.4,
            },
        ];

        let entity_uniform_size = size_of::<EntityUniforms>() as etalon::BufferAddress;
        let num_entities = 1 + cube_descs.len() as etalon::BufferAddress;
        // Make the `uniform_alignment` >= `entity_uniform_size` and aligned to `min_uniform_buffer_offset_alignment`.
        let uniform_alignment = {
            let alignment =
                device.limits().min_uniform_buffer_offset_alignment as etalon::BufferAddress;
            align_to(entity_uniform_size, alignment)
        };
        // Note: dynamic uniform offsets also have to be aligned to `Limits::min_uniform_buffer_offset_alignment`.
        let entity_uniform_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: None,
            size: num_entities * uniform_alignment,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_format = etalon::IndexFormat::Uint16;

        let mut entities = vec![{
            Entity {
                mx_world: glam::Mat4::IDENTITY,
                rotation_speed: 0.0,
                color: etalon::Color::WHITE,
                vertex_buf: plane_vertex_buf,
                index_buf: plane_index_buf,
                index_format,
                index_count: plane_index_data.len(),
                uniform_offset: 0,
            }
        }];

        for (i, cube) in cube_descs.iter().enumerate() {
            let mx_world = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::splat(cube.scale),
                glam::Quat::from_axis_angle(
                    cube.offset.normalize(),
                    cube.angle * consts::PI / 180.,
                ),
                cube.offset,
            );
            entities.push(Entity {
                mx_world,
                rotation_speed: cube.rotation,
                color: etalon::Color::GREEN,
                vertex_buf: cube_vertex_buf.clone(),
                index_buf: cube_index_buf.clone(),
                index_format,
                index_count: cube_index_data.len(),
                uniform_offset: ((i + 1) * uniform_alignment as usize) as _,
            });
        }

        let local_bind_group_layout =
            device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                entries: &[etalon::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: etalon::ShaderStages::VERTEX | etalon::ShaderStages::FRAGMENT,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: etalon::BufferSize::new(entity_uniform_size),
                    },
                    count: None,
                }],
                label: None,
            });
        let entity_bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            layout: &local_bind_group_layout,
            entries: &[etalon::BindGroupEntry {
                binding: 0,
                resource: etalon::BindingResource::Buffer(etalon::BufferBinding {
                    buffer: &entity_uniform_buf,
                    offset: 0,
                    size: etalon::BufferSize::new(entity_uniform_size),
                }),
            }],
            label: None,
        });

        // Create other resources
        let shadow_sampler = device.create_sampler(&etalon::SamplerDescriptor {
            label: Some("shadow"),
            address_mode_u: etalon::AddressMode::ClampToEdge,
            address_mode_v: etalon::AddressMode::ClampToEdge,
            address_mode_w: etalon::AddressMode::ClampToEdge,
            mag_filter: etalon::FilterMode::Linear,
            min_filter: etalon::FilterMode::Linear,
            mipmap_filter: etalon::MipmapFilterMode::Nearest,
            compare: Some(etalon::CompareFunction::LessEqual),
            ..Default::default()
        });

        let shadow_texture = device.create_texture(&etalon::TextureDescriptor {
            size: Self::SHADOW_SIZE,
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: Self::SHADOW_FORMAT,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::TEXTURE_BINDING,
            label: None,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&etalon::TextureViewDescriptor::default());

        let mut shadow_target_views = (0..2)
            .map(|i| {
                Some(shadow_texture.create_view(&etalon::TextureViewDescriptor {
                    label: Some("shadow"),
                    format: None,
                    dimension: Some(etalon::TextureViewDimension::D2),
                    usage: None,
                    aspect: etalon::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: i as u32,
                    array_layer_count: Some(1),
                    swizzle: etalon::TextureComponentSwizzle::default(),
                }))
            })
            .collect::<Vec<_>>();
        let lights = vec![
            Light {
                pos: glam::Vec3::new(7.0, -5.0, 10.0),
                color: etalon::Color {
                    r: 0.5,
                    g: 1.0,
                    b: 0.5,
                    a: 1.0,
                },
                fov: 60.0,
                depth: 1.0..20.0,
                target_view: shadow_target_views[0].take().unwrap(),
            },
            Light {
                pos: glam::Vec3::new(-5.0, 7.0, 10.0),
                color: etalon::Color {
                    r: 1.0,
                    g: 0.5,
                    b: 0.5,
                    a: 1.0,
                },
                fov: 45.0,
                depth: 1.0..20.0,
                target_view: shadow_target_views[1].take().unwrap(),
            },
        ];
        let light_uniform_size = (Self::MAX_LIGHTS * size_of::<LightRaw>()) as etalon::BufferAddress;
        let light_storage_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: None,
            size: light_uniform_size,
            usage: if supports_storage_resources {
                etalon::BufferUsages::STORAGE
            } else {
                etalon::BufferUsages::UNIFORM
            } | etalon::BufferUsages::COPY_SRC
                | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vertex_attr = etalon::vertex_attr_array![0 => Sint8x4, 1 => Sint8x4];
        let vertex_buffers = [Some(etalon::VertexBufferLayout {
            array_stride: vertex_size as etalon::BufferAddress,
            step_mode: etalon::VertexStepMode::Vertex,
            attributes: &vertex_attr,
        })];

        let shader = device.create_shader_module(etalon::include_wgsl!("shader.wgsl"));

        let shadow_pass = {
            let uniform_size = size_of::<GlobalUniforms>() as etalon::BufferAddress;
            // Create pipeline layout
            let bind_group_layout =
                device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[etalon::BindGroupLayoutEntry {
                        binding: 0, // global
                        visibility: etalon::ShaderStages::VERTEX,
                        ty: etalon::BindingType::Buffer {
                            ty: etalon::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: etalon::BufferSize::new(uniform_size),
                        },
                        count: None,
                    }],
                });
            let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
                label: Some("shadow"),
                bind_group_layouts: &[Some(&bind_group_layout), Some(&local_bind_group_layout)],
                immediate_size: 0,
            });

            let uniform_buf = device.create_buffer(&etalon::BufferDescriptor {
                label: None,
                size: uniform_size,
                usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Create bind group
            let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[etalon::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                }],
                label: None,
            });

            // Create the render pipeline
            let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
                label: Some("shadow"),
                layout: Some(&pipeline_layout),
                vertex: etalon::VertexState {
                    module: &shader,
                    entry_point: Some("vs_bake"),
                    compilation_options: Default::default(),
                    buffers: &vertex_buffers,
                },
                fragment: None,
                primitive: etalon::PrimitiveState {
                    topology: etalon::PrimitiveTopology::TriangleList,
                    front_face: etalon::FrontFace::Ccw,
                    cull_mode: Some(etalon::Face::Back),
                    unclipped_depth: device
                        .features()
                        .contains(etalon::Features::DEPTH_CLIP_CONTROL),
                    ..Default::default()
                },
                depth_stencil: Some(etalon::DepthStencilState {
                    format: Self::SHADOW_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(etalon::CompareFunction::LessEqual),
                    stencil: etalon::StencilState::default(),
                    bias: etalon::DepthBiasState {
                        constant: 2, // corresponds to bilinear filtering
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
                }),
                multisample: etalon::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

            Pass {
                pipeline,
                bind_group,
                uniform_buf,
            }
        };

        let forward_pass = {
            // Create pipeline layout
            let bind_group_layout =
                device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                    entries: &[
                        etalon::BindGroupLayoutEntry {
                            binding: 0, // global
                            visibility: etalon::ShaderStages::VERTEX | etalon::ShaderStages::FRAGMENT,
                            ty: etalon::BindingType::Buffer {
                                ty: etalon::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: etalon::BufferSize::new(
                                    size_of::<GlobalUniforms>() as _,
                                ),
                            },
                            count: None,
                        },
                        etalon::BindGroupLayoutEntry {
                            binding: 1, // lights
                            visibility: etalon::ShaderStages::FRAGMENT,
                            ty: etalon::BindingType::Buffer {
                                ty: if supports_storage_resources {
                                    etalon::BufferBindingType::Storage { read_only: true }
                                } else {
                                    etalon::BufferBindingType::Uniform
                                },
                                has_dynamic_offset: false,
                                min_binding_size: etalon::BufferSize::new(light_uniform_size),
                            },
                            count: None,
                        },
                        etalon::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: etalon::ShaderStages::FRAGMENT,
                            ty: etalon::BindingType::Texture {
                                multisampled: false,
                                sample_type: etalon::TextureSampleType::Depth,
                                view_dimension: etalon::TextureViewDimension::D2Array,
                            },
                            count: None,
                        },
                        etalon::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: etalon::ShaderStages::FRAGMENT,
                            ty: etalon::BindingType::Sampler(etalon::SamplerBindingType::Comparison),
                            count: None,
                        },
                    ],
                    label: None,
                });
            let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
                label: Some("main"),
                bind_group_layouts: &[Some(&bind_group_layout), Some(&local_bind_group_layout)],
                immediate_size: 0,
            });

            let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32);
            let forward_uniforms = GlobalUniforms {
                proj: mx_total.to_cols_array_2d(),
                num_lights: [lights.len() as u32, 0, 0, 0],
            };
            let uniform_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::bytes_of(&forward_uniforms),
                usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            });

            // Create bind group
            let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    etalon::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 1,
                        resource: light_storage_buf.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 2,
                        resource: etalon::BindingResource::TextureView(&shadow_view),
                    },
                    etalon::BindGroupEntry {
                        binding: 3,
                        resource: etalon::BindingResource::Sampler(&shadow_sampler),
                    },
                ],
                label: None,
            });

            // Create the render pipeline
            let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
                label: Some("main"),
                layout: Some(&pipeline_layout),
                vertex: etalon::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &vertex_buffers,
                },
                fragment: Some(etalon::FragmentState {
                    module: &shader,
                    entry_point: Some(if supports_storage_resources {
                        "fs_main"
                    } else {
                        "fs_main_without_storage"
                    }),
                    compilation_options: Default::default(),
                    targets: &[Some(config.view_formats[0].into())],
                }),
                primitive: etalon::PrimitiveState {
                    front_face: etalon::FrontFace::Ccw,
                    cull_mode: Some(etalon::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(etalon::DepthStencilState {
                    format: Self::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(etalon::CompareFunction::Less),
                    stencil: etalon::StencilState::default(),
                    bias: etalon::DepthBiasState::default(),
                }),
                multisample: etalon::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

            Pass {
                pipeline,
                bind_group,
                uniform_buf,
            }
        };

        let forward_depth = Self::create_depth_texture(config, device);

        Example {
            entities,
            lights,
            lights_are_dirty: true,
            shadow_pass,
            forward_pass,
            forward_depth,
            light_storage_buf,
            entity_uniform_buf,
            entity_bind_group,
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        config: &etalon::SurfaceConfiguration,
        device: &etalon::Device,
        queue: &etalon::Queue,
    ) {
        // update view-projection matrix
        let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        queue.write_buffer(
            &self.forward_pass.uniform_buf,
            0,
            bytemuck::cast_slice(mx_ref),
        );

        self.forward_depth = Self::create_depth_texture(config, device);
    }

    fn render(&mut self, view: &etalon::TextureView, device: &etalon::Device, queue: &etalon::Queue) {
        // update uniforms
        for entity in self.entities.iter_mut() {
            if entity.rotation_speed != 0.0 {
                let rotation =
                    glam::Mat4::from_rotation_x(entity.rotation_speed * consts::PI / 180.);
                entity.mx_world *= rotation;
            }
            let data = EntityUniforms {
                model: entity.mx_world.to_cols_array_2d(),
                color: [
                    entity.color.r as f32,
                    entity.color.g as f32,
                    entity.color.b as f32,
                    entity.color.a as f32,
                ],
            };
            queue.write_buffer(
                &self.entity_uniform_buf,
                entity.uniform_offset as etalon::BufferAddress,
                bytemuck::bytes_of(&data),
            );
        }

        if self.lights_are_dirty {
            self.lights_are_dirty = false;
            for (i, light) in self.lights.iter().enumerate() {
                queue.write_buffer(
                    &self.light_storage_buf,
                    (i * size_of::<LightRaw>()) as etalon::BufferAddress,
                    bytemuck::bytes_of(&light.to_raw()),
                );
            }
        }

        let mut encoder =
            device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });

        encoder.push_debug_group("shadow passes");
        for (i, light) in self.lights.iter().enumerate() {
            encoder.push_debug_group(&format!(
                "shadow pass {} (light at position {:?})",
                i, light.pos
            ));

            // The light uniform buffer already has the projection,
            // let's just copy it over to the shadow uniform buffer.
            encoder.copy_buffer_to_buffer(
                &self.light_storage_buf,
                (i * size_of::<LightRaw>()) as etalon::BufferAddress,
                &self.shadow_pass.uniform_buf,
                0,
                64,
            );

            encoder.insert_debug_marker("render entities");
            {
                let mut pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[],
                    depth_stencil_attachment: Some(etalon::RenderPassDepthStencilAttachment {
                        view: &light.target_view,
                        depth_ops: Some(etalon::Operations {
                            load: etalon::LoadOp::Clear(1.0),
                            store: etalon::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.shadow_pass.pipeline);
                pass.set_bind_group(0, &self.shadow_pass.bind_group, &[]);

                for entity in &self.entities {
                    pass.set_bind_group(1, &self.entity_bind_group, &[entity.uniform_offset]);
                    pass.set_index_buffer(entity.index_buf.slice(..), entity.index_format);
                    pass.set_vertex_buffer(0, entity.vertex_buf.slice(..));
                    pass.draw_indexed(0..entity.index_count as u32, 0, 0..1);
                }
            }

            encoder.pop_debug_group();
        }
        encoder.pop_debug_group();

        // forward pass
        encoder.push_debug_group("forward rendering pass");
        {
            let mut pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
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
                    view: &self.forward_depth,
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
            pass.set_pipeline(&self.forward_pass.pipeline);
            pass.set_bind_group(0, &self.forward_pass.bind_group, &[]);

            for entity in &self.entities {
                pass.set_bind_group(1, &self.entity_bind_group, &[entity.uniform_offset]);
                pass.set_index_buffer(entity.index_buf.slice(..), entity.index_format);
                pass.set_vertex_buffer(0, entity.vertex_buf.slice(..));
                pass.draw_indexed(0..entity.index_count as u32, 0, 0..1);
            }
        }
        encoder.pop_debug_group();

        queue.submit(iter::once(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("shadow");
}

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "shadow",
    image_path: "/examples/features/src/shadow/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::default(),
    base_test_parameters: etalon_test::TestParameters::default()
        .downlevel_flags(etalon::DownlevelFlags::COMPARISON_SAMPLERS)
        // rpi4 on VK doesn't work: https://gitlab.freedesktop.org/mesa/mesa/-/issues/3916
        .expect_fail(etalon_test::FailureCase::backend_adapter(
            etalon::Backends::VULKAN,
            "V3D",
        )),
    comparisons: &[etalon_test::ComparisonType::Mean(0.026)], // Bounded by Apple A9
    _phantom: std::marker::PhantomData::<Example>,
};
