use bytemuck::{Pod, Zeroable};
use std::num::{NonZeroU32, NonZeroU64};
use etalon::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 2],
    _tex_coord: [f32; 2],
    _index: u32,
}

fn vertex(pos: [i8; 2], tc: [i8; 2], index: i8) -> Vertex {
    Vertex {
        _pos: [pos[0] as f32, pos[1] as f32],
        _tex_coord: [tc[0] as f32, tc[1] as f32],
        _index: index as u32,
    }
}

fn create_vertices() -> Vec<Vertex> {
    vec![
        // left rectangle
        vertex([-1, -1], [0, 1], 0),
        vertex([-1, 1], [0, 0], 0),
        vertex([0, 1], [1, 0], 0),
        vertex([0, -1], [1, 1], 0),
        // right rectangle
        vertex([0, -1], [0, 1], 1),
        vertex([0, 1], [0, 0], 1),
        vertex([1, 1], [1, 0], 1),
        vertex([1, -1], [1, 1], 1),
    ]
}

fn create_indices() -> Vec<u16> {
    vec![
        // Left rectangle
        0, 1, 2, // 1st
        2, 0, 3, // 2nd
        // Right rectangle
        4, 5, 6, // 1st
        6, 4, 7, // 2nd
    ]
}

#[derive(Copy, Clone)]
enum Color {
    Red,
    Green,
    Blue,
    White,
}

fn create_texture_data(color: Color) -> [u8; 4] {
    match color {
        Color::Red => [255, 0, 0, 255],
        Color::Green => [0, 255, 0, 255],
        Color::Blue => [0, 0, 255, 255],
        Color::White => [255, 255, 255, 255],
    }
}

struct Example {
    pipeline: etalon::RenderPipeline,
    bind_group: etalon::BindGroup,
    uniform_bind_group: etalon::BindGroup,
    vertex_buffer: etalon::Buffer,
    index_buffer: etalon::Buffer,
    index_format: etalon::IndexFormat,
    uniform_workaround: bool,
}

impl crate::framework::Example for Example {
    fn optional_features() -> etalon::Features {
        etalon::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
    }
    fn required_features() -> etalon::Features {
        etalon::Features::TEXTURE_BINDING_ARRAY
    }
    fn required_limits() -> etalon::Limits {
        etalon::Limits {
            max_binding_array_elements_per_shader_stage: 6,
            max_binding_array_sampler_elements_per_shader_stage: 2,
            ..etalon::Limits::downlevel_defaults()
        }
    }
    fn init(
        config: &etalon::SurfaceConfiguration,
        _adapter: &etalon::Adapter,
        device: &etalon::Device,
        queue: &etalon::Queue,
    ) -> Self {
        let mut uniform_workaround = false;
        let base_shader_module = device.create_shader_module(etalon::include_wgsl!("indexing.wgsl"));
        let env_override = match std::env::var("etalon_TEXTURE_ARRAY_STYLE") {
            Ok(value) => match &*value.to_lowercase() {
                "nonuniform" | "non_uniform" => Some(true),
                "uniform" => Some(false),
                _ => None,
            },
            Err(_) => None,
        };
        let fragment_entry_point = match (device.features(), env_override) {
            (_, Some(false)) => {
                uniform_workaround = true;
                "uniform_main"
            }
            (_, Some(true)) => "non_uniform_main",
            (f, _)
                if f.contains(
                    etalon::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                ) =>
            {
                "non_uniform_main"
            }
            _ => {
                uniform_workaround = true;
                "uniform_main"
            }
        };
        let non_uniform_shader_module;
        // TODO: Because naga's capabilities are evaluated on validate, not on write, we cannot make a shader module with unsupported
        // capabilities even if we don't use it. So for now put it in a separate module.
        let fragment_shader_module = if !uniform_workaround {
            non_uniform_shader_module =
                device.create_shader_module(etalon::include_wgsl!("non_uniform_indexing.wgsl"));
            &non_uniform_shader_module
        } else {
            &base_shader_module
        };

        println!("Using fragment entry point '{fragment_entry_point}'");

        let vertex_size = size_of::<Vertex>();
        let vertex_data = create_vertices();
        let vertex_buffer = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: etalon::BufferUsages::VERTEX,
        });

        let index_data = create_indices();
        let index_buffer = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&index_data),
            usage: etalon::BufferUsages::INDEX,
        });

        let mut texture_index_buffer_contents = vec![0u32; 128];
        texture_index_buffer_contents[0] = 0;
        texture_index_buffer_contents[64] = 1;
        let texture_index_buffer = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&texture_index_buffer_contents),
            usage: etalon::BufferUsages::UNIFORM,
        });

        let red_texture_data = create_texture_data(Color::Red);
        let green_texture_data = create_texture_data(Color::Green);
        let blue_texture_data = create_texture_data(Color::Blue);
        let white_texture_data = create_texture_data(Color::White);

        let texture_descriptor = etalon::TextureDescriptor {
            size: etalon::Extent3d::default(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rgba8UnormSrgb,
            usage: etalon::TextureUsages::TEXTURE_BINDING | etalon::TextureUsages::COPY_DST,
            label: None,
            view_formats: &[],
        };
        let red_texture = device.create_texture(&etalon::TextureDescriptor {
            label: Some("red"),
            view_formats: &[],
            ..texture_descriptor
        });
        let green_texture = device.create_texture(&etalon::TextureDescriptor {
            label: Some("green"),
            view_formats: &[],
            ..texture_descriptor
        });
        let blue_texture = device.create_texture(&etalon::TextureDescriptor {
            label: Some("blue"),
            view_formats: &[],
            ..texture_descriptor
        });
        let white_texture = device.create_texture(&etalon::TextureDescriptor {
            label: Some("white"),
            view_formats: &[],
            ..texture_descriptor
        });

        let red_texture_view = red_texture.create_view(&etalon::TextureViewDescriptor::default());
        let green_texture_view = green_texture.create_view(&etalon::TextureViewDescriptor::default());
        let blue_texture_view = blue_texture.create_view(&etalon::TextureViewDescriptor::default());
        let white_texture_view = white_texture.create_view(&etalon::TextureViewDescriptor::default());

        queue.write_texture(
            red_texture.as_image_copy(),
            &red_texture_data,
            etalon::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            etalon::Extent3d::default(),
        );
        queue.write_texture(
            green_texture.as_image_copy(),
            &green_texture_data,
            etalon::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            etalon::Extent3d::default(),
        );
        queue.write_texture(
            blue_texture.as_image_copy(),
            &blue_texture_data,
            etalon::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            etalon::Extent3d::default(),
        );
        queue.write_texture(
            white_texture.as_image_copy(),
            &white_texture_data,
            etalon::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            etalon::Extent3d::default(),
        );

        let sampler = device.create_sampler(&etalon::SamplerDescriptor::default());

        let bind_group_layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("bind group layout"),
            entries: &[
                etalon::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: etalon::ShaderStages::FRAGMENT,
                    ty: etalon::BindingType::Texture {
                        sample_type: etalon::TextureSampleType::Float { filterable: true },
                        view_dimension: etalon::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: NonZeroU32::new(2),
                },
                etalon::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: etalon::ShaderStages::FRAGMENT,
                    ty: etalon::BindingType::Texture {
                        sample_type: etalon::TextureSampleType::Float { filterable: true },
                        view_dimension: etalon::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: NonZeroU32::new(2),
                },
                etalon::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: etalon::ShaderStages::FRAGMENT,
                    ty: etalon::BindingType::Sampler(etalon::SamplerBindingType::Filtering),
                    count: NonZeroU32::new(2),
                },
            ],
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                label: Some("uniform bind group layout"),
                entries: &[etalon::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: etalon::ShaderStages::FRAGMENT,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                }],
            });

        let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: etalon::BindingResource::TextureViewArray(&[
                        &red_texture_view,
                        &green_texture_view,
                    ]),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: etalon::BindingResource::TextureViewArray(&[
                        &blue_texture_view,
                        &white_texture_view,
                    ]),
                },
                etalon::BindGroupEntry {
                    binding: 2,
                    resource: etalon::BindingResource::SamplerArray(&[&sampler, &sampler]),
                },
            ],
            layout: &bind_group_layout,
            label: Some("bind group"),
        });

        let uniform_bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            entries: &[etalon::BindGroupEntry {
                binding: 0,
                resource: etalon::BindingResource::Buffer(etalon::BufferBinding {
                    buffer: &texture_index_buffer,
                    offset: 0,
                    size: Some(NonZeroU64::new(4).unwrap()),
                }),
            }],
            layout: &uniform_bind_group_layout,
            label: Some("uniform bind group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("main"),
            bind_group_layouts: &[Some(&bind_group_layout), Some(&uniform_bind_group_layout)],
            immediate_size: 0,
        });

        let index_format = etalon::IndexFormat::Uint16;

        let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: etalon::VertexState {
                module: &base_shader_module,
                entry_point: Some("vert_main"),
                compilation_options: Default::default(),
                buffers: &[Some(etalon::VertexBufferLayout {
                    array_stride: vertex_size as etalon::BufferAddress,
                    step_mode: etalon::VertexStepMode::Vertex,
                    attributes: &etalon::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Sint32],
                })],
            },
            fragment: Some(etalon::FragmentState {
                module: fragment_shader_module,
                entry_point: Some(fragment_entry_point),
                compilation_options: Default::default(),
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: etalon::PrimitiveState {
                front_face: etalon::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None
        });

        Self {
            pipeline,
            bind_group,
            uniform_bind_group,
            vertex_buffer,
            index_buffer,
            index_format,
            uniform_workaround,
        }
    }
    fn resize(
        &mut self,
        _sc_desc: &etalon::SurfaceConfiguration,
        _device: &etalon::Device,
        _queue: &etalon::Queue,
    ) {
        // noop
    }
    fn update(&mut self, _event: winit::event::WindowEvent) {
        // noop
    }
    fn render(&mut self, view: &etalon::TextureView, device: &etalon::Device, queue: &etalon::Queue) {
        let mut encoder = device.create_command_encoder(&etalon::CommandEncoderDescriptor {
            label: Some("primary"),
        });

        let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: None,
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

        rpass.set_pipeline(&self.pipeline);
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rpass.set_index_buffer(self.index_buffer.slice(..), self.index_format);
        if self.uniform_workaround {
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_bind_group(1, &self.uniform_bind_group, &[0]);
            rpass.draw_indexed(0..6, 0, 0..1);
            rpass.set_bind_group(1, &self.uniform_bind_group, &[256]);
            rpass.draw_indexed(6..12, 0, 0..1);
        } else {
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_bind_group(1, &self.uniform_bind_group, &[0]);
            rpass.draw_indexed(0..12, 0, 0..1);
        }

        drop(rpass);

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("texture-arrays");
}

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "texture-arrays",
    image_path: "/examples/features/src/texture_arrays/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::empty(),
    base_test_parameters: etalon_test::TestParameters::default()
        .instance_flags(etalon::InstanceFlags::GPU_BASED_VALIDATION)
        // https://github.com/mediaprophet/etalon/issues/9184
        .expect_fail(
            etalon_test::FailureCase::molten_vk()
                .validation_error("Shader library compile failed")
                .validation_error("could not be compiled into pipeline"),
        ),
    comparisons: &[etalon_test::ComparisonType::Mean(0.0001)],
    _phantom: std::marker::PhantomData::<Example>,
};

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST_UNIFORM: crate::framework::ExampleTestParams =
    crate::framework::ExampleTestParams {
        name: "texture-arrays-uniform",
        image_path: "/examples/features/src/texture_arrays/screenshot.png",
        width: 1024,
        height: 768,
        optional_features: etalon::Features::empty(),
        base_test_parameters: etalon_test::TestParameters::default()
            .instance_flags(etalon::InstanceFlags::GPU_BASED_VALIDATION)
            // https://github.com/mediaprophet/etalon/issues/9184
            .expect_fail(
                etalon_test::FailureCase::molten_vk()
                    .validation_error("Shader library compile failed")
                    .validation_error("could not be compiled into pipeline"),
            ),
        comparisons: &[etalon_test::ComparisonType::Mean(0.0001)],
        _phantom: std::marker::PhantomData::<Example>,
    };

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST_NON_UNIFORM: crate::framework::ExampleTestParams =
    crate::framework::ExampleTestParams {
        name: "texture-arrays-non-uniform",
        image_path: "/examples/features/src/texture_arrays/screenshot.png",
        width: 1024,
        height: 768,
        optional_features:
            etalon::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        base_test_parameters: etalon_test::TestParameters::default()
            .instance_flags(etalon::InstanceFlags::GPU_BASED_VALIDATION)
            // https://github.com/mediaprophet/etalon/issues/9184
            .expect_fail(
                etalon_test::FailureCase::molten_vk()
                    .validation_error("Shader library compile failed")
                    .validation_error("could not be compiled into pipeline"),
            ),
        comparisons: &[etalon_test::ComparisonType::Mean(0.0001)],
        _phantom: std::marker::PhantomData::<Example>,
    };
