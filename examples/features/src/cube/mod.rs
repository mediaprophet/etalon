use bytemuck::{Pod, Zeroable};
use etalon::util::DeviceExt;
use std::f32::consts;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 4],
    _tex_coord: [f32; 2],
}

fn vertex(pos: [i8; 3], tc: [i8; 2]) -> Vertex {
    Vertex {
        _pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
        _tex_coord: [tc[0] as f32, tc[1] as f32],
    }
}

fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // top (0, 0, 1)
        vertex([-1, -1, 1], [0, 0]),
        vertex([1, -1, 1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([-1, 1, 1], [0, 1]),
        // bottom (0, 0, -1)
        vertex([-1, 1, -1], [1, 0]),
        vertex([1, 1, -1], [0, 0]),
        vertex([1, -1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // right (1, 0, 0)
        vertex([1, -1, -1], [0, 0]),
        vertex([1, 1, -1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([1, -1, 1], [0, 1]),
        // left (-1, 0, 0)
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, 1, 1], [0, 0]),
        vertex([-1, 1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // front (0, 1, 0)
        vertex([1, 1, -1], [1, 0]),
        vertex([-1, 1, -1], [0, 0]),
        vertex([-1, 1, 1], [0, 1]),
        vertex([1, 1, 1], [1, 1]),
        // back (0, -1, 0)
        vertex([1, -1, 1], [0, 0]),
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, -1, -1], [1, 1]),
        vertex([1, -1, -1], [0, 1]),
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

fn create_texels(size: usize) -> Vec<u8> {
    (0..size * size)
        .map(|id| {
            // get high five for recognizing this ;)
            let cx = 3.0 * (id % size) as f32 / (size - 1) as f32 - 2.0;
            let cy = 2.0 * (id / size) as f32 / (size - 1) as f32 - 1.0;
            let (mut x, mut y, mut count) = (cx, cy, 0);
            while count < 0xFF && x * x + y * y < 4.0 {
                let old_x = x;
                x = x * x - y * y + cx;
                y = 2.0 * old_x * y + cy;
                count += 1;
            }
            count
        })
        .collect()
}

struct Example {
    vertex_buf: etalon::Buffer,
    index_buf: etalon::Buffer,
    index_count: usize,
    bind_group: etalon::BindGroup,
    uniform_buf: etalon::Buffer,
    pipeline: etalon::RenderPipeline,
    pipeline_wire: Option<etalon::RenderPipeline>,
}

impl Example {
    fn generate_matrix(aspect_ratio: f32) -> glam::Mat4 {
        let projection = glam::camera::rh::proj::directx::perspective(
            consts::FRAC_PI_4,
            aspect_ratio,
            1.0,
            10.0,
        );
        let view = glam::camera::rh::view::look_at_mat4(
            glam::Vec3::new(1.5f32, -5.0, 3.0),
            glam::Vec3::ZERO,
            glam::Vec3::Z,
        );
        projection * view
    }
}

impl crate::framework::Example for Example {
    fn optional_features() -> etalon::Features {
        etalon::Features::POLYGON_MODE_LINE
    }

    fn init(
        config: &etalon::SurfaceConfiguration,
        _adapter: &etalon::Adapter,
        device: &etalon::Device,
        queue: &etalon::Queue,
    ) -> Self {
        // Create the vertex and index buffers
        let vertex_size = size_of::<Vertex>();
        let (vertex_data, index_data) = create_vertices();

        let vertex_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: etalon::BufferUsages::VERTEX,
        });

        let index_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&index_data),
            usage: etalon::BufferUsages::INDEX,
        });

        // Create pipeline layout
        let bind_group_layout =
            device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    etalon::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: etalon::ShaderStages::VERTEX,
                        ty: etalon::BindingType::Buffer {
                            ty: etalon::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: etalon::BufferSize::new(64),
                        },
                        count: None,
                    },
                    etalon::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: etalon::ShaderStages::FRAGMENT,
                        ty: etalon::BindingType::Texture {
                            multisampled: false,
                            sample_type: etalon::TextureSampleType::Uint,
                            view_dimension: etalon::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Create the texture
        let size = 256u32;
        let texels = create_texels(size as usize);
        let texture_extent = etalon::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::R8Uint,
            usage: etalon::TextureUsages::TEXTURE_BINDING | etalon::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&etalon::TextureViewDescriptor::default());
        queue.write_texture(
            texture.as_image_copy(),
            &texels,
            etalon::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size),
                rows_per_image: None,
            },
            texture_extent,
        );

        // Create other resources
        let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        let uniform_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(mx_ref),
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
                    resource: etalon::BindingResource::TextureView(&texture_view),
                },
            ],
            label: None,
        });

        let shader = device.create_shader_module(etalon::include_wgsl!("shader.wgsl"));

        let vertex_buffers = [Some(etalon::VertexBufferLayout {
            array_stride: vertex_size as etalon::BufferAddress,
            step_mode: etalon::VertexStepMode::Vertex,
            attributes: &[
                etalon::VertexAttribute {
                    format: etalon::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
                etalon::VertexAttribute {
                    format: etalon::VertexFormat::Float32x2,
                    offset: 4 * 4,
                    shader_location: 1,
                },
            ],
        })];

        let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: etalon::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &vertex_buffers,
            },
            fragment: Some(etalon::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: etalon::PrimitiveState {
                cull_mode: Some(etalon::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipeline_wire = if device
            .features()
            .contains(etalon::Features::POLYGON_MODE_LINE)
        {
            let pipeline_wire = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: etalon::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &vertex_buffers,
                },
                fragment: Some(etalon::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_wire"),
                    compilation_options: Default::default(),
                    targets: &[Some(etalon::ColorTargetState {
                        format: config.view_formats[0],
                        blend: Some(etalon::BlendState {
                            color: etalon::BlendComponent {
                                operation: etalon::BlendOperation::Add,
                                src_factor: etalon::BlendFactor::SrcAlpha,
                                dst_factor: etalon::BlendFactor::OneMinusSrcAlpha,
                            },
                            alpha: etalon::BlendComponent::REPLACE,
                        }),
                        write_mask: etalon::ColorWrites::ALL,
                    })],
                }),
                primitive: etalon::PrimitiveState {
                    front_face: etalon::FrontFace::Ccw,
                    cull_mode: Some(etalon::Face::Back),
                    polygon_mode: etalon::PolygonMode::Line,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: etalon::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            Some(pipeline_wire)
        } else {
            None
        };

        // Done
        Example {
            vertex_buf,
            index_buf,
            index_count: index_data.len(),
            bind_group,
            uniform_buf,
            pipeline,
            pipeline_wire,
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        config: &etalon::SurfaceConfiguration,
        _device: &etalon::Device,
        queue: &etalon::Queue,
    ) {
        let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(mx_ref));
    }

    fn render(
        &mut self,
        view: &etalon::TextureView,
        device: &etalon::Device,
        queue: &etalon::Queue,
    ) {
        let mut encoder =
            device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
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
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.push_debug_group("Prepare data for draw.");
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_index_buffer(self.index_buf.slice(..), etalon::IndexFormat::Uint16);
            rpass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            rpass.pop_debug_group();
            rpass.insert_debug_marker("Draw!");
            rpass.draw_indexed(0..self.index_count as u32, 0, 0..1);
            if let Some(ref pipe) = self.pipeline_wire {
                rpass.set_pipeline(pipe);
                rpass.draw_indexed(0..self.index_count as u32, 0, 0..1);
            }
        }

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("cube");
}

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "cube",
    // Generated on 1080ti on Vk/Windows
    image_path: "/examples/features/src/cube/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::default(),
    base_test_parameters: etalon_test::TestParameters::default(),
    comparisons: &[
        etalon_test::ComparisonType::Mean(0.041), // Bounded by Apple A9
    ],
    _phantom: std::marker::PhantomData::<Example>,
};

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST_LINES: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "cube-lines",
    // Generated on 1080ti on Vk/Windows
    image_path: "/examples/features/src/cube/screenshot-lines.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::POLYGON_MODE_LINE,
    base_test_parameters: etalon_test::TestParameters::default(),
    // We're looking for tiny changes here, so we focus on a spike in the 95th percentile.
    comparisons: &[
        etalon_test::ComparisonType::Mean(0.05), // Bounded by Intel 630 on Vk/Windows
        etalon_test::ComparisonType::Percentile {
            percentile: 0.95,
            threshold: 0.36,
        }, // Bounded by 1080ti on DX12
    ],
    _phantom: std::marker::PhantomData::<Example>,
};
