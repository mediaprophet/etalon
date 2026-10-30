use bytemuck::{Pod, Zeroable};
use etalon::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 4],
    _color: [f32; 4],
}

fn vertex(pos: [i8; 2], _color: [f32; 4], offset: f32) -> Vertex {
    let scale = 0.5;
    Vertex {
        _pos: [
            (pos[0] as f32 + offset) * scale,
            (pos[1] as f32 + offset) * scale,
            0.0,
            1.0,
        ],
        _color,
    }
}

fn quad(vertices: &mut Vec<Vertex>, indices: &mut Vec<u16>, color: [f32; 4], offset: f32) {
    let base = vertices.len() as u16;

    vertices.extend_from_slice(&[
        vertex([-1, -1], color, offset),
        vertex([1, -1], color, offset),
        vertex([1, 1], color, offset),
        vertex([-1, 1], color, offset),
    ]);

    indices.extend([0, 1, 2, 2, 3, 0].iter().map(|i| base + *i));
}

fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let red = [1.0, 0.0, 0.0, 0.5];
    let blue = [0.0, 0.0, 1.0, 0.5];

    quad(&mut vertices, &mut indices, red, 0.5);
    quad(&mut vertices, &mut indices, blue, -0.5);

    (vertices, indices)
}

struct Example<const SRGB: bool> {
    vertex_buf: etalon::Buffer,
    index_buf: etalon::Buffer,
    index_count: usize,
    bind_group: etalon::BindGroup,
    pipeline: etalon::RenderPipeline,
}

impl<const SRGB: bool> crate::framework::Example for Example<SRGB> {
    const SRGB: bool = SRGB;

    fn optional_features() -> etalon::Features {
        etalon::Features::POLYGON_MODE_LINE
    }

    fn init(
        config: &etalon::SurfaceConfiguration,
        _adapter: &etalon::Adapter,
        device: &etalon::Device,
        _queue: &etalon::Queue,
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
                entries: &[],
            });
        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[],
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
                    format: etalon::VertexFormat::Float32x4,
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
                targets: &[Some(etalon::ColorTargetState {
                    format: config.view_formats[0],
                    blend: Some(etalon::BlendState::ALPHA_BLENDING),
                    write_mask: etalon::ColorWrites::ALL,
                })],
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

        // Done
        Example {
            vertex_buf,
            index_buf,
            index_count: index_data.len(),
            bind_group,
            pipeline,
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        _config: &etalon::SurfaceConfiguration,
        _device: &etalon::Device,
        _queue: &etalon::Queue,
    ) {
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
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
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
        }

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    let mut args = std::env::args();
    args.next();
    if Some("linear") == args.nth(1).as_deref() {
        crate::framework::run::<Example<false>>("srgb-blend-linear");
    } else {
        crate::framework::run::<Example<true>>("srgb-blend-srg");
    }
}

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST_SRGB: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "srgb-blend-srg",
    // Generated on WARP/Windows
    image_path: "/examples/features/src/srgb_blend/screenshot-srgb.png",
    width: 192,
    height: 192,
    optional_features: etalon::Features::default(),
    base_test_parameters: etalon_test::TestParameters::default(),
    comparisons: &[etalon_test::ComparisonType::Mean(0.04)],
    _phantom: std::marker::PhantomData::<Example<true>>,
};

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST_LINEAR: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "srgb-blend-linear",
    // Generated on WARP/Windows
    image_path: "/examples/features/src/srgb_blend/screenshot-linear.png",
    width: 192,
    height: 192,
    optional_features: etalon::Features::default(),
    base_test_parameters: etalon_test::TestParameters::default(),
    comparisons: &[etalon_test::ComparisonType::Mean(0.04)],
    _phantom: std::marker::PhantomData::<Example<false>>,
};
