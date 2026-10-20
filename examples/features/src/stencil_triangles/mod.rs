use bytemuck::{Pod, Zeroable};
use etalon::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 4],
}

fn vertex(x: f32, y: f32) -> Vertex {
    Vertex {
        _pos: [x, y, 0.0, 1.0],
    }
}

struct Example {
    outer_vertex_buffer: etalon::Buffer,
    mask_vertex_buffer: etalon::Buffer,
    outer_pipeline: etalon::RenderPipeline,
    mask_pipeline: etalon::RenderPipeline,
    stencil_buffer: etalon::Texture,
}

impl crate::framework::Example for Example {
    fn init(
        config: &etalon::SurfaceConfiguration,
        _adapter: &etalon::Adapter,
        device: &etalon::Device,
        _queue: &etalon::Queue,
    ) -> Self {
        // Create the vertex and index buffers
        let vertex_size = size_of::<Vertex>();
        let outer_vertices = [vertex(-1.0, -1.0), vertex(1.0, -1.0), vertex(0.0, 1.0)];
        let mask_vertices = [vertex(-0.5, 0.0), vertex(0.0, -1.0), vertex(0.5, 0.0)];

        let outer_vertex_buffer = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Outer Vertex Buffer"),
            contents: bytemuck::cast_slice(&outer_vertices),
            usage: etalon::BufferUsages::VERTEX,
        });

        let mask_vertex_buffer = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Mask Vertex Buffer"),
            contents: bytemuck::cast_slice(&mask_vertices),
            usage: etalon::BufferUsages::VERTEX,
        });

        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(etalon::include_wgsl!("shader.wgsl"));

        let vertex_buffers = [Some(etalon::VertexBufferLayout {
            array_stride: vertex_size as etalon::BufferAddress,
            step_mode: etalon::VertexStepMode::Vertex,
            attributes: &[etalon::VertexAttribute {
                format: etalon::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 0,
            }],
        })];

        let mask_pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
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
                    blend: None,
                    write_mask: etalon::ColorWrites::empty(),
                })],
            }),
            primitive: Default::default(),
            depth_stencil: Some(etalon::DepthStencilState::stencil(
                etalon::TextureFormat::Stencil8,
                etalon::StencilState {
                    front: etalon::StencilFaceState {
                        compare: etalon::CompareFunction::Always,
                        pass_op: etalon::StencilOperation::Replace,
                        ..Default::default()
                    },
                    back: etalon::StencilFaceState::IGNORE,
                    read_mask: !0,
                    write_mask: !0,
                },
            )),
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let outer_pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
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
            primitive: Default::default(),
            depth_stencil: Some(etalon::DepthStencilState::stencil(
                etalon::TextureFormat::Stencil8,
                etalon::StencilState {
                    front: etalon::StencilFaceState {
                        compare: etalon::CompareFunction::Greater,
                        ..Default::default()
                    },
                    back: etalon::StencilFaceState::IGNORE,
                    read_mask: !0,
                    write_mask: !0,
                },
            )),
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let stencil_buffer = device.create_texture(&etalon::TextureDescriptor {
            label: Some("Stencil buffer"),
            size: etalon::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Stencil8,
            view_formats: &[],
            usage: etalon::TextureUsages::RENDER_ATTACHMENT,
        });

        // Done
        Example {
            outer_vertex_buffer,
            mask_vertex_buffer,
            outer_pipeline,
            mask_pipeline,
            stencil_buffer,
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        // empty
    }

    fn resize(
        &mut self,
        config: &etalon::SurfaceConfiguration,
        device: &etalon::Device,
        _queue: &etalon::Queue,
    ) {
        self.stencil_buffer = device.create_texture(&etalon::TextureDescriptor {
            label: Some("Stencil buffer"),
            size: etalon::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Stencil8,
            view_formats: &[],
            usage: etalon::TextureUsages::RENDER_ATTACHMENT,
        });
    }

    fn render(&mut self, view: &etalon::TextureView, device: &etalon::Device, queue: &etalon::Queue) {
        let mut encoder =
            device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
        {
            let depth_view = self.stencil_buffer.create_view(&Default::default());
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
                    view: &depth_view,
                    depth_ops: None,
                    stencil_ops: Some(etalon::Operations {
                        load: etalon::LoadOp::Clear(0),
                        store: etalon::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_stencil_reference(1);

            rpass.set_pipeline(&self.mask_pipeline);
            rpass.set_vertex_buffer(0, self.mask_vertex_buffer.slice(..));
            rpass.draw(0..3, 0..1);

            rpass.set_pipeline(&self.outer_pipeline);
            rpass.set_vertex_buffer(0, self.outer_vertex_buffer.slice(..));
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("stencil-triangles");
}

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "stencil-triangles",
    image_path: "/examples/features/src/stencil_triangles/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::default(),
    base_test_parameters: etalon_test::TestParameters::default(),
    comparisons: &[etalon_test::ComparisonType::Mean(0.04)], // Bounded by Apple A9
    _phantom: std::marker::PhantomData::<Example>,
};
