use std::{borrow::Cow, iter, mem};

use bytemuck::{Pod, Zeroable};
use etalon::util::DeviceExt;
use etalon::{vertex_attr_array, IndexFormat, VertexBufferLayout};
use glam::{Mat4, Vec3};

use crate::utils;

// from cube
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 3],
    _normal: [f32; 3],
}

fn vertex(pos: [f32; 3], normal: [f32; 3]) -> Vertex {
    Vertex {
        _pos: pos,
        _normal: normal,
    }
}

fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // base
        vertex([-1.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        vertex([-1.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        vertex([1.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        vertex([1.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        //shadow caster
        vertex([-(1.0 / 3.0), 0.0, 1.0], [0.0, 0.0, 1.0]),
        vertex([-(1.0 / 3.0), 2.0 / 3.0, 1.0], [0.0, 0.0, 1.0]),
        vertex([1.0 / 3.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
        vertex([1.0 / 3.0, 2.0 / 3.0, 1.0], [0.0, 0.0, 1.0]),
    ];

    let index_data: &[u16] = &[
        0, 1, 2, 2, 3, 1, //base
        4, 5, 6, 6, 7, 5,
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_inverse: Mat4,
    proj_inverse: Mat4,
    vertex: Mat4,
}

struct Example {
    uniforms: Uniforms,
    uniform_buf: etalon::Buffer,
    vertex_buf: etalon::Buffer,
    index_buf: etalon::Buffer,
    pipeline: etalon::RenderPipeline,
    bind_group: etalon::BindGroup,
    animation_timer: utils::AnimationTimer,
}

const CAM_LOOK_AT: Vec3 = Vec3::new(0.0, 1.0, -1.5);

fn create_matrix(config: &etalon::SurfaceConfiguration) -> Uniforms {
    let view = glam::camera::rh::view::look_at_mat4(CAM_LOOK_AT, Vec3::ZERO, Vec3::Y);
    let proj = glam::camera::rh::proj::directx::perspective(
        59.0_f32.to_radians(),
        config.width as f32 / config.height as f32,
        0.1,
        1000.0,
    );

    Uniforms {
        view_inverse: view.inverse(),
        proj_inverse: proj.inverse(),
        vertex: (proj * view),
    }
}

impl crate::framework::Example for Example {
    fn required_features() -> etalon::Features {
        etalon::Features::EXPERIMENTAL_RAY_QUERY | etalon::Features::IMMEDIATES
    }

    fn required_downlevel_capabilities() -> etalon::DownlevelCapabilities {
        etalon::DownlevelCapabilities {
            flags: etalon::DownlevelFlags::COMPUTE_SHADERS,
            ..Default::default()
        }
    }

    fn required_limits() -> etalon::Limits {
        etalon::Limits {
            max_immediate_size: 16,
            ..etalon::Limits::default()
        }
        .using_minimum_supported_acceleration_structure_values()
    }

    fn init(
        config: &etalon::SurfaceConfiguration,
        _adapter: &etalon::Adapter,
        device: &etalon::Device,
        queue: &etalon::Queue,
    ) -> Self {
        let uniforms = create_matrix(config);

        let uniform_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
        });

        let (vertex_data, index_data) = create_vertices();

        let vertex_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: etalon::BufferUsages::VERTEX | etalon::BufferUsages::BLAS_INPUT,
        });

        let index_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&index_data),
            usage: etalon::BufferUsages::INDEX | etalon::BufferUsages::BLAS_INPUT,
        });

        let blas_geo_size_desc = etalon::BlasTriangleGeometrySizeDescriptor {
            vertex_format: etalon::VertexFormat::Float32x3,
            vertex_count: vertex_data.len() as u32,
            index_format: Some(etalon::IndexFormat::Uint16),
            index_count: Some(index_data.len() as u32),
            flags: etalon::AccelerationStructureGeometryFlags::OPAQUE,
        };

        let blas = device.create_blas(
            &etalon::CreateBlasDescriptor {
                label: None,
                flags: etalon::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: etalon::AccelerationStructureUpdateMode::Build,
            },
            etalon::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![blas_geo_size_desc.clone()],
            },
        );

        let mut tlas = device.create_tlas(&etalon::CreateTlasDescriptor {
            label: None,
            flags: etalon::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: etalon::AccelerationStructureUpdateMode::Build,
            max_instances: 1,
        });

        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: None,
            source: etalon::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    etalon::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: etalon::ShaderStages::VERTEX_FRAGMENT,
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
                        ty: etalon::BindingType::AccelerationStructure {
                            vertex_return: false,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 16,
        });

        let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: etalon::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(VertexBufferLayout {
                    array_stride: mem::size_of::<Vertex>() as etalon::BufferAddress,
                    step_mode: Default::default(),
                    attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                })],
            },
            fragment: Some(etalon::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(config.format.into())],
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

        tlas[0] = Some(etalon::TlasInstance::new(
            &blas,
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            0,
            0xFF,
        ));

        let mut encoder =
            device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });

        encoder.build_acceleration_structures(
            iter::once(&etalon::BlasBuildEntry {
                blas: &blas,
                geometry: etalon::BlasGeometries::TriangleGeometries(vec![
                    etalon::BlasTriangleGeometry {
                        size: &blas_geo_size_desc,
                        vertex_buffer: &vertex_buf,
                        first_vertex: 0,
                        vertex_stride: mem::size_of::<Vertex>() as u64,
                        index_buffer: Some(&index_buf),
                        first_index: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            }),
            iter::once(&tlas),
        );

        queue.submit(Some(encoder.finish()));

        let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: tlas.as_binding(),
                },
            ],
        });

        let animation_timer = utils::AnimationTimer::default();

        Example {
            uniforms,
            uniform_buf,
            vertex_buf,
            index_buf,
            pipeline,
            bind_group,
            animation_timer,
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {}

    fn resize(
        &mut self,
        config: &etalon::SurfaceConfiguration,
        _device: &etalon::Device,
        queue: &etalon::Queue,
    ) {
        self.uniforms = create_matrix(config);

        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&[self.uniforms]));
        queue.submit(None);
    }

    fn render(
        &mut self,
        view: &etalon::TextureView,
        device: &etalon::Device,
        queue: &etalon::Queue,
    ) {
        //device.push_error_scope(etalon::ErrorFilter::Validation);
        const LIGHT_DISTANCE: f32 = 5.0;
        const TIME_SCALE: f32 = -0.2;
        const INITIAL_TIME: f32 = 1.0;
        let time = self.animation_timer.time();
        let cos = (time * TIME_SCALE + INITIAL_TIME).cos() * LIGHT_DISTANCE;
        let sin = (time * TIME_SCALE + INITIAL_TIME).sin() * LIGHT_DISTANCE;

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
                            g: 0.1,
                            b: 0.1,
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

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, Some(&self.bind_group), &[]);
            rpass.set_immediates(0, &0.0_f32.to_ne_bytes());
            rpass.set_immediates(4, &cos.to_ne_bytes());
            rpass.set_immediates(8, &sin.to_ne_bytes());
            rpass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            rpass.set_index_buffer(self.index_buf.slice(..), IndexFormat::Uint16);
            rpass.draw_indexed(0..12, 0, 0..1);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(etalon::PollType::wait_indefinitely()).unwrap();
    }
}

pub fn main() {
    crate::framework::run::<Example>("ray-shadows");
}

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "ray_shadows",
    image_path: "/examples/features/src/ray_shadows/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::default(),
    base_test_parameters: etalon_test::TestParameters::default()
        // https://github.com/mediaprophet/etalon/issues/9100
        .disable_mtl_shader_validation(),
    comparisons: &[etalon_test::ComparisonType::Mean(0.02)],
    _phantom: std::marker::PhantomData::<Example>,
};
