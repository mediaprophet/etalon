use crate::utils;
use bytemuck::{Pod, Zeroable};
use etalon::util::DeviceExt;
use glam::{Mat4, Quat, Vec3};
use std::ops::IndexMut;
use std::{borrow::Cow, iter, mem};

// from cube
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_inverse: Mat4,
    proj_inverse: Mat4,
}

struct Example {
    uniforms: Uniforms,
    uniform_buf: etalon::Buffer,
    blas: etalon::Blas,
    tlas: etalon::Tlas,
    pipeline: etalon::RenderPipeline,
    bind_group: etalon::BindGroup,
    animation_timer: utils::AnimationTimer,
}

impl crate::framework::Example for Example {
    fn required_features() -> etalon::Features {
        etalon::Features::EXPERIMENTAL_RAY_QUERY
    }

    fn required_downlevel_capabilities() -> etalon::DownlevelCapabilities {
        etalon::DownlevelCapabilities {
            flags: etalon::DownlevelFlags::COMPUTE_SHADERS,
            ..Default::default()
        }
    }

    fn required_limits() -> etalon::Limits {
        etalon::Limits::default().using_minimum_supported_acceleration_structure_values()
    }

    fn init(
        config: &etalon::SurfaceConfiguration,
        _adapter: &etalon::Adapter,
        device: &etalon::Device,
        queue: &etalon::Queue,
    ) -> Self {
        let side_count = 8;

        let uniforms = {
            let view =
                glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 0.0, 2.5), Vec3::ZERO, Vec3::Y);
            let proj = glam::camera::rh::proj::directx::perspective(
                59.0_f32.to_radians(),
                config.width as f32 / config.height as f32,
                0.001,
                1000.0,
            );

            Uniforms {
                view_inverse: view.inverse(),
                proj_inverse: proj.inverse(),
            }
        };

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

        let tlas = device.create_tlas(&etalon::CreateTlasDescriptor {
            label: None,
            flags: etalon::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: etalon::AccelerationStructureUpdateMode::Build,
            max_instances: side_count * side_count,
        });

        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: None,
            source: etalon::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: None,
            layout: None,
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

        let bind_group_layout = pipeline.get_bind_group_layout(0);

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
                    resource: etalon::BindingResource::AccelerationStructure(&tlas),
                },
            ],
        });

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
            // iter::empty(),
            iter::once(&tlas),
        );

        queue.submit(Some(encoder.finish()));

        Example {
            uniforms,
            uniform_buf,
            blas,
            tlas,
            pipeline,
            bind_group,
            animation_timer: utils::AnimationTimer::default(),
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {}

    fn resize(
        &mut self,
        config: &etalon::SurfaceConfiguration,
        _device: &etalon::Device,
        queue: &etalon::Queue,
    ) {
        let proj = glam::camera::rh::proj::directx::perspective(
            59.0_f32.to_radians(),
            config.width as f32 / config.height as f32,
            0.001,
            1000.0,
        );

        self.uniforms.proj_inverse = proj.inverse();

        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&[self.uniforms]));
    }

    fn render(
        &mut self,
        view: &etalon::TextureView,
        device: &etalon::Device,
        queue: &etalon::Queue,
    ) {
        // scene update
        {
            let dist = 12.0;

            let side_count = 8;

            let anim_time = self.animation_timer.time();

            for x in 0..side_count {
                for y in 0..side_count {
                    let instance = self.tlas.index_mut((x + y * side_count) as usize);

                    let x = x as f32 / (side_count - 1) as f32;
                    let y = y as f32 / (side_count - 1) as f32;
                    let x = x * 2.0 - 1.0;
                    let y = y * 2.0 - 1.0;

                    let transform = Mat4::from_rotation_translation(
                        Quat::from_euler(
                            glam::EulerRot::XYZ,
                            anim_time * 0.5 * 0.342,
                            anim_time * 0.5 * 0.254,
                            anim_time * 0.5 * 0.832,
                        ),
                        Vec3 {
                            x: x * dist,
                            y: y * dist,
                            z: -24.0,
                        },
                    );
                    let transform = transform.transpose().to_cols_array()[..12]
                        .try_into()
                        .unwrap();

                    *instance = Some(etalon::TlasInstance::new(&self.blas, transform, 0, 0xff));
                }
            }
        }

        let mut encoder =
            device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });

        encoder.build_acceleration_structures(iter::empty(), iter::once(&self.tlas));

        {
            let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(etalon::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(etalon::Color::GREEN),
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
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("ray-cube");
}

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "ray_cube_fragment",
    image_path: "/examples/features/src/ray_cube_fragment/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::default(),
    base_test_parameters: etalon_test::TestParameters::default()
        // https://github.com/mediaprophet/etalon/issues/9100
        .disable_mtl_shader_validation(),
    comparisons: &[etalon_test::ComparisonType::Mean(0.02)],
    _phantom: std::marker::PhantomData::<Example>,
};
