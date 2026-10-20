use std::{borrow::Cow, iter, mem};

use bytemuck::{Pod, Zeroable};
use glam::{Affine3A, Mat4, Quat, Vec3};
use etalon::util::DeviceExt;

use crate::utils;
use etalon::StoreOp;

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

#[inline]
fn affine_to_rows(mat: &Affine3A) -> [f32; 12] {
    let row_0 = mat.matrix3.row(0);
    let row_1 = mat.matrix3.row(1);
    let row_2 = mat.matrix3.row(2);
    let translation = mat.translation;
    [
        row_0.x,
        row_0.y,
        row_0.z,
        translation.x,
        row_1.x,
        row_1.y,
        row_1.z,
        translation.y,
        row_2.x,
        row_2.y,
        row_2.z,
        translation.z,
    ]
}

struct Example {
    rt_target: etalon::Texture,
    tlas: etalon::Tlas,
    compute_pipeline: etalon::ComputePipeline,
    compute_bind_group: etalon::BindGroup,
    blit_pipeline: etalon::RenderPipeline,
    blit_bind_group: etalon::BindGroup,
    animation_timer: utils::AnimationTimer,
}

impl crate::framework::Example for Example {
    // Don't want srgb, so normals show up better.
    const SRGB: bool = false;
    fn required_features() -> etalon::Features {
        etalon::Features::EXPERIMENTAL_RAY_QUERY | etalon::Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN
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

        let rt_target = device.create_texture(&etalon::TextureDescriptor {
            label: Some("rt_target"),
            size: etalon::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rgba8Unorm,
            usage: etalon::TextureUsages::TEXTURE_BINDING | etalon::TextureUsages::STORAGE_BINDING,
            view_formats: &[etalon::TextureFormat::Rgba8Unorm],
        });

        let rt_view = rt_target.create_view(&etalon::TextureViewDescriptor {
            label: None,
            format: Some(etalon::TextureFormat::Rgba8Unorm),
            dimension: Some(etalon::TextureViewDimension::D2),
            usage: None,
            aspect: etalon::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            swizzle: etalon::TextureComponentSwizzle::default(),
        });

        let sampler = device.create_sampler(&etalon::SamplerDescriptor {
            label: Some("rt_sampler"),
            address_mode_u: etalon::AddressMode::ClampToEdge,
            address_mode_v: etalon::AddressMode::ClampToEdge,
            address_mode_w: etalon::AddressMode::ClampToEdge,
            mag_filter: etalon::FilterMode::Linear,
            min_filter: etalon::FilterMode::Linear,
            mipmap_filter: etalon::MipmapFilterMode::Nearest,
            ..Default::default()
        });

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
            usage: etalon::BufferUsages::UNIFORM,
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
                flags: etalon::AccelerationStructureFlags::PREFER_FAST_TRACE
                    | etalon::AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN,
                update_mode: etalon::AccelerationStructureUpdateMode::Build,
            },
            etalon::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![blas_geo_size_desc.clone()],
            },
        );

        let mut tlas = device.create_tlas(&etalon::CreateTlasDescriptor {
            label: None,
            flags: etalon::AccelerationStructureFlags::PREFER_FAST_TRACE
                | etalon::AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN,
            update_mode: etalon::AccelerationStructureUpdateMode::Build,
            max_instances: side_count * side_count,
        });

        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("rt_computer"),
            source: etalon::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let blit_shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("blit"),
            source: etalon::ShaderSource::Wgsl(Cow::Borrowed(include_str!("blit.wgsl"))),
        });

        let compute_pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("rt"),
            layout: None,
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });

        let compute_bind_group_layout = compute_pipeline.get_bind_group_layout(0);

        let compute_bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: None,
            layout: &compute_bind_group_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: etalon::BindingResource::TextureView(&rt_view),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 2,
                    resource: etalon::BindingResource::AccelerationStructure(&tlas),
                },
            ],
        });

        let blit_pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: None,
            vertex: etalon::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(etalon::FragmentState {
                module: &blit_shader,
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

        let blit_bind_group_layout = blit_pipeline.get_bind_group_layout(0);

        let blit_bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: None,
            layout: &blit_bind_group_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: etalon::BindingResource::TextureView(&rt_view),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: etalon::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let dist = 3.0;

        for x in 0..side_count {
            for y in 0..side_count {
                tlas[(x + y * side_count) as usize] = Some(etalon::TlasInstance::new(
                    &blas,
                    affine_to_rows(&Affine3A::from_rotation_translation(
                        Quat::from_rotation_y(45.9_f32.to_radians()),
                        Vec3 {
                            x: x as f32 * dist,
                            y: y as f32 * dist,
                            z: -30.0,
                        },
                    )),
                    0,
                    0xff,
                ));
            }
        }

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

        Example {
            rt_target,
            tlas,
            compute_pipeline,
            compute_bind_group,
            blit_pipeline,
            blit_bind_group,
            animation_timer: utils::AnimationTimer::default(),
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

    fn render(&mut self, view: &etalon::TextureView, device: &etalon::Device, queue: &etalon::Queue) {
        let anim_time = self.animation_timer.time();

        self.tlas[0].as_mut().unwrap().transform =
            affine_to_rows(&Affine3A::from_rotation_translation(
                Quat::from_euler(
                    glam::EulerRot::XYZ,
                    anim_time * 0.342,
                    anim_time * 0.254,
                    anim_time * 0.832,
                ),
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: -6.0,
                },
            ));

        let mut encoder =
            device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });

        encoder.build_acceleration_structures(iter::empty(), iter::once(&self.tlas));

        {
            let mut cpass = encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, Some(&self.compute_bind_group), &[]);
            cpass.dispatch_workgroups(self.rt_target.width() / 8, self.rt_target.height() / 8, 1);
        }

        {
            let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(etalon::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(etalon::Color::GREEN),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&self.blit_pipeline);
            rpass.set_bind_group(0, Some(&self.blit_bind_group), &[]);
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
    name: "ray_cube_normals",
    image_path: "/examples/features/src/ray_cube_normals/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::default(),
    base_test_parameters: etalon_test::TestParameters::default().expect_fail(
        // RADV does this fine.
        etalon_test::FailureCase {
            backends: Some(etalon::Backends::VULKAN),
            adapter: Some("AMD"),
            driver: Some("AMD proprietary driver"),
            ..etalon_test::FailureCase::default()
        }
        .panic("Image data mismatch"),
    ),
    comparisons: &[etalon_test::ComparisonType::Mean(0.02)],
    _phantom: std::marker::PhantomData::<Example>,
};
