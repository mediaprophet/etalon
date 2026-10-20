use glam::{Mat4, Vec3};
use std::mem;
use etalon::util::{BufferInitDescriptor, DeviceExt};
use etalon::{include_wgsl, BufferUsages, IndexFormat, SamplerDescriptor};
use etalon::{
    AccelerationStructureFlags, AccelerationStructureUpdateMode, BlasBuildEntry, BlasGeometries,
    BlasGeometrySizeDescriptors, BlasTriangleGeometry, BlasTriangleGeometrySizeDescriptor,
    CreateBlasDescriptor, CreateTlasDescriptor, Tlas, TlasInstance,
};

use crate::utils;

struct Example {
    tlas: Tlas,
    compute_pipeline: etalon::ComputePipeline,
    blit_pipeline: etalon::RenderPipeline,
    bind_group: etalon::BindGroup,
    blit_bind_group: etalon::BindGroup,
    storage_texture: etalon::Texture,
    animation_timer: utils::AnimationTimer,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug)]
struct Uniforms {
    view_inverse: Mat4,
    proj_inverse: Mat4,
}

impl crate::framework::Example for Example {
    fn required_features() -> etalon::Features {
        etalon::Features::EXPERIMENTAL_RAY_QUERY
    }

    fn required_limits() -> etalon::Limits {
        etalon::Limits::default().using_minimum_supported_acceleration_structure_values()
    }

    fn required_downlevel_capabilities() -> etalon::DownlevelCapabilities {
        etalon::DownlevelCapabilities {
            flags: etalon::DownlevelFlags::COMPUTE_SHADERS,
            ..Default::default()
        }
    }

    fn init(
        config: &etalon::SurfaceConfiguration,
        _adapter: &etalon::Adapter,
        device: &etalon::Device,
        queue: &etalon::Queue,
    ) -> Self {
        let shader = device.create_shader_module(include_wgsl!("shader.wgsl"));

        let blit_shader = device.create_shader_module(include_wgsl!("blit.wgsl"));

        let bgl = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("bgl for shader.wgsl"),
            entries: &[
                etalon::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::StorageTexture {
                        access: etalon::StorageTextureAccess::WriteOnly,
                        format: etalon::TextureFormat::Rgba8Unorm,
                        view_dimension: etalon::TextureViewDimension::D2,
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                    count: None,
                },
            ],
        });

        let blit_bgl = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("bgl for blit.wgsl"),
            entries: &[
                etalon::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: etalon::ShaderStages::VERTEX_FRAGMENT,
                    ty: etalon::BindingType::Texture {
                        sample_type: etalon::TextureSampleType::Float { filterable: false },
                        view_dimension: etalon::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: etalon::ShaderStages::VERTEX_FRAGMENT,
                    ty: etalon::BindingType::Sampler(etalon::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let vertices: [f32; 9] = [1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 0.0];

        let indices: [u32; 3] = [0, 1, 2];

        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vertex buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::BLAS_INPUT,
        });

        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("index buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::BLAS_INPUT,
        });

        let blas_size_desc = BlasTriangleGeometrySizeDescriptor {
            vertex_format: etalon::VertexFormat::Float32x3,
            // 3 coordinates per vertex
            vertex_count: (vertices.len() / 3) as u32,
            index_format: Some(IndexFormat::Uint32),
            index_count: Some(indices.len() as u32),
            flags: etalon::AccelerationStructureGeometryFlags::OPAQUE,
        };

        let blas = device.create_blas(
            &CreateBlasDescriptor {
                label: None,
                flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: AccelerationStructureUpdateMode::Build,
            },
            BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![blas_size_desc.clone()],
            },
        );

        let mut tlas = device.create_tlas(&CreateTlasDescriptor {
            label: None,
            max_instances: 3,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        });

        tlas[0] = Some(TlasInstance::new(
            &blas,
            Mat4::from_translation(Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            })
            .transpose()
            .to_cols_array()[..12]
                .try_into()
                .unwrap(),
            0,
            0xff,
        ));

        tlas[1] = Some(TlasInstance::new(
            &blas,
            Mat4::from_translation(Vec3 {
                x: -1.0,
                y: -1.0,
                z: -2.0,
            })
            .transpose()
            .to_cols_array()[..12]
                .try_into()
                .unwrap(),
            0,
            0xff,
        ));

        tlas[2] = Some(TlasInstance::new(
            &blas,
            Mat4::from_translation(Vec3 {
                x: 1.0,
                y: -1.0,
                z: -2.0,
            })
            .transpose()
            .to_cols_array()[..12]
                .try_into()
                .unwrap(),
            0,
            0xff,
        ));

        let uniforms = {
            let view =
                glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 0.0, 2.5), Vec3::ZERO, Vec3::Y);
            let proj = glam::camera::rh::proj::directx::perspective(
                59.0_f32.to_radians(),
                1.0,
                0.001,
                1000.0,
            );

            Uniforms {
                view_inverse: view.inverse(),
                proj_inverse: proj.inverse(),
            }
        };

        let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: BufferUsages::UNIFORM,
        });

        let mut encoder = device.create_command_encoder(&etalon::CommandEncoderDescriptor::default());

        encoder.build_acceleration_structures(
            Some(&BlasBuildEntry {
                blas: &blas,
                geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                    size: &blas_size_desc,
                    vertex_buffer: &vertex_buffer,
                    first_vertex: 0,
                    vertex_stride: mem::size_of::<[f32; 3]>() as etalon::BufferAddress,
                    // in this case since one triangle gets no compression from an index buffer `index_buffer` and `first_index` could be `None`.
                    index_buffer: Some(&index_buffer),
                    first_index: Some(0),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            }),
            Some(&tlas),
        );

        queue.submit(Some(encoder.finish()));

        let storage_tex = device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: etalon::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rgba8Unorm,
            usage: etalon::TextureUsages::STORAGE_BINDING | etalon::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: None,
            address_mode_u: Default::default(),
            address_mode_v: Default::default(),
            address_mode_w: Default::default(),
            mag_filter: etalon::FilterMode::Nearest,
            min_filter: etalon::FilterMode::Nearest,
            mipmap_filter: etalon::MipmapFilterMode::Nearest,
            lod_min_clamp: 1.0,
            lod_max_clamp: 1.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
                label: Some("pipeline layout for shader.wgsl"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

        let compute_pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("pipeline for shader.wgsl"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("pipeline layout for blit.wgsl"),
            bind_group_layouts: &[Some(&blit_bgl)],
            immediate_size: 0,
        });

        let blit_pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("pipeline for blit.wgsl"),
            layout: Some(&blit_pipeline_layout),
            vertex: etalon::VertexState {
                module: &blit_shader,
                entry_point: None,
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(etalon::FragmentState {
                module: &blit_shader,
                entry_point: None,
                compilation_options: Default::default(),
                targets: &[Some(etalon::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: Default::default(),
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("bind group for shader.wgsl"),
            layout: &bgl,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: etalon::BindingResource::TextureView(
                        &storage_tex.create_view(&etalon::TextureViewDescriptor::default()),
                    ),
                },
                etalon::BindGroupEntry {
                    binding: 2,
                    resource: etalon::BindingResource::AccelerationStructure(&tlas),
                },
            ],
        });

        let blit_bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("bind group for blit.wgsl"),
            layout: &blit_bgl,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: etalon::BindingResource::TextureView(
                        &storage_tex.create_view(&etalon::TextureViewDescriptor::default()),
                    ),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: etalon::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            tlas,
            compute_pipeline,
            blit_pipeline,
            bind_group,
            blit_bind_group,
            storage_texture: storage_tex,
            animation_timer: utils::AnimationTimer::default(),
        }
    }

    fn resize(
        &mut self,
        _config: &etalon::SurfaceConfiguration,
        _device: &etalon::Device,
        _queue: &etalon::Queue,
    ) {
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {}

    fn render(&mut self, view: &etalon::TextureView, device: &etalon::Device, queue: &etalon::Queue) {
        self.tlas[0].as_mut().unwrap().transform =
            Mat4::from_rotation_y(self.animation_timer.time())
                .transpose()
                .to_cols_array()[..12]
                .try_into()
                .unwrap();

        let mut encoder =
            device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });

        encoder.build_acceleration_structures(None, Some(&self.tlas));

        {
            let mut cpass = encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, Some(&self.bind_group), &[]);
            cpass.dispatch_workgroups(
                self.storage_texture.width() / 8,
                self.storage_texture.height() / 8,
                1,
            );
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
                        store: etalon::StoreOp::Store,
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
    crate::framework::run::<Example>("ray-traced-triangle");
}

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "ray_traced_triangle",
    image_path: "/examples/features/src/ray_traced_triangle/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::default(),
    base_test_parameters: etalon_test::TestParameters::default()
        // https://github.com/mediaprophet/etalon/issues/9100
        .disable_mtl_shader_validation(),
    comparisons: &[etalon_test::ComparisonType::Mean(0.02)],
    _phantom: std::marker::PhantomData::<Example>,
};
