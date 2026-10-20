use etalon::{
    util::{BufferInitDescriptor, DeviceExt},
    vertex_attr_array,
};
use etalon_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        DRAW_TO_2D_VIEW,
        DRAW_TO_2D_ARRAY_VIEW,
        RESOLVE_TO_2D_VIEW,
        RESOLVE_TO_2D_ARRAY_VIEW,
        DRAW_TO_3D_VIEW,
    ]);
}

#[apply(gpu_test!)]
static DRAW_TO_2D_VIEW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| run_test(ctx, etalon::TextureViewDimension::D2, false));

#[apply(gpu_test!)]
static DRAW_TO_2D_ARRAY_VIEW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| run_test(ctx, etalon::TextureViewDimension::D2Array, false));

#[apply(gpu_test!)]
static RESOLVE_TO_2D_VIEW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| run_test(ctx, etalon::TextureViewDimension::D2, true));

#[apply(gpu_test!)]
static RESOLVE_TO_2D_ARRAY_VIEW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| run_test(ctx, etalon::TextureViewDimension::D2Array, true));

async fn run_test(
    ctx: TestingContext,
    view_dimension: etalon::TextureViewDimension,
    multisample: bool,
) {
    let vertex_buffer_content: &[f32; 12] = &[
        // Triangle 1
        -1.0, -1.0, // Bottom left
        1.0, 1.0, // Top right
        -1.0, 1.0, // Top left
        // Triangle 2
        -1.0, -1.0, // Bottom left
        1.0, -1.0, // Bottom right
        1.0, 1.0, // Top right
    ];
    let vertex_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(vertex_buffer_content),
        usage: etalon::BufferUsages::VERTEX,
    });

    let shader_src = "
            @vertex
            fn vs_main(@location(0) position: vec2f) -> @builtin(position) vec4f {
                return vec4f(position, 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4f {
                return vec4f(1.0);
            }
        ";

    let shader = ctx
        .device
        .create_shader_module(etalon::ShaderModuleDescriptor {
            label: None,
            source: etalon::ShaderSource::Wgsl(shader_src.into()),
        });

    let pipeline_desc = etalon::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: etalon::VertexState {
            buffers: &[Some(etalon::VertexBufferLayout {
                array_stride: 8,
                step_mode: etalon::VertexStepMode::Vertex,
                attributes: &vertex_attr_array![0 => Float32x2],
            })],
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
        },
        primitive: etalon::PrimitiveState::default(),
        depth_stencil: None,
        multisample: etalon::MultisampleState {
            count: if multisample { 4 } else { 1 },
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(etalon::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(etalon::ColorTargetState {
                format: etalon::TextureFormat::R8Unorm,
                blend: None,
                write_mask: etalon::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    };
    let pipeline = ctx.device.create_render_pipeline(&pipeline_desc);

    const SIZE: u32 = 512;
    const LAYERS: u32 = 2;
    const MIPS: u32 = 2;
    const fn size_for_mips(mips: u32) -> u64 {
        let mut out: u64 = 0;
        let mut mip = 0;
        while mip < mips {
            let size = SIZE as u64 >> mip;
            out += size * size;

            mip += 1;
        }
        out * LAYERS as u64
    }

    let out_texture = ctx.device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size: etalon::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: LAYERS,
        },
        mip_level_count: MIPS,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::R8Unorm,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let readback_buffer = ctx.device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: size_for_mips(MIPS),
        usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

    for mip in 0..MIPS {
        let ms_texture_view = if multisample {
            let ms_texture = ctx.device.create_texture(&etalon::TextureDescriptor {
                label: None,
                size: etalon::Extent3d {
                    width: SIZE >> mip,
                    height: SIZE >> mip,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 4,
                dimension: etalon::TextureDimension::D2,
                format: etalon::TextureFormat::R8Unorm,
                usage: etalon::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let ms_texture_view = ms_texture.create_view(&etalon::TextureViewDescriptor::default());
            Some(ms_texture_view)
        } else {
            None
        };
        for layer in 0..LAYERS {
            let out_texture_view = out_texture.create_view(&etalon::TextureViewDescriptor {
                label: None,
                format: Some(etalon::TextureFormat::R8Unorm),
                dimension: Some(view_dimension),
                usage: Some(etalon::TextureUsages::RENDER_ATTACHMENT),
                aspect: etalon::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: layer,
                array_layer_count: Some(1),
                swizzle: etalon::TextureComponentSwizzle::default(),
            });
            let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(etalon::RenderPassColorAttachment {
                    view: ms_texture_view.as_ref().unwrap_or(&out_texture_view),
                    depth_slice: None,
                    resolve_target: multisample.then_some(&out_texture_view),
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(etalon::Color::BLACK),
                        store: if multisample {
                            etalon::StoreOp::Discard
                        } else {
                            etalon::StoreOp::Store
                        },
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&pipeline);
            rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
            rpass.draw(0..6, 0..1);
        }
    }

    for mip in 0..MIPS {
        encoder.copy_texture_to_buffer(
            etalon::TexelCopyTextureInfo {
                texture: &out_texture,
                mip_level: mip,
                origin: etalon::Origin3d::ZERO,
                aspect: etalon::TextureAspect::All,
            },
            etalon::TexelCopyBufferInfo {
                buffer: &readback_buffer,
                layout: etalon::TexelCopyBufferLayout {
                    offset: size_for_mips(mip),
                    bytes_per_row: Some(SIZE >> mip),
                    rows_per_image: Some(SIZE >> mip),
                },
            },
            etalon::Extent3d {
                width: SIZE >> mip,
                height: SIZE >> mip,
                depth_or_array_layers: LAYERS,
            },
        );
    }

    ctx.queue.submit([encoder.finish()]);

    let slice = readback_buffer.slice(..);
    slice.map_async(etalon::MapMode::Read, |_| ());

    ctx.async_poll(etalon::PollType::wait_indefinitely())
        .await
        .unwrap();

    let data = slice.get_mapped_range().unwrap();
    let succeeded = data.iter().all(|b| *b == u8::MAX);
    assert!(succeeded);
}

#[apply(gpu_test!)]
static DRAW_TO_3D_VIEW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .limits(etalon::Limits {
                max_texture_dimension_3d: 512,
                ..etalon::Limits::downlevel_webgl2_defaults()
            })
            // https://github.com/mediaprophet/etalon/issues/9184
            .expect_fail(
                etalon_test::FailureCase::molten_vk()
                    .validation_error("VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT"),
            ),
    )
    .run_async(run_test_3d);

async fn run_test_3d(ctx: TestingContext) {
    let vertex_buffer_content: &[f32; 12] = &[
        // Triangle 1
        -1.0, -1.0, // Bottom left
        1.0, 1.0, // Top right
        -1.0, 1.0, // Top left
        // Triangle 2
        -1.0, -1.0, // Bottom left
        1.0, -1.0, // Bottom right
        1.0, 1.0, // Top right
    ];
    let vertex_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(vertex_buffer_content),
        usage: etalon::BufferUsages::VERTEX,
    });

    let shader_src = "
            @vertex
            fn vs_main(@location(0) position: vec2f) -> @builtin(position) vec4f {
                return vec4f(position, 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4f {
                return vec4f(1.0);
            }
        ";

    let shader = ctx
        .device
        .create_shader_module(etalon::ShaderModuleDescriptor {
            label: None,
            source: etalon::ShaderSource::Wgsl(shader_src.into()),
        });

    let pipeline_desc = etalon::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: etalon::VertexState {
            buffers: &[Some(etalon::VertexBufferLayout {
                array_stride: 8,
                step_mode: etalon::VertexStepMode::Vertex,
                attributes: &vertex_attr_array![0 => Float32x2],
            })],
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
        },
        primitive: etalon::PrimitiveState::default(),
        depth_stencil: None,
        multisample: etalon::MultisampleState::default(),
        fragment: Some(etalon::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(etalon::ColorTargetState {
                format: etalon::TextureFormat::R8Unorm,
                blend: None,
                write_mask: etalon::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    };
    let pipeline = ctx.device.create_render_pipeline(&pipeline_desc);

    const SIZE: u32 = 512;
    const DEPTH: u32 = 2;
    const MIPS: u32 = 2;
    const fn size_for_mips(mips: u32) -> u64 {
        let mut out: u64 = 0;
        let mut mip = 0;
        while mip < mips {
            let size = SIZE as u64 >> mip;
            let z = DEPTH as u64 >> mip;
            out += size * size * z;

            mip += 1;
        }
        out
    }

    let out_texture = ctx.device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size: etalon::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: DEPTH,
        },
        mip_level_count: MIPS,
        sample_count: 1,
        dimension: etalon::TextureDimension::D3,
        format: etalon::TextureFormat::R8Unorm,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let readback_buffer = ctx.device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: size_for_mips(MIPS),
        usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

    for mip in 0..MIPS {
        let out_texture_view = out_texture.create_view(&etalon::TextureViewDescriptor {
            base_mip_level: mip,
            mip_level_count: Some(1),
            ..Default::default()
        });
        for layer in 0..DEPTH >> mip {
            let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(etalon::RenderPassColorAttachment {
                    view: &out_texture_view,
                    depth_slice: Some(layer),
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
            rpass.set_pipeline(&pipeline);
            rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
            rpass.draw(0..6, 0..1);
        }
    }

    for mip in 0..MIPS {
        encoder.copy_texture_to_buffer(
            etalon::TexelCopyTextureInfo {
                texture: &out_texture,
                mip_level: mip,
                origin: etalon::Origin3d::ZERO,
                aspect: etalon::TextureAspect::All,
            },
            etalon::TexelCopyBufferInfo {
                buffer: &readback_buffer,
                layout: etalon::TexelCopyBufferLayout {
                    offset: size_for_mips(mip),
                    bytes_per_row: Some(SIZE >> mip),
                    rows_per_image: Some(SIZE >> mip),
                },
            },
            etalon::Extent3d {
                width: SIZE >> mip,
                height: SIZE >> mip,
                depth_or_array_layers: DEPTH >> mip,
            },
        );
    }

    ctx.queue.submit([encoder.finish()]);

    let slice = readback_buffer.slice(..);
    slice.map_async(etalon::MapMode::Read, |_| ());

    ctx.async_poll(etalon::PollType::wait_indefinitely())
        .await
        .unwrap();

    let data = slice.get_mapped_range().unwrap();
    let succeeded = data.iter().all(|b| *b == u8::MAX);
    assert!(succeeded);
}
