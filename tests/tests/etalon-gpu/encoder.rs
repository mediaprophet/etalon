use etalon::util::DeviceExt;
use etalon::CommandEncoder;
use etalon_test::{
    apply, fail, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        DROP_ENCODER,
        DROP_QUEUE_BEFORE_CREATING_COMMAND_ENCODER,
        DROP_ENCODER_AFTER_ERROR,
        ENCODER_OPERATIONS_FAIL_WHILE_PASS_ALIVE,
    ]);
}

#[apply(gpu_test!)]
static DROP_ENCODER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        let encoder = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor::default());
        drop(encoder);
    });

#[apply(gpu_test!)]
static DROP_QUEUE_BEFORE_CREATING_COMMAND_ENCODER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .expect_fail(FailureCase::always())
                .enable_noop(),
        )
        .run_sync(|ctx| {
            // Use the device after the queue is dropped. Currently this panics
            // but it probably shouldn't.
            // TODO(https://github.com/mediaprophet/etalon/issues/7781) revisit this
            let TestingContext { device, queue, .. } = ctx;
            drop(queue);
            let _encoder =
                device.create_command_encoder(&etalon::CommandEncoderDescriptor::default());
        });

#[apply(gpu_test!)]
static DROP_ENCODER_AFTER_ERROR: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        let mut encoder = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

        let target_tex = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: etalon::Extent3d {
                width: 100,
                height: 100,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::R8Unorm,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target_tex.create_view(&etalon::TextureViewDescriptor::default());

        let mut renderpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: Some("renderpass"),
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                ops: etalon::Operations::default(),
                resolve_target: None,
                view: &target_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // This viewport is invalid because it has negative size.
        renderpass.set_viewport(0.0, 0.0, -1.0, -1.0, 0.0, 1.0);
        drop(renderpass);

        fail(&ctx.device, || encoder.finish(), Some("less than zero"));
    });

#[apply(gpu_test!)]
static ENCODER_OPERATIONS_FAIL_WHILE_PASS_ALIVE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(etalon::Features::CLEAR_TEXTURE | etalon::Features::TIMESTAMP_QUERY)
            .enable_noop(),
    )
    .run_sync(encoder_operations_fail_while_pass_alive);

fn encoder_operations_fail_while_pass_alive(ctx: TestingContext) {
    let buffer_source = ctx
        .device
        .create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: None,
            contents: &[0u8; 4],
            usage: etalon::BufferUsages::COPY_SRC,
        });
    let buffer_dest = ctx
        .device
        .create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: None,
            contents: &[0u8; 4],
            usage: etalon::BufferUsages::COPY_DST,
        });

    let texture_desc = etalon::TextureDescriptor {
        label: None,
        size: etalon::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Rgba8Unorm,
        usage: etalon::TextureUsages::COPY_DST,
        view_formats: &[],
    };
    let texture_dst = ctx.device.create_texture(&texture_desc);
    let texture_src = ctx.device.create_texture(&etalon::TextureDescriptor {
        usage: etalon::TextureUsages::COPY_SRC,
        ..texture_desc
    });
    let query_set = ctx.device.create_query_set(&etalon::QuerySetDescriptor {
        count: 1,
        ty: etalon::QueryType::Timestamp,
        label: None,
    });

    let target_desc = etalon::TextureDescriptor {
        label: Some("target_tex"),
        size: etalon::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Bgra8UnormSrgb,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[etalon::TextureFormat::Bgra8UnormSrgb],
    };
    let target_tex = ctx.device.create_texture(&target_desc);
    let color_attachment_view = target_tex.create_view(&etalon::TextureViewDescriptor::default());

    #[allow(clippy::type_complexity)]
    let recording_ops: Vec<(_, Box<dyn Fn(&mut CommandEncoder)>)> = vec![
        (
            "begin_compute_pass",
            Box::new(|encoder: &mut etalon::CommandEncoder| {
                encoder.begin_compute_pass(&etalon::ComputePassDescriptor::default());
            }),
        ),
        (
            "begin_render_pass",
            Box::new(|encoder: &mut etalon::CommandEncoder| {
                encoder.begin_render_pass(&etalon::RenderPassDescriptor::default());
            }),
        ),
        (
            "copy_buffer_to_buffer",
            Box::new(|encoder: &mut etalon::CommandEncoder| {
                encoder.copy_buffer_to_buffer(&buffer_source, 0, &buffer_dest, 0, 4);
            }),
        ),
        (
            "copy_buffer_to_texture",
            Box::new(|encoder: &mut etalon::CommandEncoder| {
                encoder.copy_buffer_to_texture(
                    etalon::TexelCopyBufferInfo {
                        buffer: &buffer_source,
                        layout: etalon::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(4),
                            rows_per_image: None,
                        },
                    },
                    texture_dst.as_image_copy(),
                    texture_dst.size(),
                );
            }),
        ),
        (
            "copy_texture_to_buffer",
            Box::new(|encoder: &mut etalon::CommandEncoder| {
                encoder.copy_texture_to_buffer(
                    etalon::TexelCopyTextureInfo {
                        texture: &texture_src,
                        mip_level: 0,
                        origin: etalon::Origin3d::ZERO,
                        aspect: etalon::TextureAspect::All,
                    },
                    etalon::TexelCopyBufferInfo {
                        buffer: &buffer_dest,
                        layout: etalon::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(4),
                            rows_per_image: None,
                        },
                    },
                    texture_dst.size(),
                );
            }),
        ),
        (
            "copy_texture_to_texture",
            Box::new(|encoder: &mut etalon::CommandEncoder| {
                encoder.copy_texture_to_texture(
                    etalon::TexelCopyTextureInfo {
                        texture: &texture_src,
                        mip_level: 0,
                        origin: etalon::Origin3d::ZERO,
                        aspect: etalon::TextureAspect::All,
                    },
                    etalon::TexelCopyTextureInfo {
                        texture: &texture_dst,
                        mip_level: 0,
                        origin: etalon::Origin3d::ZERO,
                        aspect: etalon::TextureAspect::All,
                    },
                    texture_dst.size(),
                );
            }),
        ),
        (
            "clear_texture",
            Box::new(|encoder: &mut etalon::CommandEncoder| {
                encoder.clear_texture(&texture_dst, &etalon::ImageSubresourceRange::default());
            }),
        ),
        (
            "clear_buffer",
            Box::new(|encoder: &mut etalon::CommandEncoder| {
                encoder.clear_buffer(&buffer_dest, 0, None);
            }),
        ),
        (
            "insert_debug_marker",
            Box::new(|encoder: &mut etalon::CommandEncoder| {
                encoder.insert_debug_marker("marker");
            }),
        ),
        (
            "push_debug_group",
            Box::new(|encoder: &mut etalon::CommandEncoder| {
                encoder.push_debug_group("marker");
            }),
        ),
        (
            "pop_debug_group",
            Box::new(|encoder: &mut etalon::CommandEncoder| {
                encoder.pop_debug_group();
            }),
        ),
        (
            "resolve_query_set",
            Box::new(|encoder: &mut etalon::CommandEncoder| {
                encoder.resolve_query_set(&query_set, 0..1, &buffer_dest, 0);
            }),
        ),
        (
            "write_timestamp",
            Box::new(|encoder: &mut etalon::CommandEncoder| {
                encoder.write_timestamp(&query_set, 0);
            }),
        ),
    ];

    #[derive(Clone, Copy, Debug)]
    enum PassType {
        Compute,
        Render,
    }

    let create_pass = |encoder: &mut etalon::CommandEncoder, pass_type| -> Box<dyn std::any::Any> {
        match pass_type {
            PassType::Compute => Box::new(
                encoder
                    .begin_compute_pass(&etalon::ComputePassDescriptor::default())
                    .forget_lifetime(),
            ),
            PassType::Render => Box::new(
                encoder
                    .begin_render_pass(&etalon::RenderPassDescriptor {
                        color_attachments: &[Some(etalon::RenderPassColorAttachment {
                            view: &color_attachment_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: etalon::Operations::default(),
                        })],
                        ..Default::default()
                    })
                    .forget_lifetime(),
            ),
        }
    };

    for &pass_type in [PassType::Compute, PassType::Render].iter() {
        for (op_name, op) in recording_ops.iter() {
            // Test the case where the pass is not ended before calling finish()
            let mut encoder = ctx
                .device
                .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

            let pass = create_pass(&mut encoder, pass_type);

            let _scope = ctx.device.push_error_scope(etalon::ErrorFilter::Validation);

            log::info!("Testing operation {op_name:?} on a locked command encoder while a {pass_type:?} pass is active");
            op(&mut encoder);

            fail(&ctx.device, || encoder.finish(), Some("encoder is locked"));

            drop(pass);

            // ...and the case where the pass is ended before calling finish()
            let mut encoder = ctx
                .device
                .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

            let pass = create_pass(&mut encoder, pass_type);

            log::info!("Testing operation {op_name:?} on a locked command encoder while a {pass_type:?} pass is active");
            op(&mut encoder);

            drop(pass);

            fail(&ctx.device, || encoder.finish(), Some("encoder is locked"));

            // We don't care about any errors that happen outside of a `fail` call.
            drop(_scope);
        }
    }
}
