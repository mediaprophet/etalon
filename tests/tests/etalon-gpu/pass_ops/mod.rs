use etalon_test::{
    apply, fail, gpu_test, image::ReadbackBuffers, GpuTestConfiguration, GpuTestInitializer,
    TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        DONT_CARE,
        DONT_CARE_COLOR_STRICT_WEBGPU_COMPLIANCE,
        DONT_CARE_DEPTH_STRICT_WEBGPU_COMPLIANCE,
        DONT_CARE_STENCIL_STRICT_WEBGPU_COMPLIANCE,
    ]);
}

#[apply(gpu_test!)]
static DONT_CARE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(dont_care);

async fn dont_care(ctx: TestingContext) {
    let shader_src = "
        const triangles = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));

        @vertex
        fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
            return vec4f(triangles[vertex_index], 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            return vec4f(127.0 / 255.0);
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
            buffers: &[],
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
                format: etalon::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: etalon::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    };
    let pipeline = ctx.device.create_render_pipeline(&pipeline_desc);

    let out_texture = ctx.device.create_texture(&etalon::TextureDescriptor {
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
        usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let readbacks = ReadbackBuffers::new(&ctx.device, &out_texture);

    let mut encoder = ctx
        .device
        .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

    let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(etalon::RenderPassColorAttachment {
            view: &out_texture.create_view(&etalon::TextureViewDescriptor::default()),
            depth_slice: None,
            resolve_target: None,
            ops: etalon::Operations {
                load: etalon::LoadOp::DontCare(unsafe { etalon::LoadOpDontCare::enabled() }),
                store: etalon::StoreOp::Store,
            },
        })],
        ..Default::default()
    });
    rpass.set_pipeline(&pipeline);
    rpass.draw(0..3, 0..1);

    drop(rpass);

    readbacks.copy_from(&ctx.device, &mut encoder, &out_texture);

    ctx.queue.submit([encoder.finish()]);

    // Assert that DONT_CARE load op was fully overridden by the draw.
    readbacks
        .assert_buffer_contents(&ctx, &[127, 127, 127, 127])
        .await;
}

#[apply(gpu_test!)]
static DONT_CARE_COLOR_STRICT_WEBGPU_COMPLIANCE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .instance_flags(etalon::InstanceFlags::STRICT_WEBGPU_COMPLIANCE)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        let tex = ctx.device.create_texture(&etalon::TextureDescriptor {
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
            usage: etalon::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = tex.create_view(&etalon::TextureViewDescriptor::default());
        let mut encoder = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor::default());
        encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: etalon::Operations {
                    load: etalon::LoadOp::DontCare(unsafe { etalon::LoadOpDontCare::enabled() }),
                    store: etalon::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        fail(
            &ctx.device,
            || encoder.finish(),
            Some("STRICT_WEBGPU_COMPLIANCE"),
        );
    });

#[apply(gpu_test!)]
static DONT_CARE_DEPTH_STRICT_WEBGPU_COMPLIANCE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .instance_flags(etalon::InstanceFlags::STRICT_WEBGPU_COMPLIANCE)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        let tex = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: etalon::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Depth16Unorm,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = tex.create_view(&etalon::TextureViewDescriptor::default());
        let mut encoder = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor::default());
        encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            depth_stencil_attachment: Some(etalon::RenderPassDepthStencilAttachment {
                view: &view,
                depth_ops: Some(etalon::Operations {
                    load: etalon::LoadOp::DontCare(unsafe { etalon::LoadOpDontCare::enabled() }),
                    store: etalon::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        fail(
            &ctx.device,
            || encoder.finish(),
            Some("STRICT_WEBGPU_COMPLIANCE"),
        );
    });

#[apply(gpu_test!)]
static DONT_CARE_STENCIL_STRICT_WEBGPU_COMPLIANCE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .instance_flags(etalon::InstanceFlags::STRICT_WEBGPU_COMPLIANCE)
                .enable_noop(),
        )
        .run_sync(|ctx| {
            let tex = ctx.device.create_texture(&etalon::TextureDescriptor {
                label: None,
                size: etalon::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: etalon::TextureDimension::D2,
                format: etalon::TextureFormat::Stencil8,
                usage: etalon::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = tex.create_view(&etalon::TextureViewDescriptor::default());
            let mut encoder = ctx
                .device
                .create_command_encoder(&etalon::CommandEncoderDescriptor::default());
            encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                depth_stencil_attachment: Some(etalon::RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: None,
                    stencil_ops: Some(etalon::Operations {
                        load: etalon::LoadOp::DontCare(unsafe { etalon::LoadOpDontCare::enabled() }),
                        store: etalon::StoreOp::Store,
                    }),
                }),
                ..Default::default()
            });
            fail(
                &ctx.device,
                || encoder.finish(),
                Some("STRICT_WEBGPU_COMPLIANCE"),
            );
        });
