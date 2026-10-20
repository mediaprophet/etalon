use etalon_test::{
    apply, gpu_test, image, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(DEGENERATE_SWITCH);
}

/// FXC and potentially some glsl consumers have a bug when handling switch statements on a constant
/// with just a default case. (not sure if the constant part is relevant)
/// See <https://github.com/mediaprophet/etalon/issues/4514>.
///
/// This test will fail on Dx12 with FXC if this issue is not worked around.
///
/// So far no specific buggy glsl consumers have been identified and it isn't known whether the
/// bug is avoided there.
#[apply(gpu_test!)]
static DEGENERATE_SWITCH: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().force_fxc(true).enable_noop())
    .run_async(|ctx| async move { test_impl(&ctx).await });

async fn test_impl(ctx: &TestingContext) {
    const TEXTURE_HEIGHT: u32 = 2;
    const TEXTURE_WIDTH: u32 = 2;
    const BUFFER_SIZE: usize = (TEXTURE_WIDTH * TEXTURE_HEIGHT * 4) as usize;

    let texture = ctx.device.create_texture(&etalon::TextureDescriptor {
        label: Some("Offscreen texture"),
        size: etalon::Extent3d {
            width: TEXTURE_WIDTH,
            height: TEXTURE_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Rgba8Unorm,
        usage: etalon::TextureUsages::COPY_SRC | etalon::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&etalon::TextureViewDescriptor::default());

    let shader = ctx
        .device
        .create_shader_module(etalon::include_wgsl!("issue_4514.wgsl"));

    let pipeline = ctx
        .device
        .create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("Pipeline"),
            layout: None,
            vertex: etalon::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
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
        });

    let readback_buffer = image::ReadbackBuffers::new(&ctx.device, &texture);
    {
        let mut encoder = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
        {
            let mut render_pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                label: Some("Renderpass"),
                color_attachments: &[Some(etalon::RenderPassColorAttachment {
                    view: &texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: etalon::Operations {
                        // Important: this isn't the color expected below
                        load: etalon::LoadOp::Clear(etalon::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: etalon::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            render_pass.set_pipeline(&pipeline);
            render_pass.draw(0..3, 0..1);
        }
        readback_buffer.copy_from(&ctx.device, &mut encoder, &texture);
        ctx.queue.submit(Some(encoder.finish()));
    }

    let expected_data = [255; BUFFER_SIZE];
    readback_buffer
        .assert_buffer_contents(ctx, &expected_data)
        .await;
}
