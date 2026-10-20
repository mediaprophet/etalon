use etalon_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(TRANSITION_RESOURCES);
}

#[apply(gpu_test!)]
static TRANSITION_RESOURCES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        let texture = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: etalon::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rgba8Unorm,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });

        encoder.transition_resources(
            std::iter::empty(),
            [etalon::TextureTransition {
                texture: &texture,
                selector: None,
                state: etalon::TextureUses::COLOR_TARGET,
            }]
            .into_iter(),
        );

        ctx.queue.submit([encoder.finish()]);
    });
