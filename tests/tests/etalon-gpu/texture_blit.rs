use etalon_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        TEXTURE_BLIT_WITH_LINEAR_FILTER_TEST,
        TEXTURE_BLIT_WITH_NEAREST_FILTER_TEST,
    ]);
}

#[apply(gpu_test!)]
static TEXTURE_BLIT_WITH_LINEAR_FILTER_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        let source = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: etalon::Extent3d {
                width: 100,
                height: 100,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rgba16Float,
            usage: etalon::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let target = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: etalon::Extent3d {
                width: 100,
                height: 100,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rgba8UnormSrgb,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let blitter = etalon::util::TextureBlitterBuilder::new(
            &ctx.device,
            etalon::TextureFormat::Rgba8UnormSrgb,
        )
        .sample_type(etalon::FilterMode::Linear)
        .build();
        let mut encoder = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

        blitter.copy(
            &ctx.device,
            &mut encoder,
            &source.create_view(&etalon::TextureViewDescriptor::default()),
            &target.create_view(&etalon::TextureViewDescriptor::default()),
        );
    });

#[apply(gpu_test!)]
static TEXTURE_BLIT_WITH_NEAREST_FILTER_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        let source = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: etalon::Extent3d {
                width: 100,
                height: 100,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rgba16Float,
            usage: etalon::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let target = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: etalon::Extent3d {
                width: 100,
                height: 100,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rgba8UnormSrgb,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let blitter = etalon::util::TextureBlitterBuilder::new(
            &ctx.device,
            etalon::TextureFormat::Rgba8UnormSrgb,
        )
        .sample_type(etalon::FilterMode::Linear)
        .build();

        let mut encoder = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

        blitter.copy(
            &ctx.device,
            &mut encoder,
            &source.create_view(&etalon::TextureViewDescriptor::default()),
            &target.create_view(&etalon::TextureViewDescriptor::default()),
        );
    });
