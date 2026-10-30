use etalon_test::{
    apply, gpu_test, image::ReadbackBuffers, GpuTestConfiguration, GpuTestInitializer,
    TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        TEXTURE_BLIT_WITH_LINEAR_FILTER_TEST,
        TEXTURE_BLIT_WITH_NEAREST_FILTER_TEST,
        TEXTURE_BLIT_DOWNSAMPLE_LINEAR_TEST,
        TEXTURE_BLIT_DOWNSAMPLE_NEAREST_TEST,
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

/// Blits a 2x2 texture with a black left column and a white right column into a
/// 1x1 texture, and returns the red channel of the result.
async fn downsample_red_channel(ctx: &TestingContext, filter: etalon::FilterMode) -> u8 {
    let source = ctx.device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size: etalon::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Rgba8Unorm,
        usage: etalon::TextureUsages::TEXTURE_BINDING | etalon::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let column = [[0, 0, 0, 255], [255, 255, 255, 255]];
    let texels = [column, column].concat().concat();
    ctx.queue.write_texture(
        source.as_image_copy(),
        &texels,
        etalon::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: None,
        },
        source.size(),
    );

    let target = ctx.device.create_texture(&etalon::TextureDescriptor {
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

    let blitter =
        etalon::util::TextureBlitterBuilder::new(&ctx.device, etalon::TextureFormat::Rgba8Unorm)
            .sample_type(filter)
            .build();

    let readback = ReadbackBuffers::new(&ctx.device, &target);
    let mut encoder = ctx
        .device
        .create_command_encoder(&etalon::CommandEncoderDescriptor::default());
    blitter.copy(
        &ctx.device,
        &mut encoder,
        &source.create_view(&etalon::TextureViewDescriptor::default()),
        &target.create_view(&etalon::TextureViewDescriptor::default()),
    );
    readback.copy_from(&ctx.device, &mut encoder, &target);
    ctx.queue.submit(Some(encoder.finish()));

    let data = readback.retrieve(ctx, etalon::TextureAspect::All).await;
    data.all()[0]
}

#[apply(gpu_test!)]
static TEXTURE_BLIT_DOWNSAMPLE_LINEAR_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_async(|ctx| async move {
        let red = downsample_red_channel(&ctx, etalon::FilterMode::Linear).await;
        assert!(
            (120..=135).contains(&red),
            "linear downsampling should blend the texels, got {red}"
        );
    });

#[apply(gpu_test!)]
static TEXTURE_BLIT_DOWNSAMPLE_NEAREST_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_async(|ctx| async move {
        let red = downsample_red_channel(&ctx, etalon::FilterMode::Nearest).await;
        assert!(
            red == 0 || red == 255,
            "nearest downsampling should pick a single texel, got {red}"
        );
    });
