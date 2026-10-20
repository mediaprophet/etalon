use etalon_test::{
    apply, gpu_test, image::ReadbackBuffers, FailureCase, GpuTestConfiguration, GpuTestInitializer,
    TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        CLEAR_TEXTURE_UNCOMPRESSED_GLES,
        CLEAR_TEXTURE_UNCOMPRESSED,
        CLEAR_TEXTURE_DEPTH,
        CLEAR_TEXTURE_DEPTH32_STENCIL8,
        CLEAR_TEXTURE_COMPRESSED_BCN,
        CLEAR_TEXTURE_COMPRESSED_ASTC,
        CLEAR_TEXTURE_COMPRESSED_ETC2,
    ]);
}

static TEXTURE_FORMATS_UNCOMPRESSED_GLES_COMPAT: &[etalon::TextureFormat] = &[
    etalon::TextureFormat::R8Unorm,
    etalon::TextureFormat::R8Snorm,
    etalon::TextureFormat::R8Uint,
    etalon::TextureFormat::R8Sint,
    etalon::TextureFormat::R16Uint,
    etalon::TextureFormat::R16Sint,
    etalon::TextureFormat::R16Float,
    etalon::TextureFormat::R32Uint,
    etalon::TextureFormat::R32Sint,
    etalon::TextureFormat::R32Float,
    etalon::TextureFormat::Rg16Uint,
    etalon::TextureFormat::Rg16Sint,
    etalon::TextureFormat::Rg16Float,
    etalon::TextureFormat::Rgba8Unorm,
    etalon::TextureFormat::Rgba8UnormSrgb,
    etalon::TextureFormat::Rgba8Snorm,
    etalon::TextureFormat::Rgba8Uint,
    etalon::TextureFormat::Rgba8Sint,
    etalon::TextureFormat::Bgra8Unorm,
    etalon::TextureFormat::Bgra8UnormSrgb,
    etalon::TextureFormat::Rgb10a2Uint,
    etalon::TextureFormat::Rgb10a2Unorm,
    etalon::TextureFormat::Rg11b10Ufloat,
    etalon::TextureFormat::Rg32Uint,
    etalon::TextureFormat::Rg32Sint,
    etalon::TextureFormat::Rg32Float,
    etalon::TextureFormat::Rgba16Uint,
    etalon::TextureFormat::Rgba16Sint,
    etalon::TextureFormat::Rgba16Float,
    etalon::TextureFormat::Rgba32Uint,
    etalon::TextureFormat::Rgba32Sint,
    etalon::TextureFormat::Rgba32Float,
];

static TEXTURE_FORMATS_UNCOMPRESSED: &[etalon::TextureFormat] = &[
    etalon::TextureFormat::Rg8Unorm,
    etalon::TextureFormat::Rg8Snorm,
    etalon::TextureFormat::Rg8Uint,
    etalon::TextureFormat::Rg8Sint,
    etalon::TextureFormat::Rgb9e5Ufloat,
];

static TEXTURE_FORMATS_DEPTH: &[etalon::TextureFormat] = &[
    etalon::TextureFormat::Stencil8,
    etalon::TextureFormat::Depth16Unorm,
    etalon::TextureFormat::Depth24Plus,
    etalon::TextureFormat::Depth24PlusStencil8,
    etalon::TextureFormat::Depth32Float,
];

// needs TEXTURE_COMPRESSION_BC
static TEXTURE_FORMATS_BC: &[etalon::TextureFormat] = &[
    etalon::TextureFormat::Bc1RgbaUnorm,
    etalon::TextureFormat::Bc1RgbaUnormSrgb,
    etalon::TextureFormat::Bc2RgbaUnorm,
    etalon::TextureFormat::Bc2RgbaUnormSrgb,
    etalon::TextureFormat::Bc3RgbaUnorm,
    etalon::TextureFormat::Bc3RgbaUnormSrgb,
    etalon::TextureFormat::Bc4RUnorm,
    etalon::TextureFormat::Bc4RSnorm,
    etalon::TextureFormat::Bc5RgUnorm,
    etalon::TextureFormat::Bc5RgSnorm,
    etalon::TextureFormat::Bc6hRgbUfloat,
    etalon::TextureFormat::Bc6hRgbFloat,
    etalon::TextureFormat::Bc7RgbaUnorm,
    etalon::TextureFormat::Bc7RgbaUnormSrgb,
];

// needs TEXTURE_COMPRESSION_ETC2
static TEXTURE_FORMATS_ETC2: &[etalon::TextureFormat] = &[
    etalon::TextureFormat::Etc2Rgb8Unorm,
    etalon::TextureFormat::Etc2Rgb8UnormSrgb,
    etalon::TextureFormat::Etc2Rgb8A1Unorm,
    etalon::TextureFormat::Etc2Rgb8A1UnormSrgb,
    etalon::TextureFormat::Etc2Rgba8Unorm,
    etalon::TextureFormat::Etc2Rgba8UnormSrgb,
    etalon::TextureFormat::EacR11Unorm,
    etalon::TextureFormat::EacR11Snorm,
    etalon::TextureFormat::EacRg11Unorm,
    etalon::TextureFormat::EacRg11Snorm,
];

// needs TEXTURE_COMPRESSION_ASTC
use etalon::{AstcBlock, AstcChannel};
static TEXTURE_FORMATS_ASTC: &[etalon::TextureFormat] = &[
    etalon::TextureFormat::Astc {
        block: AstcBlock::B4x4,
        channel: AstcChannel::Unorm,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B5x4,
        channel: AstcChannel::Unorm,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B5x5,
        channel: AstcChannel::Unorm,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B6x5,
        channel: AstcChannel::Unorm,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B6x6,
        channel: AstcChannel::Unorm,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B8x5,
        channel: AstcChannel::Unorm,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B8x6,
        channel: AstcChannel::Unorm,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B8x8,
        channel: AstcChannel::Unorm,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B10x5,
        channel: AstcChannel::Unorm,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B10x6,
        channel: AstcChannel::Unorm,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B10x8,
        channel: AstcChannel::Unorm,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B10x10,
        channel: AstcChannel::Unorm,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B12x10,
        channel: AstcChannel::Unorm,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B12x12,
        channel: AstcChannel::Unorm,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B4x4,
        channel: AstcChannel::UnormSrgb,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B5x4,
        channel: AstcChannel::UnormSrgb,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B5x5,
        channel: AstcChannel::UnormSrgb,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B6x5,
        channel: AstcChannel::UnormSrgb,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B6x6,
        channel: AstcChannel::UnormSrgb,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B8x5,
        channel: AstcChannel::UnormSrgb,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B8x6,
        channel: AstcChannel::UnormSrgb,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B8x8,
        channel: AstcChannel::UnormSrgb,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B10x5,
        channel: AstcChannel::UnormSrgb,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B10x6,
        channel: AstcChannel::UnormSrgb,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B10x8,
        channel: AstcChannel::UnormSrgb,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B10x10,
        channel: AstcChannel::UnormSrgb,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B12x10,
        channel: AstcChannel::UnormSrgb,
    },
    etalon::TextureFormat::Astc {
        block: AstcBlock::B12x12,
        channel: AstcChannel::UnormSrgb,
    },
];

async fn single_texture_clear_test(
    ctx: &TestingContext,
    format: etalon::TextureFormat,
    size: etalon::Extent3d,
    dimension: etalon::TextureDimension,
) {
    log::info!("clearing texture with {format:?}, dimension {dimension:?}, size {size:?}");

    let extra_usages = match format {
        etalon::TextureFormat::Depth24Plus | etalon::TextureFormat::Depth24PlusStencil8 => {
            etalon::TextureUsages::TEXTURE_BINDING
        }
        _ => etalon::TextureUsages::empty(),
    };

    let texture = ctx.device.create_texture(&etalon::TextureDescriptor {
        label: Some(&format!("texture {format:?}")),
        size,
        mip_level_count: if dimension == etalon::TextureDimension::D1 {
            1
        } else {
            // arbitrary value between 2 and max
            3
        },
        sample_count: 1, // multisampling is not supported for clear
        dimension,
        format,
        usage: etalon::TextureUsages::COPY_SRC | extra_usages,
        view_formats: &[],
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&etalon::CommandEncoderDescriptor::default());
    encoder.clear_texture(
        &texture,
        &etalon::ImageSubresourceRange {
            aspect: etalon::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        },
    );

    let readback_buffers = ReadbackBuffers::new(&ctx.device, &texture);

    readback_buffers.copy_from(&ctx.device, &mut encoder, &texture);

    ctx.queue.submit([encoder.finish()]);

    assert!(
        readback_buffers.are_zero(ctx).await,
        "texture with format {format:?} was not fully cleared"
    );
}

async fn clear_texture_tests(ctx: TestingContext, formats: &'static [etalon::TextureFormat]) {
    for &format in formats {
        let (block_width, block_height) = format.block_dimensions();
        let rounded_width = block_width * etalon::COPY_BYTES_PER_ROW_ALIGNMENT;
        let rounded_height = block_height * 4;

        let is_compressed_or_depth_stencil_format =
            format.is_compressed() || format.is_depth_stencil_format();
        let supports_1d = !is_compressed_or_depth_stencil_format;
        let supports_3d = format.is_bcn() || !is_compressed_or_depth_stencil_format;

        // 1D texture
        if supports_1d {
            single_texture_clear_test(
                &ctx,
                format,
                etalon::Extent3d {
                    width: rounded_width,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                etalon::TextureDimension::D1,
            )
            .await;
        }
        // 2D texture
        single_texture_clear_test(
            &ctx,
            format,
            etalon::Extent3d {
                width: rounded_width,
                height: rounded_height,
                depth_or_array_layers: 1,
            },
            etalon::TextureDimension::D2,
        )
        .await;
        // 2D array texture
        single_texture_clear_test(
            &ctx,
            format,
            etalon::Extent3d {
                width: rounded_width,
                height: rounded_height,
                depth_or_array_layers: 4,
            },
            etalon::TextureDimension::D2,
        )
        .await;
        if supports_3d {
            // volume texture
            single_texture_clear_test(
                &ctx,
                format,
                etalon::Extent3d {
                    width: rounded_width,
                    height: rounded_height,
                    depth_or_array_layers: 16,
                },
                etalon::TextureDimension::D3,
            )
            .await;
        }
    }
}

#[apply(gpu_test!)]
static CLEAR_TEXTURE_UNCOMPRESSED_GLES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(etalon::Features::CLEAR_TEXTURE)
            .skip(FailureCase::webgl2()),
    )
    .run_async(|ctx| clear_texture_tests(ctx, TEXTURE_FORMATS_UNCOMPRESSED_GLES_COMPAT));

#[apply(gpu_test!)]
static CLEAR_TEXTURE_UNCOMPRESSED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .expect_fail(
                FailureCase::backend(etalon::Backends::GL)
                    .panic("texture with format Rg8Snorm was not fully cleared")
                    .panic("texture with format Rgb9e5Ufloat was not fully cleared")
                    .validation_error("GL_INVALID_FRAMEBUFFER_OPERATION")
                    .validation_error("GL_INVALID_OPERATION"),
            )
            .features(etalon::Features::CLEAR_TEXTURE),
    )
    .run_async(|ctx| clear_texture_tests(ctx, TEXTURE_FORMATS_UNCOMPRESSED));

#[apply(gpu_test!)]
static CLEAR_TEXTURE_DEPTH: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(
                etalon::DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES
                    | etalon::DownlevelFlags::COMPUTE_SHADERS,
            )
            // https://github.com/mediaprophet/etalon/issues/5016
            .skip(FailureCase::adapter("Apple Paravirtual device"))
            .skip(FailureCase::webgl2())
            .limits(etalon::Limits::downlevel_defaults())
            .features(etalon::Features::CLEAR_TEXTURE),
    )
    .run_async(|ctx| clear_texture_tests(ctx, TEXTURE_FORMATS_DEPTH));

#[apply(gpu_test!)]
static CLEAR_TEXTURE_DEPTH32_STENCIL8: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(etalon::Features::CLEAR_TEXTURE | etalon::Features::DEPTH32FLOAT_STENCIL8)
            .downlevel_flags(etalon::DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES)
            // https://github.com/mediaprophet/etalon/issues/5016
            .skip(FailureCase::adapter("Apple Paravirtual device")),
    )
    .run_async(|ctx| clear_texture_tests(ctx, &[etalon::TextureFormat::Depth32FloatStencil8]));

#[apply(gpu_test!)]
static CLEAR_TEXTURE_COMPRESSED_BCN: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                etalon::Features::CLEAR_TEXTURE
                    | etalon::Features::TEXTURE_COMPRESSION_BC
                    | etalon::Features::TEXTURE_COMPRESSION_BC_SLICED_3D,
            )
            .limits(etalon::Limits {
                max_texture_dimension_3d: 1024,
                ..etalon::Limits::downlevel_defaults()
            })
            // https://bugs.chromium.org/p/angleproject/issues/detail?id=7056
            .expect_fail(FailureCase::backend_adapter(etalon::Backends::GL, "ANGLE"))
            // compressed texture copy to buffer not yet implemented
            .expect_fail(FailureCase::backend(etalon::Backends::GL)),
    )
    .run_async(|ctx| clear_texture_tests(ctx, TEXTURE_FORMATS_BC));

#[apply(gpu_test!)]
static CLEAR_TEXTURE_COMPRESSED_ASTC: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                etalon::Features::CLEAR_TEXTURE
                    | etalon::Features::TEXTURE_COMPRESSION_ASTC
                    | etalon::Features::TEXTURE_COMPRESSION_ASTC_SLICED_3D,
            )
            .limits(etalon::Limits {
                max_texture_dimension_2d: etalon::COPY_BYTES_PER_ROW_ALIGNMENT * 12,
                max_texture_dimension_3d: etalon::COPY_BYTES_PER_ROW_ALIGNMENT * 12,
                ..etalon::Limits::downlevel_defaults()
            })
            // https://bugs.chromium.org/p/angleproject/issues/detail?id=7056
            .expect_fail(FailureCase::backend_adapter(etalon::Backends::GL, "ANGLE"))
            // compressed texture copy to buffer not yet implemented
            .expect_fail(FailureCase::backend(etalon::Backends::GL)),
    )
    .run_async(|ctx| clear_texture_tests(ctx, TEXTURE_FORMATS_ASTC));

#[apply(gpu_test!)]
static CLEAR_TEXTURE_COMPRESSED_ETC2: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(etalon::Features::CLEAR_TEXTURE | etalon::Features::TEXTURE_COMPRESSION_ETC2)
            // https://bugs.chromium.org/p/angleproject/issues/detail?id=7056
            .expect_fail(FailureCase::backend_adapter(etalon::Backends::GL, "ANGLE"))
            // compressed texture copy to buffer not yet implemented
            .expect_fail(FailureCase::backend(etalon::Backends::GL)),
    )
    .run_async(|ctx| clear_texture_tests(ctx, TEXTURE_FORMATS_ETC2));
