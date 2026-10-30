use std::num::NonZeroU32;

use etalon::*;
use etalon_test::{
    apply, gpu_test, image::ReadbackBuffers, GpuTestConfiguration, GpuTestInitializer,
    TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        BINDING_ARRAY_STORAGE_TEXTURES,
        PARTIAL_BINDING_ARRAY_STORAGE_TEXTURES,
        UNIFORM_BINDING_ARRAY_STORAGE_TEXTURES,
        PARTIAL_UNIFORM_BINDING_ARRAY_STORAGE_TEXTURES,
        NON_UNIFORM_STORAGE_TEXTURE_ARRAY_REQUIRES_FEATURE,
    ]);
}

#[apply(gpu_test!)]
static BINDING_ARRAY_STORAGE_TEXTURES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .instance_flags(etalon::InstanceFlags::GPU_BASED_VALIDATION)
            .features(
                Features::TEXTURE_BINDING_ARRAY
                    | Features::STORAGE_RESOURCE_BINDING_ARRAY
                    | Features::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
                    | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            )
            .limits(Limits {
                max_binding_array_elements_per_shader_stage: 17,
                ..Limits::default()
            })
            .expect_fail(
                // https://github.com/mediaprophet/etalon/issues/9849 and/or https://github.com/mediaprophet/etalon/issues/6745
                etalon_test::FailureCase::molten_vk()
                    .validation_error("Shader library compile failed")
                    .validation_error("could not be compiled into pipeline"),
            ),
    )
    .run_async(|ctx| async move { binding_array_storage_textures(ctx, false, true).await });

#[apply(gpu_test!)]
static PARTIAL_BINDING_ARRAY_STORAGE_TEXTURES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .instance_flags(etalon::InstanceFlags::GPU_BASED_VALIDATION)
            .features(
                Features::TEXTURE_BINDING_ARRAY
                    | Features::PARTIALLY_BOUND_BINDING_ARRAY
                    | Features::STORAGE_RESOURCE_BINDING_ARRAY
                    | Features::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
                    | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            )
            .limits(Limits {
                max_binding_array_elements_per_shader_stage: 33,
                ..Limits::default()
            })
            .expect_fail(
                // https://github.com/mediaprophet/etalon/issues/9849 and/or https://github.com/mediaprophet/etalon/issues/6745
                etalon_test::FailureCase::molten_vk()
                    .validation_error("Shader library compile failed")
                    .validation_error("could not be compiled into pipeline"),
            ),
    )
    .run_async(|ctx| async move { binding_array_storage_textures(ctx, true, true).await });

fn uniform_array_parameters(partially_bound: bool) -> TestParameters {
    let mut features = Features::TEXTURE_BINDING_ARRAY
        | Features::STORAGE_RESOURCE_BINDING_ARRAY
        | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    if partially_bound {
        features |= Features::PARTIALLY_BOUND_BINDING_ARRAY;
    }
    TestParameters::default()
        .features(features)
        .limits(Limits {
            max_binding_array_elements_per_shader_stage: if partially_bound { 33 } else { 17 },
            ..Limits::default()
        })
        .expect_fail(
            etalon_test::FailureCase::molten_vk()
                .validation_error("Shader library compile failed")
                .validation_error("could not be compiled into pipeline"),
        )
}

#[apply(gpu_test!)]
static UNIFORM_BINDING_ARRAY_STORAGE_TEXTURES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(uniform_array_parameters(false))
    .run_async(|ctx| async move { binding_array_storage_textures(ctx, false, false).await });

#[apply(gpu_test!)]
static PARTIAL_UNIFORM_BINDING_ARRAY_STORAGE_TEXTURES: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(uniform_array_parameters(true))
        .run_async(|ctx| async move { binding_array_storage_textures(ctx, true, false).await });

#[apply(gpu_test!)]
static NON_UNIFORM_STORAGE_TEXTURE_ARRAY_REQUIRES_FEATURE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .features(
                    Features::TEXTURE_BINDING_ARRAY
                        | Features::STORAGE_RESOURCE_BINDING_ARRAY
                        | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                )
                .limits(Limits {
                    max_binding_array_elements_per_shader_stage: 17,
                    ..Limits::default()
                }),
        )
        .run_async(|ctx| async move {
            assert!(!ctx
                .device
                .features()
                .contains(Features::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING));
            let module = etalon_test::fail(
                &ctx.device,
                || {
                    ctx.device.create_shader_module(ShaderModuleDescriptor {
                        label: Some("non-uniform storage texture array without capability"),
                        source: ShaderSource::Wgsl(STORAGE_ARRAY_SHADER.into()),
                    })
                },
                Some("validation error"),
            );
            let info = module.get_compilation_info().await;
            assert!(info.messages.iter().any(|message| {
                message.message_type == CompilationMessageType::Error
                    && message
                        .message
                        .contains("STORAGE_TEXTURE_BINDING_ARRAY_NON_UNIFORM_INDEXING")
            }));
        });

const STORAGE_ARRAY_SHADER: &str = r#"
        enable etalon_binding_array;
        @group(0) @binding(0)
        var textures: binding_array<texture_storage_2d<rgba8unorm, read_write> >;

        @compute
        @workgroup_size(4, 4, 1)
        fn compMain(@builtin(global_invocation_id) id: vec3u) {
            // Read from the 4x4 textures in 0-15, then write to the 4x4 texture in 16

            let pixel = vec2u(id.xy);
            let index = pixel.y * 4 + pixel.x;

            let color = textureLoad(textures[index], vec2u(0));
            textureStore(textures[16], pixel, color);
        }
    "#;

async fn binding_array_storage_textures(
    ctx: TestingContext,
    partially_bound: bool,
    non_uniform: bool,
) {
    let shader = STORAGE_ARRAY_SHADER;

    let image = image::load_from_memory(include_bytes!("../3x3_colors.png")).unwrap();
    // Resize image to 4x4
    let image = image
        .resize_exact(4, 4, image::imageops::FilterType::Gaussian)
        .into_rgba8();

    // Create one texture for each pixel
    let mut input_views = Vec::with_capacity(64);
    let mut input_textures = Vec::with_capacity(16);
    let mut input_readbacks = Vec::with_capacity(16);
    let mut upload_encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    for data in image.pixels() {
        let texture = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        ctx.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &data.0,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let readback = ReadbackBuffers::new(&ctx.device, &texture);
        readback.copy_from(&ctx.device, &mut upload_encoder, &texture);
        input_readbacks.push((readback, data.0));
        input_views.push(texture.create_view(&TextureViewDescriptor::default()));
        input_textures.push(texture);
    }
    ctx.queue.submit([upload_encoder.finish()]);
    for (index, (readback, expected)) in input_readbacks.iter().enumerate() {
        println!("storage array input upload oracle: {index}");
        readback.assert_buffer_contents(&ctx, expected).await;
    }

    let output_texture = ctx.device.create_texture(&etalon::TextureDescriptor {
        label: Some("Output Texture"),
        size: Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let output_view = output_texture.create_view(&TextureViewDescriptor::default());

    let multiplier = if partially_bound { 2 } else { 1 };

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: TextureFormat::Rgba8Unorm,
                    view_dimension: TextureViewDimension::D2,
                },
                count: Some(NonZeroU32::new(4 * 4 * multiplier + 1).unwrap()),
            }],
        });

    let mut input_view_references: Vec<_> = input_views.iter().collect();
    input_view_references.push(&output_view);

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureViewArray(&input_view_references),
        }],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

    for (phase, color_load, expected) in [
        (
            "constant_store",
            "vec4f(17.0 / 255.0, 34.0 / 255.0, 51.0 / 255.0, 1.0)",
            [17, 34, 51, 255].repeat(16),
        ),
        (
            "uniform_load",
            "textureLoad(textures[0], vec2u(0))",
            image.pixels().next().unwrap().0.repeat(16),
        ),
        (
            "varying_load",
            "textureLoad(textures[index], vec2u(0))",
            image.as_raw().clone(),
        ),
        (
            "pending_upload_varying_load",
            "textureLoad(textures[index], vec2u(0))",
            image
                .as_raw()
                .iter()
                .enumerate()
                .map(|(index, &value)| if index % 4 == 3 { value } else { value ^ 0x7f })
                .collect(),
        ),
    ] {
        if !non_uniform && phase.contains("varying_load") {
            continue;
        }
        println!("storage array compute oracle: {phase}");
        if phase == "pending_upload_varying_load" {
            for (texture, bytes) in input_textures.iter().zip(expected.chunks_exact(4)) {
                ctx.queue.write_texture(
                    TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    bytes,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4),
                        rows_per_image: Some(1),
                    },
                    Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
        let source = shader.replace("textureLoad(textures[index], vec2u(0))", color_load);
        let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(phase),
            source: ShaderSource::Wgsl(source.into()),
        });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("compMain"),
                compilation_options: Default::default(),
                cache: None,
            });

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut render_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            render_pass.set_pipeline(&pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.dispatch_workgroups(1, 1, 1);
        }

        let readback_buffers = ReadbackBuffers::new(&ctx.device, &output_texture);
        readback_buffers.copy_from(&ctx.device, &mut encoder, &output_texture);

        ctx.queue.submit(Some(encoder.finish()));

        readback_buffers
            .assert_buffer_contents(&ctx, &expected)
            .await;
    }
}
