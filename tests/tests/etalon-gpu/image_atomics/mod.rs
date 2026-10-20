//! Tests for image atomics.

use etalon::ShaderModuleDescriptor;
use etalon_test::{
    apply, fail, gpu_test, image::ReadbackBuffers, GpuTestConfiguration, GpuTestInitializer,
    TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        IMAGE_64_ATOMICS,
        IMAGE_32_ATOMICS,
        IMAGE_ATOMICS_NOT_ENABLED,
        IMAGE_ATOMICS_NOT_SUPPORTED,
    ]);
}

#[apply(gpu_test!)]
static IMAGE_64_ATOMICS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .limits(etalon::Limits {
                max_storage_textures_per_shader_stage: 1,
                max_compute_invocations_per_workgroup: 64,
                max_compute_workgroup_size_x: 4,
                max_compute_workgroup_size_y: 4,
                max_compute_workgroup_size_z: 4,
                max_compute_workgroups_per_dimension: etalon::COPY_BYTES_PER_ROW_ALIGNMENT,
                ..etalon::Limits::downlevel_webgl2_defaults()
            })
            .features(
                etalon::Features::TEXTURE_ATOMIC
                    | etalon::Features::TEXTURE_INT64_ATOMIC
                    | etalon::Features::SHADER_INT64,
            ),
    )
    .run_async(|ctx| async move {
        test_format(
            ctx,
            etalon::TextureFormat::R64Uint,
            etalon::include_wgsl!("image_64_atomics.wgsl"),
        )
        .await;
    });

#[apply(gpu_test!)]
static IMAGE_32_ATOMICS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .limits(etalon::Limits {
                max_storage_textures_per_shader_stage: 1,
                max_compute_invocations_per_workgroup: 64,
                max_compute_workgroup_size_x: 4,
                max_compute_workgroup_size_y: 4,
                max_compute_workgroup_size_z: 4,
                max_compute_workgroups_per_dimension: etalon::COPY_BYTES_PER_ROW_ALIGNMENT,
                ..etalon::Limits::downlevel_webgl2_defaults()
            })
            .features(etalon::Features::TEXTURE_ATOMIC),
    )
    .run_async(|ctx| async move {
        test_format(
            ctx,
            etalon::TextureFormat::R32Uint,
            etalon::include_wgsl!("image_32_atomics.wgsl"),
        )
        .await;
    });

async fn test_format(
    ctx: TestingContext,
    format: etalon::TextureFormat,
    desc: ShaderModuleDescriptor<'_>,
) {
    let pixel_bytes = format.target_pixel_byte_cost().unwrap();
    let size = etalon::Extent3d {
        width: etalon::COPY_BYTES_PER_ROW_ALIGNMENT,
        height: etalon::COPY_BYTES_PER_ROW_ALIGNMENT,
        depth_or_array_layers: 1,
    };
    let bind_group_layout_entry = etalon::BindGroupLayoutEntry {
        binding: 0,
        visibility: etalon::ShaderStages::COMPUTE,
        ty: etalon::BindingType::StorageTexture {
            access: etalon::StorageTextureAccess::Atomic,
            format,
            view_dimension: etalon::TextureViewDimension::D2,
        },
        count: None,
    };

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: None,
            entries: &[bind_group_layout_entry],
        });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
    let shader = ctx.device.create_shader_module(desc);
    let pipeline = ctx
        .device
        .create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("image atomics pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: etalon::PipelineCompilationOptions::default(),
            cache: None,
        });

    let tex = ctx.device.create_texture(&etalon::TextureDescriptor {
        label: None,
        dimension: etalon::TextureDimension::D2,
        size,
        format,
        usage: etalon::TextureUsages::STORAGE_BINDING
            | etalon::TextureUsages::STORAGE_ATOMIC
            | etalon::TextureUsages::COPY_SRC,
        mip_level_count: 1,
        sample_count: 1,
        view_formats: &[],
    });
    let view = tex.create_view(&etalon::TextureViewDescriptor {
        format: Some(format),
        aspect: etalon::TextureAspect::All,
        ..etalon::TextureViewDescriptor::default()
    });
    let bind_group = ctx.device.create_bind_group(&etalon::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[etalon::BindGroupEntry {
            binding: 0,
            resource: etalon::BindingResource::TextureView(&view),
        }],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&etalon::CommandEncoderDescriptor::default());
    let mut rpass = encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    rpass.set_pipeline(&pipeline);
    rpass.set_bind_group(0, Some(&bind_group), &[]);
    rpass.dispatch_workgroups(size.width, size.height, 1);
    drop(rpass);

    let readback_buffers = ReadbackBuffers::new(&ctx.device, &tex);
    readback_buffers.copy_from(&ctx.device, &mut encoder, &tex);

    ctx.queue.submit([encoder.finish()]);

    let padding = [0].repeat(pixel_bytes as usize - size_of::<u32>());
    let data: Vec<u8> = (0..size.width as usize * size.height as usize)
        .flat_map(|i| {
            let x = i as u32 % size.width;
            let y = i as u32 / size.width;
            [bytemuck::bytes_of(&u32::min(x, y)), &padding].concat()
        })
        .collect();

    readback_buffers.assert_buffer_contents(&ctx, &data).await;
}

#[apply(gpu_test!)]
static IMAGE_ATOMICS_NOT_ENABLED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        let size = etalon::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        };

        fail(
            &ctx.device,
            || {
                let _ = ctx.device.create_texture(&etalon::TextureDescriptor {
                    label: None,
                    dimension: etalon::TextureDimension::D2,
                    size,
                    format: etalon::TextureFormat::R32Uint,
                    usage: etalon::TextureUsages::STORAGE_ATOMIC,
                    mip_level_count: 1,
                    sample_count: 1,
                    view_formats: &[],
                });
            },
            Some("Texture usages TextureUsages(STORAGE_ATOMIC) are not allowed on a texture of type R32Uint"),
        );
    });

#[apply(gpu_test!)]
static IMAGE_ATOMICS_NOT_SUPPORTED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(etalon::Features::TEXTURE_ATOMIC).enable_noop())
    .run_sync(|ctx| {
        let size = etalon::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        };

        fail(
            &ctx.device,
            || {
                let _ = ctx.device.create_texture(&etalon::TextureDescriptor {
                    label: None,
                    dimension: etalon::TextureDimension::D2,
                    size,
                    format: etalon::TextureFormat::R8Uint,
                    usage: etalon::TextureUsages::STORAGE_ATOMIC,
                    mip_level_count: 1,
                    sample_count: 1,
                    view_formats: &[],
                });
            },
            Some("Texture usages TextureUsages(STORAGE_ATOMIC) are not allowed on a texture of type R8Uint"),
        );
    });
