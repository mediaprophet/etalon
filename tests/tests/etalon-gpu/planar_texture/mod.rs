//! Tests for nv12 texture creation and sampling.

use etalon_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        NV12_TEXTURE_CREATION_SAMPLING,
        P010_TEXTURE_CREATION_SAMPLING,
        NV12_TEXTURE_RENDERING,
        NV12_TEXTURE_COPYING,
        P010_TEXTURE_COPYING,
        NV12_PLANE_TO_SINGLE_PLANE_COPY,
    ]);
}

// Helper function to test planar texture creation and sampling.
fn test_planar_texture_creation_sampling(
    ctx: &TestingContext,
    y_view: &etalon::TextureView,
    uv_view: &etalon::TextureView,
) {
    let target_format = etalon::TextureFormat::Bgra8UnormSrgb;

    let shader = ctx
        .device
        .create_shader_module(etalon::include_wgsl!("planar_texture_sampling.wgsl"));
    let pipeline = ctx
        .device
        .create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("planar texture pipeline"),
            layout: None,
            vertex: etalon::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(etalon::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(target_format.into())],
            }),
            primitive: etalon::PrimitiveState {
                topology: etalon::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(etalon::IndexFormat::Uint32),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

    let sampler = ctx.device.create_sampler(&etalon::SamplerDescriptor {
        min_filter: etalon::FilterMode::Linear,
        mag_filter: etalon::FilterMode::Linear,
        ..Default::default()
    });
    let bind_group = ctx.device.create_bind_group(&etalon::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            etalon::BindGroupEntry {
                binding: 0,
                resource: etalon::BindingResource::Sampler(&sampler),
            },
            etalon::BindGroupEntry {
                binding: 1,
                resource: etalon::BindingResource::TextureView(y_view),
            },
            etalon::BindGroupEntry {
                binding: 2,
                resource: etalon::BindingResource::TextureView(uv_view),
            },
        ],
    });

    let target_tex = ctx.device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size: y_view.texture().size(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: target_format,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target_view = target_tex.create_view(&etalon::TextureViewDescriptor::default());

    let mut encoder = ctx
        .device
        .create_command_encoder(&etalon::CommandEncoderDescriptor::default());
    let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
        label: None,
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
    rpass.set_pipeline(&pipeline);
    rpass.set_bind_group(0, &bind_group, &[]);
    rpass.draw(0..4, 0..1);
    drop(rpass);
    ctx.queue.submit([encoder.finish()]);
}

// Helper function to test rendering onto planar texture.
fn test_planar_texture_rendering(
    ctx: &TestingContext,
    (y_view, y_format): (&etalon::TextureView, etalon::TextureFormat),
    (uv_view, uv_format): (&etalon::TextureView, etalon::TextureFormat),
) {
    let shader = ctx
        .device
        .create_shader_module(etalon::include_wgsl!("planar_texture_rendering.wgsl"));
    let y_pipeline = ctx
        .device
        .create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("y plane pipeline"),
            layout: None,
            vertex: etalon::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(etalon::FragmentState {
                module: &shader,
                entry_point: Some("fs_y_main"),
                compilation_options: Default::default(),
                targets: &[Some(y_format.into())],
            }),
            primitive: etalon::PrimitiveState {
                topology: etalon::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(etalon::IndexFormat::Uint32),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

    let uv_pipeline = ctx
        .device
        .create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("uv plane pipeline"),
            layout: None,
            vertex: etalon::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(etalon::FragmentState {
                module: &shader,
                entry_point: Some("fs_uv_main"),
                compilation_options: Default::default(),
                targets: &[Some(uv_format.into())],
            }),
            primitive: etalon::PrimitiveState {
                topology: etalon::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(etalon::IndexFormat::Uint32),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

    {
        let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                ops: etalon::Operations::default(),
                resolve_target: None,
                view: y_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&y_pipeline);
        rpass.draw(0..3, 0..1);
    }
    {
        let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                ops: etalon::Operations::default(),
                resolve_target: None,
                view: uv_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&uv_pipeline);
        rpass.draw(0..3, 0..1);
    }

    ctx.queue.submit([encoder.finish()]);
}

/// Ensures that creation and sampling of an NV12 format texture works as
/// expected.
#[apply(gpu_test!)]
static NV12_TEXTURE_CREATION_SAMPLING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(etalon::Features::TEXTURE_FORMAT_NV12)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        // Deliberately non-square so a width/height swap is caught. Both
        // dimensions stay even, as required by NV12/P010 (chroma is half-res).
        let size = etalon::Extent3d {
            width: 256,
            height: 128,
            depth_or_array_layers: 1,
        };
        let tex = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            dimension: etalon::TextureDimension::D2,
            size,
            format: etalon::TextureFormat::NV12,
            usage: etalon::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let y_view = tex.create_view(&etalon::TextureViewDescriptor {
            format: Some(etalon::TextureFormat::R8Unorm),
            aspect: etalon::TextureAspect::Plane0,
            ..Default::default()
        });
        let uv_view = tex.create_view(&etalon::TextureViewDescriptor {
            format: Some(etalon::TextureFormat::Rg8Unorm),
            aspect: etalon::TextureAspect::Plane1,
            ..Default::default()
        });

        test_planar_texture_creation_sampling(&ctx, &y_view, &uv_view);
    });

/// Ensures that creation and sampling of a P010 format texture works as
/// expected.
#[apply(gpu_test!)]
static P010_TEXTURE_CREATION_SAMPLING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                etalon::Features::TEXTURE_FORMAT_P010 | etalon::Features::TEXTURE_FORMAT_16BIT_NORM,
            )
            .enable_noop(),
    )
    .run_sync(|ctx| {
        // Deliberately non-square so a width/height swap is caught. Both
        // dimensions stay even, as required by NV12/P010 (chroma is half-res).
        let size = etalon::Extent3d {
            width: 256,
            height: 128,
            depth_or_array_layers: 1,
        };
        let tex = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            dimension: etalon::TextureDimension::D2,
            size,
            format: etalon::TextureFormat::P010,
            usage: etalon::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let y_view = tex.create_view(&etalon::TextureViewDescriptor {
            format: Some(etalon::TextureFormat::R16Unorm),
            aspect: etalon::TextureAspect::Plane0,
            ..Default::default()
        });
        let uv_view = tex.create_view(&etalon::TextureViewDescriptor {
            format: Some(etalon::TextureFormat::Rg16Unorm),
            aspect: etalon::TextureAspect::Plane1,
            ..Default::default()
        });

        test_planar_texture_creation_sampling(&ctx, &y_view, &uv_view);
    });

/// Ensures that rendering on to NV12 format texture works as expected.
#[apply(gpu_test!)]
static NV12_TEXTURE_RENDERING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(etalon::Features::TEXTURE_FORMAT_NV12)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        // Deliberately non-square so a width/height swap is caught. Both
        // dimensions stay even, as required by NV12/P010 (chroma is half-res).
        let size = etalon::Extent3d {
            width: 256,
            height: 128,
            depth_or_array_layers: 1,
        };
        let tex = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            dimension: etalon::TextureDimension::D2,
            size,
            format: etalon::TextureFormat::NV12,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let y_view = tex.create_view(&etalon::TextureViewDescriptor {
            format: Some(etalon::TextureFormat::R8Unorm),
            aspect: etalon::TextureAspect::Plane0,
            ..Default::default()
        });
        let uv_view = tex.create_view(&etalon::TextureViewDescriptor {
            format: Some(etalon::TextureFormat::Rg8Unorm),
            aspect: etalon::TextureAspect::Plane1,
            ..Default::default()
        });

        test_planar_texture_rendering(
            &ctx,
            (&y_view, etalon::TextureFormat::R8Unorm),
            (&uv_view, etalon::TextureFormat::Rg8Unorm),
        );
    });

/// Ensures that copying NV12 texture to NV12 texture works as expected
#[apply(gpu_test!)]
static NV12_TEXTURE_COPYING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(etalon::Features::TEXTURE_FORMAT_NV12)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        // Deliberately non-square so a width/height swap is caught. Both
        // dimensions stay even, as required by NV12/P010 (chroma is half-res).
        let size = etalon::Extent3d {
            width: 256,
            height: 128,
            depth_or_array_layers: 1,
        };
        let input_texture = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            dimension: etalon::TextureDimension::D2,
            size,
            format: etalon::TextureFormat::NV12,
            usage: etalon::TextureUsages::COPY_SRC,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let output_texture = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            dimension: etalon::TextureDimension::D2,
            size,
            format: etalon::TextureFormat::NV12,
            usage: etalon::TextureUsages::COPY_DST,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });

        let mut command_encoder = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor::default());
        command_encoder.copy_texture_to_texture(
            input_texture.as_image_copy(),
            output_texture.as_image_copy(),
            size,
        );
        ctx.queue.submit([command_encoder.finish()]);
    });

/// Ensures that copying a single plane of an NV12 source into a matching
/// single-plane destination (Plane0 → R8Unorm, Plane1 → Rg8Unorm) round-trips
/// byte-for-byte. Exercises the planar→single-plane copy-compatibility
/// extension in `copy_texture_to_texture`.
#[apply(gpu_test!)]
static NV12_PLANE_TO_SINGLE_PLANE_COPY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(etalon::Features::TEXTURE_FORMAT_NV12))
    .run_async(|ctx| async move {
        // Width chosen so that bytes-per-row is 256-aligned for both planes:
        //   luma   R8Unorm:  256 px * 1 byte/px = 256
        //   chroma Rg8Unorm: 128 px * 2 byte/px = 256
        // Height is deliberately different from width so a swap is caught.
        const WIDTH: u32 = 256;
        const HEIGHT: u32 = 128;
        let luma_size = etalon::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        };
        let chroma_size = etalon::Extent3d {
            width: WIDTH / 2,
            height: HEIGHT / 2,
            depth_or_array_layers: 1,
        };

        let nv12 = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: Some("nv12 src"),
            dimension: etalon::TextureDimension::D2,
            size: luma_size,
            format: etalon::TextureFormat::NV12,
            usage: etalon::TextureUsages::COPY_SRC | etalon::TextureUsages::COPY_DST,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });

        // Distinct patterns per plane so a swap or plane-0-fallback would fail
        // the assertion at the end. Each pattern depends on both `x` and `y`
        // with different per-axis weights, so it is asymmetric under transpose,
        // horizontal mirror, and vertical mirror — a row/column mix-up changes
        // the bytes and fails the comparison.
        let luma_bytes: Vec<u8> = (0..HEIGHT)
            .flat_map(|y| {
                (0..WIDTH).map(move |x| x.wrapping_mul(3).wrapping_add(y.wrapping_mul(101)) as u8)
            })
            .collect();
        let chroma_bytes: Vec<u8> = (0..HEIGHT / 2)
            .flat_map(|y| {
                (0..WIDTH / 2).flat_map(move |x| {
                    let r = x.wrapping_mul(7).wrapping_add(y.wrapping_mul(53)) as u8;
                    let g = (x.wrapping_mul(29).wrapping_add(y.wrapping_mul(13)) ^ 0xA5) as u8;
                    [r, g]
                })
            })
            .collect();

        ctx.queue.write_texture(
            etalon::TexelCopyTextureInfo {
                texture: &nv12,
                mip_level: 0,
                origin: etalon::Origin3d::ZERO,
                aspect: etalon::TextureAspect::Plane0,
            },
            &luma_bytes,
            etalon::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH),
                rows_per_image: Some(HEIGHT),
            },
            luma_size,
        );
        ctx.queue.write_texture(
            etalon::TexelCopyTextureInfo {
                texture: &nv12,
                mip_level: 0,
                origin: etalon::Origin3d::ZERO,
                aspect: etalon::TextureAspect::Plane1,
            },
            &chroma_bytes,
            etalon::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH / 2 * 2),
                rows_per_image: Some(HEIGHT / 2),
            },
            chroma_size,
        );

        let r8 = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: Some("r8 dst"),
            dimension: etalon::TextureDimension::D2,
            size: luma_size,
            format: etalon::TextureFormat::R8Unorm,
            usage: etalon::TextureUsages::COPY_DST | etalon::TextureUsages::COPY_SRC,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let rg8 = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: Some("rg8 dst"),
            dimension: etalon::TextureDimension::D2,
            size: chroma_size,
            format: etalon::TextureFormat::Rg8Unorm,
            usage: etalon::TextureUsages::COPY_DST | etalon::TextureUsages::COPY_SRC,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });

        let r8_readback = ctx.device.create_buffer(&etalon::BufferDescriptor {
            label: None,
            size: luma_bytes.len() as u64,
            usage: etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rg8_readback = ctx.device.create_buffer(&etalon::BufferDescriptor {
            label: None,
            size: chroma_bytes.len() as u64,
            usage: etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

        // The path under test.
        encoder.copy_texture_to_texture(
            etalon::TexelCopyTextureInfo {
                texture: &nv12,
                mip_level: 0,
                origin: etalon::Origin3d::ZERO,
                aspect: etalon::TextureAspect::Plane0,
            },
            etalon::TexelCopyTextureInfo {
                texture: &r8,
                mip_level: 0,
                origin: etalon::Origin3d::ZERO,
                aspect: etalon::TextureAspect::All,
            },
            luma_size,
        );
        encoder.copy_texture_to_texture(
            etalon::TexelCopyTextureInfo {
                texture: &nv12,
                mip_level: 0,
                origin: etalon::Origin3d::ZERO,
                aspect: etalon::TextureAspect::Plane1,
            },
            etalon::TexelCopyTextureInfo {
                texture: &rg8,
                mip_level: 0,
                origin: etalon::Origin3d::ZERO,
                aspect: etalon::TextureAspect::All,
            },
            chroma_size,
        );

        encoder.copy_texture_to_buffer(
            etalon::TexelCopyTextureInfo {
                texture: &r8,
                mip_level: 0,
                origin: etalon::Origin3d::ZERO,
                aspect: etalon::TextureAspect::All,
            },
            etalon::TexelCopyBufferInfo {
                buffer: &r8_readback,
                layout: etalon::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(WIDTH),
                    rows_per_image: Some(HEIGHT),
                },
            },
            luma_size,
        );
        encoder.copy_texture_to_buffer(
            etalon::TexelCopyTextureInfo {
                texture: &rg8,
                mip_level: 0,
                origin: etalon::Origin3d::ZERO,
                aspect: etalon::TextureAspect::All,
            },
            etalon::TexelCopyBufferInfo {
                buffer: &rg8_readback,
                layout: etalon::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(WIDTH / 2 * 2),
                    rows_per_image: Some(HEIGHT / 2),
                },
            },
            chroma_size,
        );

        ctx.queue.submit([encoder.finish()]);

        let r8_slice = r8_readback.slice(..);
        r8_slice.map_async(etalon::MapMode::Read, |_| ());
        let rg8_slice = rg8_readback.slice(..);
        rg8_slice.map_async(etalon::MapMode::Read, |_| ());
        ctx.async_poll(etalon::PollType::wait_indefinitely())
            .await
            .unwrap();

        let r8_data: Vec<u8> = r8_slice.get_mapped_range().unwrap().to_vec();
        let rg8_data: Vec<u8> = rg8_slice.get_mapped_range().unwrap().to_vec();
        assert_eq!(r8_data, luma_bytes, "luma plane mismatch");
        assert_eq!(rg8_data, chroma_bytes, "chroma plane mismatch");
    });

/// Ensures that copying P010 texture to P010 texture works as expected
#[apply(gpu_test!)]
static P010_TEXTURE_COPYING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                etalon::Features::TEXTURE_FORMAT_P010 | etalon::Features::TEXTURE_FORMAT_16BIT_NORM,
            )
            .enable_noop(),
    )
    .run_sync(|ctx| {
        // Deliberately non-square so a width/height swap is caught. Both
        // dimensions stay even, as required by NV12/P010 (chroma is half-res).
        let size = etalon::Extent3d {
            width: 256,
            height: 128,
            depth_or_array_layers: 1,
        };
        let input_texture = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            dimension: etalon::TextureDimension::D2,
            size,
            format: etalon::TextureFormat::P010,
            usage: etalon::TextureUsages::COPY_SRC,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let output_texture = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            dimension: etalon::TextureDimension::D2,
            size,
            format: etalon::TextureFormat::P010,
            usage: etalon::TextureUsages::COPY_DST,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });

        let mut command_encoder = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor::default());
        command_encoder.copy_texture_to_texture(
            input_texture.as_image_copy(),
            output_texture.as_image_copy(),
            size,
        );
        ctx.queue.submit([command_encoder.finish()]);
    });
