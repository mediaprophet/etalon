use etalon::{util::DeviceExt, DownlevelFlags, Limits, TextureFormat};
use etalon_test::{apply, gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

pub fn all_tests(vec: &mut Vec<etalon_test::GpuTestInitializer>) {
    vec.push(REINTERPRET_SRGB);
}

#[apply(gpu_test!)]
static REINTERPRET_SRGB: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::VIEW_FORMATS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        let unorm_data: [[u8; 4]; 4] = [
            [180, 0, 0, 255],
            [0, 84, 0, 127],
            [0, 0, 62, 100],
            [62, 180, 84, 90],
        ];
        let srgb_data: [[u8; 4]; 4] = [
            [116, 0, 0, 255],
            [0, 23, 0, 127],
            [0, 0, 12, 100],
            [12, 116, 23, 90],
        ];

        let size = etalon::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        };

        let shader = ctx
            .device
            .create_shader_module(etalon::include_wgsl!("view_format.wgsl"));

        // Reinterpret Rgba8Unorm as Rgba8UnormSrgb
        reinterpret(
            &ctx,
            &shader,
            size,
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb,
            &unorm_data,
            &srgb_data,
        )
        .await;

        // Reinterpret Rgba8UnormSrgb back to Rgba8Unorm
        reinterpret(
            &ctx,
            &shader,
            size,
            TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Rgba8Unorm,
            &srgb_data,
            &unorm_data,
        )
        .await;
    });

async fn reinterpret(
    ctx: &TestingContext,
    shader: &etalon::ShaderModule,
    size: etalon::Extent3d,
    src_format: etalon::TextureFormat,
    reinterpret_to: etalon::TextureFormat,
    src_data: &[[u8; 4]],
    expect_data: &[[u8; 4]],
) {
    let tex = ctx.device.create_texture_with_data(
        &ctx.queue,
        &etalon::TextureDescriptor {
            label: None,
            dimension: etalon::TextureDimension::D2,
            size,
            format: src_format,
            usage: etalon::TextureUsages::COPY_DST | etalon::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[reinterpret_to],
        },
        etalon::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(src_data),
    );
    let tv = tex.create_view(&etalon::TextureViewDescriptor {
        format: Some(reinterpret_to),
        ..Default::default()
    });
    let pipeline = ctx
        .device
        .create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("reinterpret pipeline"),
            layout: None,
            vertex: etalon::VertexState {
                module: shader,
                entry_point: Some("vs_main"),

                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(etalon::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(src_format.into())],
            }),
            primitive: etalon::PrimitiveState {
                front_face: etalon::FrontFace::Cw,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
    let bind_group = ctx.device.create_bind_group(&etalon::BindGroupDescriptor {
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[etalon::BindGroupEntry {
            binding: 0,
            resource: etalon::BindingResource::TextureView(&tv),
        }],
        label: None,
    });

    let target_tex = ctx.device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: src_format,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::COPY_SRC,
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
    rpass.draw(0..3, 0..1);
    drop(rpass);
    ctx.queue.submit(Some(encoder.finish()));

    let read_buffer = ctx.device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: etalon::COPY_BYTES_PER_ROW_ALIGNMENT as u64 * 2,
        usage: etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        etalon::TexelCopyTextureInfo {
            texture: &target_tex,
            mip_level: 0,
            origin: etalon::Origin3d::ZERO,
            aspect: etalon::TextureAspect::All,
        },
        etalon::TexelCopyBufferInfo {
            buffer: &read_buffer,
            layout: etalon::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(etalon::COPY_BYTES_PER_ROW_ALIGNMENT),
                rows_per_image: None,
            },
        },
        size,
    );
    ctx.queue.submit(Some(encoder.finish()));

    let slice = read_buffer.slice(..);
    slice.map_async(etalon::MapMode::Read, |_| ());
    ctx.async_poll(etalon::PollType::wait_indefinitely())
        .await
        .unwrap();

    let data: Vec<u8> = slice.get_mapped_range().unwrap().to_vec();
    let tolerance_data: [[u8; 4]; 4] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0]];

    for h in 0..size.height {
        let offset = h * etalon::COPY_BYTES_PER_ROW_ALIGNMENT;
        for w in 0..size.width {
            let expect = expect_data[(h * size.width + w) as usize];
            let tolerance = tolerance_data[(h * size.width + w) as usize];
            let index = (w * 4 + offset) as usize;
            if expect[0].abs_diff(data[index]) > tolerance[0]
                || expect[1].abs_diff(data[index + 1]) > tolerance[1]
                || expect[2].abs_diff(data[index + 2]) > tolerance[2]
                || expect[3].abs_diff(data[index + 3]) > tolerance[3]
            {
                panic!(
                    "Reinterpret {:?} as {:?} mismatch! expect {:?} get [{}, {}, {}, {}]",
                    src_format,
                    reinterpret_to,
                    expect,
                    data[index],
                    data[index + 1],
                    data[index + 2],
                    data[index + 3]
                )
            }
        }
    }
}
