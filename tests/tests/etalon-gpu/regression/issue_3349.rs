use etalon::util::DeviceExt;
use etalon_test::{
    apply, gpu_test, image::ReadbackBuffers, GpuTestConfiguration, GpuTestInitializer,
    TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(MULTI_STAGE_DATA_BINDING);
}

/// We thought we had an OpenGL bug that, when running without explicit in-shader locations,
/// we will not properly bind uniform buffers to both the vertex and fragment
/// shaders. This turned out to not reproduce at all with this test case.
///
/// However, it also caught issues with the immediate data implementation,
/// making sure that it works correctly with different definitions for the immediate data
/// block in vertex and fragment shaders.
///
/// This test needs to be able to run on GLES 3.0
///
/// What this test does is render a 2x2 texture. Each pixel corresponds to a different
/// data source.
///
/// top left: Vertex Shader / Uniform Buffer
/// top right: Vertex Shader / Immediate data
/// bottom left: Fragment Shader / Uniform Buffer
/// bottom right: Fragment Shader / Immediate data
///
/// We then validate the data is correct from every position.
#[apply(gpu_test!)]
static MULTI_STAGE_DATA_BINDING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(etalon::Features::IMMEDIATES)
            .limits(etalon::Limits {
                max_immediate_size: 16,
                ..Default::default()
            }),
    )
    .run_async(multi_stage_data_binding_test);

async fn multi_stage_data_binding_test(ctx: TestingContext) {
    // We use different shader modules to allow us to use different
    // types for the uniform and immediate data blocks between stages.
    let vs_sm = ctx
        .device
        .create_shader_module(etalon::include_wgsl!("issue_3349.vs.wgsl"));

    let fs_sm = ctx
        .device
        .create_shader_module(etalon::include_wgsl!("issue_3349.fs.wgsl"));

    // We start with u8s then convert to float, to make sure we don't have
    // cross-vendor rounding issues unorm.
    let input_as_unorm: [u8; 4] = [25_u8, 50, 75, 100];
    let input = input_as_unorm.map(|v| v as f32 / 255.0);

    let buffer = ctx
        .device
        .create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("buffer"),
            contents: bytemuck::cast_slice(&input),
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[etalon::BindGroupLayoutEntry {
                binding: 0,
                visibility: etalon::ShaderStages::VERTEX_FRAGMENT,
                ty: etalon::BindingType::Buffer {
                    ty: etalon::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let bg = ctx.device.create_bind_group(&etalon::BindGroupDescriptor {
        label: Some("bg"),
        layout: &bgl,
        entries: &[etalon::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let pll = ctx
        .device
        .create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("pll"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 16,
        });

    let pipeline = ctx
        .device
        .create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pll),
            vertex: etalon::VertexState {
                module: &vs_sm,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(etalon::FragmentState {
                module: &fs_sm,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(etalon::ColorTargetState {
                    format: etalon::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: etalon::ColorWrites::ALL,
                })],
            }),
            primitive: etalon::PrimitiveState::default(),
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

    let texture = ctx.device.create_texture(&etalon::TextureDescriptor {
        label: Some("texture"),
        size: etalon::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        // Important: NOT srgb.
        format: etalon::TextureFormat::Rgba8Unorm,
        usage: etalon::TextureUsages::COPY_SRC | etalon::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let view = texture.create_view(&etalon::TextureViewDescriptor::default());

    let mut encoder = ctx
        .device
        .create_command_encoder(&etalon::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

    {
        let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: Some("rpass"),
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: etalon::Operations {
                    load: etalon::LoadOp::Clear(etalon::Color::BLACK),
                    store: etalon::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&pipeline);
        rpass.set_bind_group(0, &bg, &[]);
        rpass.set_immediates(0, bytemuck::cast_slice(&input));
        rpass.draw(0..3, 0..1);
    }

    let buffers = ReadbackBuffers::new(&ctx.device, &texture);
    buffers.copy_from(&ctx.device, &mut encoder, &texture);
    ctx.queue.submit([encoder.finish()]);

    let result = input_as_unorm.repeat(4);
    buffers.assert_buffer_contents(&ctx, &result).await;
}
