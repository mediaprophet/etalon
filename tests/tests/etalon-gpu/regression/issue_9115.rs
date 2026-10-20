use etalon::util::DeviceExt;
use etalon_test::{
    apply, gpu_test, image::ReadbackBuffers, GpuTestConfiguration, GpuTestInitializer,
    TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(IMMEDIATES_WITH_UNIFORM_IN_SINGLE_MODULE);
}

/// On the GLES backend, using immediates in both vertex and fragment shaders from a single
/// shader module, while also having a uniform buffer with a struct type, caused a panic.
/// The backend incorrectly encountered the uniform's struct type when processing immediates.
///
/// See <https://github.com/mediaprophet/etalon/issues/9115>.
#[apply(gpu_test!)]
static IMMEDIATES_WITH_UNIFORM_IN_SINGLE_MODULE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(etalon::Features::IMMEDIATES)
            .limits(etalon::Limits {
                max_immediate_size: 32,
                ..Default::default()
            }),
    )
    .run_async(immediates_with_uniform_in_single_module);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Immediates {
    position: [f32; 2],
    size: [f32; 2],
    color: [f32; 4],
}

async fn immediates_with_uniform_in_single_module(ctx: TestingContext) {
    let shader = ctx
        .device
        .create_shader_module(etalon::include_wgsl!("issue_9115.wgsl"));

    let globals_buffer = ctx
        .device
        .create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("globals"),
            contents: bytemuck::cast_slice(&[1.0_f32, 1.0]),
            usage: etalon::BufferUsages::UNIFORM,
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
            resource: globals_buffer.as_entire_binding(),
        }],
    });

    let pll = ctx
        .device
        .create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("pll"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: size_of::<Immediates>() as u32,
        });

    let pipeline = ctx
        .device
        .create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pll),
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
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
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
        rpass.set_immediates(
            0,
            bytemuck::cast_slice(&[Immediates {
                position: [0.0, 0.0],
                size: [1.0, 1.0],
                color: [0.0, 1.0, 0.0, 1.0],
            }]),
        );
        rpass.draw(0..3, 0..1);
    }

    let buffers = ReadbackBuffers::new(&ctx.device, &texture);
    buffers.copy_from(&ctx.device, &mut encoder, &texture);
    ctx.queue.submit([encoder.finish()]);

    // The fragment shader outputs immediates.color, which is green (0, 1, 0, 1).
    let expected: [u8; 4] = [0, 255, 0, 255];
    buffers.assert_buffer_contents(&ctx, &expected).await;
}
