use etalon::util::DeviceExt;
use etalon_test::{apply, gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

pub fn all_tests(vec: &mut Vec<etalon_test::GpuTestInitializer>) {
    vec.push(BARYCENTRIC);
    vec.push(BARYCENTRIC_NO_PERSPECTIVE);
}

//
// This test renders one triangle to a 2x2 render target. The triangle
// covers the bottom-left, bottom-right, and the top-left pixel.
// XY layout of the render target, with the triangle:
//
//     (-1,1)  (0,1)  (1,1)
//        +------+------+
//        |      |      |
//        |   o  |      |
//        |   |\ |      |
//        |   | \|      |
// (-1,0) +---|--\------+ (1,0)
//        |   |  |\     |
//        |   |  | \    |
//        |   o--+--o   |
//        |      |      |
//        +------+------+
//     (-1,-1) (0,-1) (1,-1)
//
// The fragment shader outputs color based on builtin(barycentric):
//
//     return vec4<f32>(bary * 1.1 - 0.05, 1.0);
//

#[apply(gpu_test!)]
static BARYCENTRIC: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(etalon::Features::SHADER_BARYCENTRICS),
    )
    .run_async(|ctx| barycentric(ctx, false));

#[apply(gpu_test!)]
static BARYCENTRIC_NO_PERSPECTIVE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(etalon::Features::SHADER_BARYCENTRICS),
    )
    .run_async(|ctx| barycentric(ctx, true));

async fn barycentric(ctx: TestingContext, no_perspective: bool) {
    let shader = ctx
        .device
        .create_shader_module(etalon::include_wgsl!("barycentric.wgsl"));

    let n = -0.505;
    let p = 0.51;
    let triangle_xy: [f32; 6] = [n, n, p, n, n, p];
    let vertex_buffer = ctx
        .device
        .create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&triangle_xy),
            usage: etalon::BufferUsages::VERTEX | etalon::BufferUsages::COPY_DST,
        });

    let indices = [0u32, 1, 2];
    let index_buffer = ctx
        .device
        .create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&indices),
            usage: etalon::BufferUsages::INDEX | etalon::BufferUsages::COPY_DST,
        });

    let pipeline = ctx
        .device
        .create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: etalon::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(etalon::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: etalon::VertexStepMode::Vertex,
                    attributes: &[etalon::VertexAttribute {
                        format: etalon::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                })],
            },
            primitive: etalon::PrimitiveState::default(),
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            fragment: Some(etalon::FragmentState {
                module: &shader,
                entry_point: if no_perspective {
                    Some("fs_main_no_perspective")
                } else {
                    Some("fs_main")
                },
                compilation_options: Default::default(),
                targets: &[Some(etalon::ColorTargetState {
                    format: etalon::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: etalon::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

    let width = 2;
    let height = 2;
    let texture_size = etalon::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let color_texture = ctx.device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Rgba8Unorm,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&etalon::TextureViewDescriptor::default());

    let readback_buffer = etalon_test::image::ReadbackBuffers::new(&ctx.device, &color_texture);

    let mut encoder = ctx
        .device
        .create_command_encoder(&etalon::CommandEncoderDescriptor::default());
    {
        let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                ops: etalon::Operations {
                    load: etalon::LoadOp::Clear(etalon::Color::WHITE),
                    store: etalon::StoreOp::Store,
                },
                resolve_target: None,
                view: &color_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&pipeline);
        rpass.set_index_buffer(index_buffer.slice(..), etalon::IndexFormat::Uint32);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.draw(0..3, 0..1);
    }
    readback_buffer.copy_from(&ctx.device, &mut encoder, &color_texture);
    ctx.queue.submit(Some(encoder.finish()));

    //
    //   +-----+-----+
    //   |blue |white|
    //   +-----+-----+
    //   | red |green|
    //   +-----+-----+
    //
    let expected = [
        0, 0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0, 255, 0, 255,
    ];
    readback_buffer
        .assert_buffer_contents(&ctx, &expected)
        .await;
}
