use etalon_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(CLIP_DISTANCES);
}

#[apply(gpu_test!)]
static CLIP_DISTANCES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(etalon::Features::CLIP_DISTANCES)
            // https://github.com/mediaprophet/etalon/issues/9184
            .expect_fail(
                etalon_test::FailureCase::molten_vk()
                    .validation_error("Shader library compile failed")
                    .validation_error("could not be compiled into pipeline")
                    .unexpected_error("Unexpected Vulkan error: ERROR_INITIALIZATION_FAILED"),
            ),
    )
    .run_async(clip_distances);

async fn clip_distances(ctx: TestingContext) {
    // Create pipeline
    let shader = ctx
        .device
        .create_shader_module(etalon::ShaderModuleDescriptor {
            label: None,
            source: etalon::ShaderSource::Wgsl(SHADER_SRC.into()),
        });
    let pipeline = ctx
        .device
        .create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: etalon::VertexState {
                buffers: &[],
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
            },
            primitive: etalon::PrimitiveState::default(),
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            fragment: Some(etalon::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(etalon::ColorTargetState {
                    format: etalon::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: etalon::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

    // Deliberately non-square so a width/height swap in the copy or the
    // readback index math is caught. `WIDTH` is a multiple of
    // `COPY_BYTES_PER_ROW_ALIGNMENT` so `bytes_per_row` stays aligned.
    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 192;

    // Create render target
    let render_texture = ctx.device.create_texture(&etalon::TextureDescriptor {
        label: Some("Render Texture"),
        size: etalon::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::R8Unorm,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    // Perform render
    let mut encoder = ctx
        .device
        .create_command_encoder(&etalon::CommandEncoderDescriptor::default());
    {
        let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                ops: etalon::Operations {
                    load: etalon::LoadOp::Clear(etalon::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: etalon::StoreOp::Store,
                },
                resolve_target: None,
                view: &render_texture.create_view(&etalon::TextureViewDescriptor::default()),
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&pipeline);
        rpass.draw(0..3, 0..1);
    }

    // Read texture data
    let readback_buffer = ctx.device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: (WIDTH * HEIGHT) as u64,
        usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        etalon::TexelCopyTextureInfo {
            texture: &render_texture,
            mip_level: 0,
            origin: etalon::Origin3d::ZERO,
            aspect: etalon::TextureAspect::All,
        },
        etalon::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: etalon::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH),
                rows_per_image: None,
            },
        },
        etalon::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit([encoder.finish()]);
    let slice = readback_buffer.slice(..);
    slice.map_async(etalon::MapMode::Read, |_| ());
    ctx.async_poll(etalon::PollType::wait_indefinitely())
        .await
        .unwrap();
    let data: &[u8] = &slice.get_mapped_range().unwrap();

    // The shader fills the upper wedge of the framebuffer (`y_ndc >= |x_ndc|`).
    // Sample one point inside the wedge and three outside it. The points are
    // deliberately off-axis and use a non-square stride (`x + y * WIDTH`) so a
    // width/height confusion would read the wrong texel.
    let texel = |x: u32, y: u32| data[(x + y * WIDTH) as usize];
    assert_eq!(
        texel(144, 48),
        0xFF,
        "inside wedge (upper, right of center)"
    );
    assert_eq!(texel(32, 96), 0x00, "outside wedge (far left, middle row)");
    assert_eq!(
        texel(224, 96),
        0x00,
        "outside wedge (far right, middle row)"
    );
    assert_eq!(texel(128, 160), 0x00, "outside wedge (bottom center)");
}

const SHADER_SRC: &str = "
enable clip_distances;
struct VertexOutput {
    @builtin(position) pos: vec4f,
    @builtin(clip_distances) clip_distances: array<f32, 2>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index) / 2) * 4.0 - 1.0;
    let y = f32(i32(vertex_index) & 1) * 4.0 - 1.0;
    out.pos = vec4f(x, y, 0.5, 1.0);
    out.clip_distances[0] = x + y;
    out.clip_distances[1] = y - x;
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(1.0);
}
";
