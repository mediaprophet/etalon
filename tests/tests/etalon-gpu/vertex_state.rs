use etalon::{
    util::{BufferInitDescriptor, DeviceExt},
    vertex_attr_array,
};
use etalon_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(SET_ARRAY_STRIDE_TO_0);
}

#[apply(gpu_test!)]
static SET_ARRAY_STRIDE_TO_0: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .limits(etalon::Limits::downlevel_defaults())
            // https://github.com/mediaprophet/etalon/issues/9184
            .expect_fail(
                etalon_test::FailureCase::molten_vk()
                    .validation_error("vertexAttributeAccessBeyondStride"),
            ),
    )
    .run_async(set_array_stride_to_0);

/// Tests that draws using a vertex buffer with stride of 0 works correctly (especially on the
/// D3D12 backend; see commentary within).
async fn set_array_stride_to_0(ctx: TestingContext) {
    let position_buffer_content: &[f32; 12] = &[
        // Triangle 1
        -1.0, -1.0, // Bottom left
        1.0, 1.0, // Top right
        -1.0, 1.0, // Top left
        // Triangle 2
        -1.0, -1.0, // Bottom left
        1.0, -1.0, // Bottom right
        1.0, 1.0, // Top right
    ];
    let position_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice::<f32, u8>(position_buffer_content),
        usage: etalon::BufferUsages::VERTEX,
    });

    let color_buffer_content: &[f32; 4] = &[1.0, 1.0, 1.0, 1.0];
    let color_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice::<f32, u8>(color_buffer_content),
        usage: etalon::BufferUsages::VERTEX,
    });

    let shader_src = "
        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) color: vec4f,
        }

        @vertex
        fn vs_main(@location(0) position: vec2f, @location(1) color: vec4f) -> VertexOutput {
            return VertexOutput(vec4f(position, 0.0, 1.0), color);
        }

        @fragment
        fn fs_main(@location(0) color: vec4f) -> @location(0) vec4f {
            return color;
        }
    ";

    let shader = ctx
        .device
        .create_shader_module(etalon::ShaderModuleDescriptor {
            label: None,
            source: etalon::ShaderSource::Wgsl(shader_src.into()),
        });

    let vbl = [
        Some(etalon::VertexBufferLayout {
            array_stride: 8,
            step_mode: etalon::VertexStepMode::Vertex,
            attributes: &vertex_attr_array![0 => Float32x2],
        }),
        Some(etalon::VertexBufferLayout {
            array_stride: 0,
            step_mode: etalon::VertexStepMode::Vertex,
            attributes: &vertex_attr_array![1 => Float32x4],
        }),
    ];
    let pipeline_desc = etalon::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: etalon::VertexState {
            buffers: &vbl,
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
    };
    let mut first_pipeline_desc = pipeline_desc.clone();
    let mut first_vbl = vbl.clone();
    first_vbl[1].as_mut().unwrap().array_stride = 16;
    first_pipeline_desc.vertex.buffers = &first_vbl;
    let pipeline = ctx.device.create_render_pipeline(&pipeline_desc);
    let first_pipeline = ctx.device.create_render_pipeline(&first_pipeline_desc);

    let out_texture = ctx.device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size: etalon::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::R8Unorm,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_texture_view = out_texture.create_view(&etalon::TextureViewDescriptor::default());

    let readback_buffer = ctx.device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: 256 * 256,
        usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
        mapped_at_creation: false,
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
                view: &out_texture_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // The D3D12 backend used to not set the stride of vertex buffers if it was 0.
        rpass.set_pipeline(&first_pipeline); // This call caused the D3D12 backend to set the stride for the 2nd vertex buffer to 16.
        rpass.set_pipeline(&pipeline); // This call doesn't set the stride for the 2nd vertex buffer to 0.
        rpass.set_vertex_buffer(0, position_buffer.slice(..));
        rpass.set_vertex_buffer(1, color_buffer.slice(..));
        rpass.draw(0..6, 0..1); // Causing this draw to be skipped since it would read OOB of the 2nd vertex buffer.
    }

    encoder.copy_texture_to_buffer(
        etalon::TexelCopyTextureInfo {
            texture: &out_texture,
            mip_level: 0,
            origin: etalon::Origin3d::ZERO,
            aspect: etalon::TextureAspect::All,
        },
        etalon::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: etalon::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        etalon::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
    );

    ctx.queue.submit([encoder.finish()]);

    let slice = readback_buffer.slice(..);
    slice.map_async(etalon::MapMode::Read, |_| ());

    ctx.async_poll(etalon::PollType::wait_indefinitely())
        .await
        .unwrap();

    let data = slice.get_mapped_range().unwrap();
    let succeeded = data.iter().all(|b| *b == u8::MAX);
    assert!(succeeded);
}
