use etalon_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(RESOLVE_WITH_TRANSIENT);
}

#[apply(gpu_test!)]
static RESOLVE_WITH_TRANSIENT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| async move {
        const SIZE: etalon::Extent3d = etalon::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        };

        let shader_src = "
            @vertex
            fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                let positions: array<vec2f, 3> = array<vec2f, 3>(
                    vec2f(-1.0, -1.0),
                    vec2f(-1.0, 3.0),
                    vec2f(3.0, -1.0)
                );
                return vec4f(positions[index], 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4f {
                return vec4f(1.0);
            }
        ";

        let shader = ctx
            .device
            .create_shader_module(etalon::ShaderModuleDescriptor {
                label: None,
                source: etalon::ShaderSource::Wgsl(shader_src.into()),
            });

        let pipeline_desc = etalon::RenderPipelineDescriptor {
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
            multisample: etalon::MultisampleState {
                count: 4,
                mask: !0,
                alpha_to_coverage_enabled: false,
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
            multiview_mask: None,
            cache: None,
        };
        let pipeline = ctx.device.create_render_pipeline(&pipeline_desc);

        let transient_texture = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: SIZE,
            mip_level_count: 1,
            sample_count: 4,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rgba8Unorm,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT
                | etalon::TextureUsages::TRANSIENT_ATTACHMENT,
            view_formats: &[],
        });

        let target_texture = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: SIZE,
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rgba8Unorm,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let readback_buffer = ctx.device.create_buffer(&etalon::BufferDescriptor {
            label: None,
            size: 256 * 256 * 4,
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
                    view: &transient_texture.create_view(&etalon::TextureViewDescriptor::default()),
                    depth_slice: None,
                    resolve_target: Some(
                        &target_texture.create_view(&etalon::TextureViewDescriptor::default()),
                    ),
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(etalon::Color::TRANSPARENT),
                        store: etalon::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&pipeline);
            rpass.draw(0..3, 0..1);
        }

        encoder.copy_texture_to_buffer(
            etalon::TexelCopyTextureInfo {
                texture: &target_texture,
                mip_level: 0,
                origin: etalon::Origin3d { x: 0, y: 0, z: 0 },
                aspect: etalon::TextureAspect::All,
            },
            etalon::TexelCopyBufferInfo {
                buffer: &readback_buffer,
                layout: etalon::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256 * 4),
                    rows_per_image: Some(256),
                },
            },
            SIZE,
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
    });
