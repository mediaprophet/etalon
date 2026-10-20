use bytemuck::{Pod, Zeroable};
use std::f32::consts;
use etalon::util::DeviceExt;

const TEXTURE_FORMAT: etalon::TextureFormat = etalon::TextureFormat::Rgba8UnormSrgb;
const MIP_LEVEL_COUNT: u32 = 10;
const MIP_PASS_COUNT: u32 = MIP_LEVEL_COUNT - 1;

const QUERY_FEATURES: etalon::Features = {
    etalon::Features::TIMESTAMP_QUERY
        .union(etalon::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
        .union(etalon::Features::PIPELINE_STATISTICS_QUERY)
};

fn create_texels(size: usize, cx: f32, cy: f32) -> Vec<u8> {
    use std::iter;

    (0..size * size)
        .flat_map(|id| {
            // get high five for recognizing this ;)
            let mut x = 4.0 * (id % size) as f32 / (size - 1) as f32 - 2.0;
            let mut y = 2.0 * (id / size) as f32 / (size - 1) as f32 - 1.0;
            let mut count = 0;
            while count < 0xFF && x * x + y * y < 4.0 {
                let old_x = x;
                x = x * x - y * y + cx;
                y = 2.0 * old_x * y + cy;
                count += 1;
            }
            iter::once(0xFF - (count * 2) as u8)
                .chain(iter::once(0xFF - (count * 5) as u8))
                .chain(iter::once(0xFF - (count * 13) as u8))
                .chain(iter::once(u8::MAX))
        })
        .collect()
}

struct QuerySets {
    timestamp: etalon::QuerySet,
    timestamp_period: f32,
    pipeline_statistics: etalon::QuerySet,
    data_buffer: etalon::Buffer,
    mapping_buffer: etalon::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TimestampData {
    start: u64,
    end: u64,
}

type TimestampQueries = [TimestampData; MIP_PASS_COUNT as usize];
type PipelineStatisticsQueries = [u64; MIP_PASS_COUNT as usize];

fn pipeline_statistics_offset() -> etalon::BufferAddress {
    (size_of::<TimestampQueries>() as etalon::BufferAddress).max(etalon::QUERY_RESOLVE_BUFFER_ALIGNMENT)
}

struct Example {
    bind_group: etalon::BindGroup,
    uniform_buf: etalon::Buffer,
    draw_pipeline: etalon::RenderPipeline,
}

impl Example {
    fn generate_matrix(aspect_ratio: f32) -> glam::Mat4 {
        let projection = glam::camera::rh::proj::directx::perspective(
            consts::FRAC_PI_4,
            aspect_ratio,
            1.0,
            1000.0,
        );
        let view = glam::camera::rh::view::look_at_mat4(
            glam::Vec3::new(0f32, 0.0, 10.0),
            glam::Vec3::new(0f32, 50.0, 0.0),
            glam::Vec3::Z,
        );
        projection * view
    }

    fn generate_mipmaps(
        encoder: &mut etalon::CommandEncoder,
        device: &etalon::Device,
        texture: &etalon::Texture,
        query_sets: &Option<QuerySets>,
        mip_count: u32,
    ) {
        let shader = device.create_shader_module(etalon::include_wgsl!("blit.wgsl"));

        let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("blit"),
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
                targets: &[Some(TEXTURE_FORMAT.into())],
            }),
            primitive: etalon::PrimitiveState {
                topology: etalon::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let sampler = device.create_sampler(&etalon::SamplerDescriptor {
            label: Some("mip"),
            address_mode_u: etalon::AddressMode::ClampToEdge,
            address_mode_v: etalon::AddressMode::ClampToEdge,
            address_mode_w: etalon::AddressMode::ClampToEdge,
            mag_filter: etalon::FilterMode::Linear,
            min_filter: etalon::FilterMode::Linear,
            mipmap_filter: etalon::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let views = (0..mip_count)
            .map(|mip| {
                texture.create_view(&etalon::TextureViewDescriptor {
                    label: Some("mip"),
                    format: None,
                    dimension: None,
                    usage: None,
                    aspect: etalon::TextureAspect::All,
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: None,
                    swizzle: etalon::TextureComponentSwizzle::default(),
                })
            })
            .collect::<Vec<_>>();

        for target_mip in 1..mip_count as usize {
            let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    etalon::BindGroupEntry {
                        binding: 0,
                        resource: etalon::BindingResource::TextureView(&views[target_mip - 1]),
                    },
                    etalon::BindGroupEntry {
                        binding: 1,
                        resource: etalon::BindingResource::Sampler(&sampler),
                    },
                ],
                label: None,
            });

            let pipeline_query_index_base = target_mip as u32 - 1;
            let timestamp_query_index_base = (target_mip as u32 - 1) * 2;

            let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(etalon::RenderPassColorAttachment {
                    view: &views[target_mip],
                    depth_slice: None,
                    resolve_target: None,
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(etalon::Color::WHITE),
                        store: etalon::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(ref query_sets) = query_sets {
                rpass.write_timestamp(&query_sets.timestamp, timestamp_query_index_base);
                rpass.begin_pipeline_statistics_query(
                    &query_sets.pipeline_statistics,
                    pipeline_query_index_base,
                );
            }
            rpass.set_pipeline(&pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
            if let Some(ref query_sets) = query_sets {
                rpass.write_timestamp(&query_sets.timestamp, timestamp_query_index_base + 1);
                rpass.end_pipeline_statistics_query();
            }
        }

        if let Some(ref query_sets) = query_sets {
            let timestamp_query_count = MIP_PASS_COUNT * 2;
            encoder.resolve_query_set(
                &query_sets.timestamp,
                0..timestamp_query_count,
                &query_sets.data_buffer,
                0,
            );
            encoder.resolve_query_set(
                &query_sets.pipeline_statistics,
                0..MIP_PASS_COUNT,
                &query_sets.data_buffer,
                pipeline_statistics_offset(),
            );
        }
    }
}

impl crate::framework::Example for Example {
    fn optional_features() -> etalon::Features {
        QUERY_FEATURES
    }

    fn init(
        config: &etalon::SurfaceConfiguration,
        _adapter: &etalon::Adapter,
        device: &etalon::Device,
        queue: &etalon::Queue,
    ) -> Self {
        let mut init_encoder =
            device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });

        // Create the texture
        let size = 1 << MIP_PASS_COUNT;
        let texels = create_texels(size as usize, -0.8, 0.156);
        let texture_extent = etalon::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&etalon::TextureDescriptor {
            size: texture_extent,
            mip_level_count: MIP_LEVEL_COUNT,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: etalon::TextureUsages::TEXTURE_BINDING
                | etalon::TextureUsages::RENDER_ATTACHMENT
                | etalon::TextureUsages::COPY_DST,
            label: None,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&etalon::TextureViewDescriptor::default());
        //Note: we could use queue.write_texture instead, and this is what other
        // examples do, but here we want to show another way to do this.
        let temp_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Temporary Buffer"),
            contents: texels.as_slice(),
            usage: etalon::BufferUsages::COPY_SRC,
        });
        init_encoder.copy_buffer_to_texture(
            etalon::TexelCopyBufferInfo {
                buffer: &temp_buf,
                layout: etalon::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * size),
                    rows_per_image: None,
                },
            },
            texture.as_image_copy(),
            texture_extent,
        );

        // Create other resources
        let sampler = device.create_sampler(&etalon::SamplerDescriptor {
            label: None,
            address_mode_u: etalon::AddressMode::Repeat,
            address_mode_v: etalon::AddressMode::Repeat,
            address_mode_w: etalon::AddressMode::Repeat,
            mag_filter: etalon::FilterMode::Linear,
            min_filter: etalon::FilterMode::Linear,
            mipmap_filter: etalon::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        let uniform_buf = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(mx_ref),
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
        });

        // Create the render pipeline
        let shader = device.create_shader_module(etalon::include_wgsl!("draw.wgsl"));

        let draw_pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("draw"),
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
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: etalon::PrimitiveState {
                topology: etalon::PrimitiveTopology::TriangleStrip,
                front_face: etalon::FrontFace::Ccw,
                cull_mode: Some(etalon::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Create bind group
        let bind_group_layout = draw_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: etalon::BindingResource::TextureView(&texture_view),
                },
                etalon::BindGroupEntry {
                    binding: 2,
                    resource: etalon::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });

        // If both kinds of query are supported, use queries
        let query_sets = if device.features().contains(QUERY_FEATURES) {
            // For N total mips, it takes N - 1 passes to generate them, and we're measuring those.
            let mip_passes = MIP_LEVEL_COUNT - 1;

            // Create the timestamp query set. We need twice as many queries as we have passes,
            // as we need a query at the beginning and at the end of the operation.
            let timestamp = device.create_query_set(&etalon::QuerySetDescriptor {
                label: None,
                count: mip_passes * 2,
                ty: etalon::QueryType::Timestamp,
            });
            // Timestamp queries use an device-specific timestamp unit. We need to figure out how many
            // nanoseconds go by for the timestamp to be incremented by one. The period is this value.
            let timestamp_period = queue.get_timestamp_period();

            // We only need one pipeline statistics query per pass.
            let pipeline_statistics = device.create_query_set(&etalon::QuerySetDescriptor {
                label: None,
                count: mip_passes,
                ty: etalon::QueryType::PipelineStatistics(
                    etalon::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS,
                ),
            });

            // This databuffer has to store all of the query results, 2 * passes timestamp queries
            // and 1 * passes statistics queries. Each query returns a u64 value.
            let buffer_size = pipeline_statistics_offset()
                + size_of::<PipelineStatisticsQueries>() as etalon::BufferAddress;
            let data_buffer = device.create_buffer(&etalon::BufferDescriptor {
                label: Some("query buffer"),
                size: buffer_size,
                usage: etalon::BufferUsages::QUERY_RESOLVE | etalon::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // Mapping buffer
            let mapping_buffer = device.create_buffer(&etalon::BufferDescriptor {
                label: Some("query buffer"),
                size: buffer_size,
                usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            Some(QuerySets {
                timestamp,
                timestamp_period,
                pipeline_statistics,
                data_buffer,
                mapping_buffer,
            })
        } else {
            None
        };

        Self::generate_mipmaps(
            &mut init_encoder,
            device,
            &texture,
            &query_sets,
            MIP_LEVEL_COUNT,
        );

        if let Some(ref query_sets) = query_sets {
            init_encoder.copy_buffer_to_buffer(
                &query_sets.data_buffer,
                0,
                &query_sets.mapping_buffer,
                0,
                query_sets.data_buffer.size(),
            );
        }

        queue.submit(Some(init_encoder.finish()));
        if let Some(ref query_sets) = query_sets {
            // We can ignore the callback as we're about to wait for the device.
            query_sets
                .mapping_buffer
                .slice(..)
                .map_async(etalon::MapMode::Read, |_| ());
            // Wait for device to be done rendering mipmaps
            device.poll(etalon::PollType::wait_indefinitely()).unwrap();
            // This is guaranteed to be ready.
            let timestamp_view = query_sets
                .mapping_buffer
                .slice(..size_of::<TimestampQueries>() as etalon::BufferAddress)
                .get_mapped_range()
                .unwrap();
            let pipeline_stats_view = query_sets
                .mapping_buffer
                .slice(pipeline_statistics_offset()..)
                .get_mapped_range()
                .unwrap();
            // Convert the raw data into a useful structure
            let timestamp_data: &TimestampQueries = bytemuck::from_bytes(&timestamp_view);
            let pipeline_stats_data: &PipelineStatisticsQueries =
                bytemuck::from_bytes(&pipeline_stats_view);
            // Iterate over the data
            for (idx, (timestamp, pipeline)) in timestamp_data
                .iter()
                .zip(pipeline_stats_data.iter())
                .enumerate()
            {
                // Figure out the timestamp differences and multiply by the period to get nanoseconds
                let nanoseconds =
                    (timestamp.end - timestamp.start) as f32 * query_sets.timestamp_period;
                // Nanoseconds is a bit small, so lets use microseconds.
                let microseconds = nanoseconds / 1000.0;
                // Print the data!
                println!(
                    "Generating mip level {} took {:.3} μs and called the fragment shader {} times",
                    idx + 1,
                    microseconds,
                    pipeline
                );
            }
        }

        Example {
            bind_group,
            uniform_buf,
            draw_pipeline,
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        config: &etalon::SurfaceConfiguration,
        _device: &etalon::Device,
        queue: &etalon::Queue,
    ) {
        let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(mx_ref));
    }

    fn render(&mut self, view: &etalon::TextureView, device: &etalon::Device, queue: &etalon::Queue) {
        let mut encoder =
            device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
        {
            let clear_color = etalon::Color {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 1.0,
            };
            let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(etalon::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(clear_color),
                        store: etalon::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.draw_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..4, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("mipmap");
}

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "mipmap",
    image_path: "/examples/features/src/mipmap/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::default(),
    base_test_parameters: etalon_test::TestParameters::default(),
    comparisons: &[etalon_test::ComparisonType::Mean(0.02)],
    _phantom: std::marker::PhantomData::<Example>,
};

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST_QUERY: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "mipmap-query",
    image_path: "/examples/features/src/mipmap/screenshot_query.png",
    width: 1024,
    height: 768,
    optional_features: QUERY_FEATURES,
    base_test_parameters: etalon_test::TestParameters::default(),
    // Somehow, this test on CI lavapipe reasonably often gets error of 0.025341, significantly higher
    // than the comparison we usually do with mean 0.005. This only happens when the query is used.
    comparisons: &[
        etalon_test::ComparisonType::Mean(0.03),
        etalon_test::ComparisonType::Percentile {
            percentile: 0.99,
            threshold: 0.1,
        },
    ],
    _phantom: std::marker::PhantomData::<Example>,
};
