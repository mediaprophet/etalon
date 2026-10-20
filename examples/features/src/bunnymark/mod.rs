use bytemuck::{Pod, Zeroable};
use nanorand::{Rng, WyRand};
use std::borrow::Cow;
use etalon::util::DeviceExt;
use winit::{
    event::{ElementState, KeyEvent},
    keyboard::{Key, NamedKey},
};

const MAX_BUNNIES: usize = 1 << 20;
const BUNNY_SIZE: f32 = 0.15 * 256.0;
const GRAVITY: f32 = -9.8 * 100.0;
const MAX_VELOCITY: f32 = 750.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    mvp: [[f32; 4]; 4],
    size: [f32; 2],
    pad: [f32; 2],
}

#[repr(C, align(256))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Bunny {
    position: [f32; 2],
    velocity: [f32; 2],
    color: u32,
    _pad: [u32; (256 - 20) / 4],
}

impl Bunny {
    fn update_data(&mut self, delta: f32, extent: &[u32; 2]) {
        self.position[0] += self.velocity[0] * delta;
        self.position[1] += self.velocity[1] * delta;
        self.velocity[1] += GRAVITY * delta;

        if (self.velocity[0] > 0.0 && self.position[0] + 0.5 * BUNNY_SIZE > extent[0] as f32)
            || (self.velocity[0] < 0.0 && self.position[0] - 0.5 * BUNNY_SIZE < 0.0)
        {
            self.velocity[0] *= -1.0;
        }

        if self.velocity[1] < 0.0 && self.position[1] < 0.5 * BUNNY_SIZE {
            self.velocity[1] *= -1.0;
        }

        // Top boundary check
        if self.velocity[1] > 0.0 && self.position[1] + 0.5 * BUNNY_SIZE > extent[1] as f32 {
            self.velocity[1] *= -1.0;
        }
    }
}

/// Example struct holds references to etalon resources and frame persistent data
struct Example {
    view: etalon::TextureView,
    sampler: etalon::Sampler,
    global_bind_group_layout: etalon::BindGroupLayout,
    global_group: etalon::BindGroup,
    local_group: etalon::BindGroup,
    pipeline: etalon::RenderPipeline,
    bunnies: Vec<Bunny>,
    local_buffer: etalon::Buffer,
    extent: [u32; 2],
    rng: WyRand,
}

impl Example {
    fn spawn_bunnies(&mut self) {
        let spawn_count = 64;
        let color = self.rng.generate::<u32>();
        println!(
            "Spawning {} bunnies, total at {}",
            spawn_count,
            self.bunnies.len() + spawn_count
        );
        for _ in 0..spawn_count {
            let speed = self.rng.generate::<f32>() * MAX_VELOCITY - (MAX_VELOCITY * 0.5);
            self.bunnies.push(Bunny {
                position: [0.0, 0.5 * (self.extent[1] as f32)],
                velocity: [speed, 0.0],
                color,
                _pad: Zeroable::zeroed(),
            });
        }
    }

    fn render_inner(
        &mut self,
        view: &etalon::TextureView,
        device: &etalon::Device,
        queue: &etalon::Queue,
    ) {
        let delta = 0.01;
        for bunny in self.bunnies.iter_mut() {
            bunny.update_data(delta, &self.extent);
        }

        let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment;
        queue.write_buffer(&self.local_buffer, 0, bytemuck::cast_slice(&self.bunnies));

        let mut encoder = device.create_command_encoder(&etalon::CommandEncoderDescriptor::default());
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
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.global_group, &[]);
            for i in 0..self.bunnies.len() {
                let offset =
                    (i as etalon::DynamicOffset) * (uniform_alignment as etalon::DynamicOffset);
                rpass.set_bind_group(1, &self.local_group, &[offset]);
                rpass.draw(0..4, 0..1);
            }
        }

        queue.submit(Some(encoder.finish()));
    }
}

impl crate::framework::Example for Example {
    fn init(
        config: &etalon::SurfaceConfiguration,
        _adapter: &etalon::Adapter,
        device: &etalon::Device,
        queue: &etalon::Queue,
    ) -> Self {
        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: None,
            source: etalon::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../../../../etalon-hal/examples/halmark/shader.wgsl"
            ))),
        });

        let global_bind_group_layout =
            device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                entries: &[
                    etalon::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: etalon::ShaderStages::VERTEX,
                        ty: etalon::BindingType::Buffer {
                            ty: etalon::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: etalon::BufferSize::new(size_of::<Globals>() as _),
                        },
                        count: None,
                    },
                    etalon::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: etalon::ShaderStages::FRAGMENT,
                        ty: etalon::BindingType::Texture {
                            sample_type: etalon::TextureSampleType::Float { filterable: true },
                            view_dimension: etalon::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    etalon::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: etalon::ShaderStages::FRAGMENT,
                        ty: etalon::BindingType::Sampler(etalon::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: None,
            });
        let local_bind_group_layout =
            device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                entries: &[etalon::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: etalon::ShaderStages::VERTEX,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: etalon::BufferSize::new(size_of::<Bunny>() as _),
                    },
                    count: None,
                }],
                label: None,
            });
        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&global_bind_group_layout),
                Some(&local_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
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
                    format: config.view_formats[0],
                    blend: Some(etalon::BlendState::ALPHA_BLENDING),
                    write_mask: etalon::ColorWrites::default(),
                })],
            }),
            primitive: etalon::PrimitiveState {
                topology: etalon::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(etalon::IndexFormat::Uint16),
                ..etalon::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let texture = {
            let img_data = include_bytes!("../../../../logo.png");
            let decoder = png::Decoder::new(std::io::Cursor::new(img_data));
            let mut reader = decoder.read_info().unwrap();
            let buf_len = reader
                .output_buffer_size()
                .expect("output buffer would not fit in memory");
            let mut buf = vec![0; buf_len];
            let info = reader.next_frame(&mut buf).unwrap();

            let size = etalon::Extent3d {
                width: info.width,
                height: info.height,
                depth_or_array_layers: 1,
            };
            let texture = device.create_texture(&etalon::TextureDescriptor {
                label: None,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: etalon::TextureDimension::D2,
                format: etalon::TextureFormat::Rgba8UnormSrgb,
                usage: etalon::TextureUsages::COPY_DST | etalon::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            queue.write_texture(
                texture.as_image_copy(),
                &buf,
                etalon::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(info.width * 4),
                    rows_per_image: None,
                },
                size,
            );
            texture
        };

        let sampler = device.create_sampler(&etalon::SamplerDescriptor {
            label: None,
            address_mode_u: etalon::AddressMode::ClampToEdge,
            address_mode_v: etalon::AddressMode::ClampToEdge,
            address_mode_w: etalon::AddressMode::ClampToEdge,
            mag_filter: etalon::FilterMode::Linear,
            min_filter: etalon::FilterMode::Nearest,
            mipmap_filter: etalon::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let globals = Globals {
            mvp: glam::camera::rh::proj::directx::orthographic(
                0.0,
                config.width as f32,
                0.0,
                config.height as f32,
                -1.0,
                1.0,
            )
            .to_cols_array_2d(),
            size: [BUNNY_SIZE; 2],
            pad: [0.0; 2],
        };

        let global_buffer = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("global"),
            contents: bytemuck::bytes_of(&globals),
            usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::UNIFORM,
        });
        let uniform_alignment =
            device.limits().min_uniform_buffer_offset_alignment as etalon::BufferAddress;
        let local_buffer = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("local"),
            size: (MAX_BUNNIES as etalon::BufferAddress) * uniform_alignment,
            usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let view = texture.create_view(&etalon::TextureViewDescriptor::default());
        let global_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            layout: &global_bind_group_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: global_buffer.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: etalon::BindingResource::TextureView(&view),
                },
                etalon::BindGroupEntry {
                    binding: 2,
                    resource: etalon::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });
        let local_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            layout: &local_bind_group_layout,
            entries: &[etalon::BindGroupEntry {
                binding: 0,
                resource: etalon::BindingResource::Buffer(etalon::BufferBinding {
                    buffer: &local_buffer,
                    offset: 0,
                    size: etalon::BufferSize::new(size_of::<Bunny>() as _),
                }),
            }],
            label: None,
        });

        let rng = WyRand::new_seed(42);

        let mut ex = Example {
            view,
            sampler,
            global_bind_group_layout,
            pipeline,
            global_group,
            local_group,
            bunnies: Vec::new(),
            local_buffer,
            extent: [config.width, config.height],
            rng,
        };

        ex.spawn_bunnies();

        ex
    }

    fn update(&mut self, event: winit::event::WindowEvent) {
        if let winit::event::WindowEvent::KeyboardInput {
            event:
                KeyEvent {
                    logical_key: Key::Named(NamedKey::Space),
                    state: ElementState::Pressed,
                    ..
                },
            ..
        } = event
        {
            self.spawn_bunnies();
        }
    }

    fn resize(
        &mut self,
        sc_desc: &etalon::SurfaceConfiguration,
        device: &etalon::Device,
        _queue: &etalon::Queue,
    ) {
        self.extent = [sc_desc.width, sc_desc.height];

        let globals = Globals {
            mvp: glam::camera::rh::proj::directx::orthographic(
                0.0,
                sc_desc.width as f32,
                0.0,
                sc_desc.height as f32,
                -1.0,
                1.0,
            )
            .to_cols_array_2d(),
            size: [BUNNY_SIZE; 2],
            pad: [0.0; 2],
        };

        let global_buffer = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("global"),
            contents: bytemuck::bytes_of(&globals),
            usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::UNIFORM,
        });

        let global_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            layout: &self.global_bind_group_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: global_buffer.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: etalon::BindingResource::TextureView(&self.view),
                },
                etalon::BindGroupEntry {
                    binding: 2,
                    resource: etalon::BindingResource::Sampler(&self.sampler),
                },
            ],
            label: None,
        });
        self.global_group = global_group;
    }

    fn render(&mut self, view: &etalon::TextureView, device: &etalon::Device, queue: &etalon::Queue) {
        self.render_inner(view, device, queue);
    }
}

pub fn main() {
    crate::framework::run::<Example>("bunnymark");
}

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "bunnymark",
    image_path: "/examples/features/src/bunnymark/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::default(),
    base_test_parameters: etalon_test::TestParameters::default(),
    // We're looking for very small differences, so look in the high percentiles.
    comparisons: &[
        etalon_test::ComparisonType::Mean(0.05),
        etalon_test::ComparisonType::Percentile {
            percentile: 0.99,
            threshold: 0.37,
        },
    ],
    _phantom: std::marker::PhantomData::<Example>,
};
