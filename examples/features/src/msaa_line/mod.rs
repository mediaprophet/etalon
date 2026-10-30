//! The parts of this example enabling MSAA are:
//! *    The render pipeline is created with a sample_count > 1.
//! *    A new texture with a sample_count > 1 is created and set as the color_attachment instead of the swapchain.
//! *    The swapchain is now specified as a resolve_target.
//!
//! The parts of this example enabling LineList are:
//! *   Set the primitive_topology to PrimitiveTopology::LineList.
//! *   Vertices and Indices describe the two points that make up a line.

use std::iter;

use bytemuck::{Pod, Zeroable};
use etalon::util::DeviceExt;

use winit::{
    event::{ElementState, KeyEvent, WindowEvent},
    keyboard::{Key, NamedKey},
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 2],
    _color: [f32; 4],
}

struct Example {
    bundle: etalon::RenderBundle,
    shader: etalon::ShaderModule,
    pipeline_layout: etalon::PipelineLayout,
    multisampled_framebuffer: etalon::TextureView,
    vertex_buffer: etalon::Buffer,
    vertex_count: u32,
    sample_count: u32,
    rebuild_bundle: bool,
    config: etalon::SurfaceConfiguration,
    max_sample_count: u32,
}

impl Example {
    fn create_bundle(
        device: &etalon::Device,
        config: &etalon::SurfaceConfiguration,
        shader: &etalon::ShaderModule,
        pipeline_layout: &etalon::PipelineLayout,
        sample_count: u32,
        vertex_buffer: &etalon::Buffer,
        vertex_count: u32,
    ) -> etalon::RenderBundle {
        log::info!("sample_count: {sample_count}");
        let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: None,
            layout: Some(pipeline_layout),
            vertex: etalon::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(etalon::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as etalon::BufferAddress,
                    step_mode: etalon::VertexStepMode::Vertex,
                    attributes: &etalon::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
                })],
            },
            fragment: Some(etalon::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: etalon::PrimitiveState {
                topology: etalon::PrimitiveTopology::LineList,
                front_face: etalon::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: etalon::MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        });
        let mut encoder =
            device.create_render_bundle_encoder(&etalon::RenderBundleEncoderDescriptor {
                label: None,
                color_formats: &[Some(config.view_formats[0])],
                depth_stencil: None,
                sample_count,
                multiview: None,
            });
        encoder.set_pipeline(&pipeline);
        encoder.set_vertex_buffer(0, vertex_buffer.slice(..));
        encoder.draw(0..vertex_count, 0..1);
        encoder.finish(&etalon::RenderBundleDescriptor {
            label: Some("main"),
        })
    }

    fn create_multisampled_framebuffer(
        device: &etalon::Device,
        config: &etalon::SurfaceConfiguration,
        sample_count: u32,
    ) -> etalon::TextureView {
        let multisampled_texture_extent = etalon::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let multisampled_frame_descriptor = &etalon::TextureDescriptor {
            size: multisampled_texture_extent,
            mip_level_count: 1,
            sample_count,
            dimension: etalon::TextureDimension::D2,
            format: config.view_formats[0],
            usage: etalon::TextureUsages::RENDER_ATTACHMENT
                | etalon::TextureUsages::TRANSIENT_ATTACHMENT,
            label: None,
            view_formats: &[],
        };

        device
            .create_texture(multisampled_frame_descriptor)
            .create_view(&etalon::TextureViewDescriptor::default())
    }
}

impl crate::framework::Example for Example {
    fn optional_features() -> etalon::Features {
        etalon::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
    }

    fn init(
        config: &etalon::SurfaceConfiguration,
        _adapter: &etalon::Adapter,
        device: &etalon::Device,
        _queue: &etalon::Queue,
    ) -> Self {
        log::info!("Press left/right arrow keys to change sample_count.");

        let sample_flags = _adapter
            .get_texture_format_features(config.view_formats[0])
            .flags;

        let max_sample_count = {
            if sample_flags.contains(etalon::TextureFormatFeatureFlags::MULTISAMPLE_X16) {
                16
            } else if sample_flags.contains(etalon::TextureFormatFeatureFlags::MULTISAMPLE_X8) {
                8
            } else if sample_flags.contains(etalon::TextureFormatFeatureFlags::MULTISAMPLE_X4) {
                4
            } else if sample_flags.contains(etalon::TextureFormatFeatureFlags::MULTISAMPLE_X2) {
                2
            } else {
                1
            }
        };

        let sample_count = max_sample_count;

        let shader = device.create_shader_module(etalon::include_wgsl!("shader.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let multisampled_framebuffer =
            Example::create_multisampled_framebuffer(device, config, sample_count);

        let mut vertex_data = vec![];

        let max = 50;
        for i in 0..max {
            let percent = i as f32 / max as f32;
            let (sin, cos) = (percent * 2.0 * std::f32::consts::PI).sin_cos();
            vertex_data.push(Vertex {
                _pos: [0.0, 0.0],
                _color: [1.0, -sin, cos, 1.0],
            });
            vertex_data.push(Vertex {
                _pos: [1.0 * cos, 1.0 * sin],
                _color: [sin, -cos, 1.0, 1.0],
            });
        }

        let vertex_buffer = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: etalon::BufferUsages::VERTEX,
        });
        let vertex_count = vertex_data.len() as u32;

        let bundle = Example::create_bundle(
            device,
            config,
            &shader,
            &pipeline_layout,
            sample_count,
            &vertex_buffer,
            vertex_count,
        );

        Example {
            bundle,
            shader,
            pipeline_layout,
            multisampled_framebuffer,
            vertex_buffer,
            vertex_count,
            sample_count,
            max_sample_count,
            rebuild_bundle: false,
            config: config.clone(),
        }
    }

    #[expect(clippy::single_match)]
    fn update(&mut self, event: winit::event::WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match logical_key {
                // Cycle supported MSAA sample counts (1 vs max)
                Key::Named(NamedKey::ArrowLeft) if self.sample_count == self.max_sample_count => {
                    self.sample_count = 1;
                    self.rebuild_bundle = true;
                }
                Key::Named(NamedKey::ArrowRight) if self.sample_count == 1 => {
                    self.sample_count = self.max_sample_count;
                    self.rebuild_bundle = true;
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn resize(
        &mut self,
        config: &etalon::SurfaceConfiguration,
        device: &etalon::Device,
        _queue: &etalon::Queue,
    ) {
        self.config = config.clone();
        self.multisampled_framebuffer =
            Example::create_multisampled_framebuffer(device, config, self.sample_count);
    }

    fn render(
        &mut self,
        view: &etalon::TextureView,
        device: &etalon::Device,
        queue: &etalon::Queue,
    ) {
        if self.rebuild_bundle {
            self.bundle = Example::create_bundle(
                device,
                &self.config,
                &self.shader,
                &self.pipeline_layout,
                self.sample_count,
                &self.vertex_buffer,
                self.vertex_count,
            );
            self.multisampled_framebuffer =
                Example::create_multisampled_framebuffer(device, &self.config, self.sample_count);
            self.rebuild_bundle = false;
        }

        let mut encoder =
            device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
        {
            let rpass_color_attachment = if self.sample_count == 1 {
                etalon::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(etalon::Color::BLACK),
                        store: etalon::StoreOp::Store,
                    },
                }
            } else {
                etalon::RenderPassColorAttachment {
                    view: &self.multisampled_framebuffer,
                    depth_slice: None,
                    resolve_target: Some(view),
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(etalon::Color::BLACK),
                        // Storing pre-resolve MSAA data is unnecessary if it isn't used later.
                        // On tile-based GPU, avoid store can reduce your app's memory footprint.
                        store: etalon::StoreOp::Discard,
                    },
                }
            };

            encoder
                .begin_render_pass(&etalon::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(rpass_color_attachment)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .execute_bundles(iter::once(&self.bundle));
        }

        queue.submit(iter::once(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("msaa-line");
}

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "msaa-line",
    image_path: "/examples/features/src/msaa_line/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
    base_test_parameters: etalon_test::TestParameters::default(),
    // There's a lot of natural variance so we check the weighted median too to differentiate
    // real failures from variance.
    comparisons: &[
        etalon_test::ComparisonType::Mean(0.065),
        etalon_test::ComparisonType::Percentile {
            percentile: 0.5,
            threshold: 0.29,
        },
    ],
    _phantom: std::marker::PhantomData::<Example>,
};
