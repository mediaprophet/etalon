//! Renders different content to different layers of an array texture using multiview,
//! a feature commonly used for VR rendering.

use std::{num::NonZero, time::Instant};

use etalon::util::TextureBlitter;

const TEXTURE_SIZE: u32 = 512;

// Change this to demonstrate non-contiguous multiview functionality
const LAYER_MASK: u32 = 0b11;

const NUM_LAYERS: u32 = 32 - LAYER_MASK.leading_zeros();

pub struct Example {
    pipeline: etalon::RenderPipeline,
    entire_texture_view: etalon::TextureView,
    views: Vec<etalon::TextureView>,
    start_time: Instant,
    blitter: TextureBlitter,
}

impl crate::framework::Example for Example {
    fn init(
        config: &etalon::SurfaceConfiguration,
        _adapter: &etalon::Adapter,
        device: &etalon::Device,
        _queue: &etalon::Queue,
    ) -> Self {
        let shader_src = include_str!("shader.wgsl");

        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: None,
            source: etalon::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: None,
            vertex: etalon::VertexState {
                buffers: &[],
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
            },
            primitive: etalon::PrimitiveState::default(),
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
            multiview_mask: NonZero::new(LAYER_MASK),
            multisample: Default::default(),
            layout: None,
            depth_stencil: None,
            cache: None,
        });
        let texture = device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: etalon::Extent3d {
                width: TEXTURE_SIZE,
                height: TEXTURE_SIZE,
                depth_or_array_layers: NUM_LAYERS,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rgba8Unorm,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT
                | etalon::TextureUsages::COPY_SRC
                | etalon::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let entire_texture_view = texture.create_view(&etalon::TextureViewDescriptor {
            label: None,
            format: Some(etalon::TextureFormat::Rgba8Unorm),
            dimension: Some(etalon::TextureViewDimension::D2Array),
            usage: Some(etalon::TextureUsages::RENDER_ATTACHMENT),
            aspect: etalon::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(NUM_LAYERS),
            swizzle: etalon::TextureComponentSwizzle::default(),
        });
        let mut views = Vec::new();
        for i in 0..NUM_LAYERS {
            views.push(texture.create_view(&etalon::TextureViewDescriptor {
                label: None,
                format: Some(etalon::TextureFormat::Rgba8Unorm),
                dimension: Some(etalon::TextureViewDimension::D2),
                usage: Some(etalon::TextureUsages::TEXTURE_BINDING),
                aspect: etalon::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: i,
                array_layer_count: Some(1),
                swizzle: etalon::TextureComponentSwizzle::default(),
            }));
        }
        let blitter = etalon::util::TextureBlitter::new(device, config.format);
        Self {
            pipeline,
            entire_texture_view,
            views,
            blitter,
            start_time: Instant::now(),
        }
    }

    fn render(&mut self, view: &etalon::TextureView, device: &etalon::Device, queue: &etalon::Queue) {
        let mut encoder = device.create_command_encoder(&etalon::CommandEncoderDescriptor::default());
        {
            let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(etalon::RenderPassColorAttachment {
                    view: &self.entire_texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(etalon::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.02,
                            a: 1.0,
                        }),
                        store: etalon::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: NonZero::new(LAYER_MASK),
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.draw(0..6, 0..1);
        }

        let layer = (Instant::now() - self.start_time).as_secs() % NUM_LAYERS as u64;
        self.blitter
            .copy(device, &mut encoder, &self.views[layer as usize], view);
        queue.submit(Some(encoder.finish()));
    }

    fn required_downlevel_capabilities() -> etalon::DownlevelCapabilities {
        Default::default()
    }

    fn required_features() -> etalon::Features {
        etalon::Features::MULTIVIEW
            | if !(LAYER_MASK + 1).is_power_of_two() {
                etalon::Features::SELECTIVE_MULTIVIEW
            } else {
                etalon::Features::empty()
            }
    }

    fn required_limits() -> etalon::Limits {
        etalon::Limits {
            max_multiview_view_count: NUM_LAYERS,
            ..Default::default()
        }
    }

    fn resize(
        &mut self,
        _config: &etalon::SurfaceConfiguration,
        _device: &etalon::Device,
        _queue: &etalon::Queue,
    ) {
        // empty
    }
    fn update(&mut self, _event: winit::event::WindowEvent) {
        // empty
    }
}

pub fn main() {
    crate::framework::run::<Example>("multiview");
}
