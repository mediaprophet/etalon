#[cfg(not(target_arch = "wasm32"))]
use crate::utils::output_image_native;
#[cfg(target_arch = "wasm32")]
use crate::utils::output_image_wasm;

const TEXTURE_DIMS: (usize, usize) = (512, 512);

async fn run(_path: Option<String>) {
    // This will later store the raw pixel value data locally. We'll create it now as
    // a convenient size reference.
    let mut texture_data = Vec::<u8>::with_capacity(TEXTURE_DIMS.0 * TEXTURE_DIMS.1 * 4);

    let instance = etalon::Instance::default();
    let adapter = instance
        .request_adapter(&etalon::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&etalon::DeviceDescriptor {
            label: None,
            required_features: etalon::Features::empty(),
            required_limits: etalon::Limits::downlevel_defaults(),
            default_queue: etalon::QueueDescriptor { label: None },
            experimental_features: etalon::ExperimentalFeatures::disabled(),
            memory_hints: etalon::MemoryHints::MemoryUsage,
            trace: etalon::Trace::Off,
        })
        .await
        .unwrap();

    let shader = device.create_shader_module(etalon::include_wgsl!("shader.wgsl"));

    let render_target = device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size: etalon::Extent3d {
            width: TEXTURE_DIMS.0 as u32,
            height: TEXTURE_DIMS.1 as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Rgba8UnormSrgb,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::COPY_SRC,
        view_formats: &[etalon::TextureFormat::Rgba8UnormSrgb],
    });
    let output_staging_buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: texture_data.capacity() as u64,
        usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
        label: None,
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
            targets: &[Some(etalon::TextureFormat::Rgba8UnormSrgb.into())],
        }),
        primitive: etalon::PrimitiveState::default(),
        depth_stencil: None,
        multisample: etalon::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    log::info!("etalon context set up.");

    //-----------------------------------------------

    let texture_view = render_target.create_view(&etalon::TextureViewDescriptor::default());

    let mut command_encoder =
        device.create_command_encoder(&etalon::CommandEncoderDescriptor::default());
    {
        let mut render_pass = command_encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                view: &texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: etalon::Operations {
                    load: etalon::LoadOp::Clear(etalon::Color::GREEN),
                    store: etalon::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(&pipeline);
        render_pass.draw(0..3, 0..1);
    }
    // The texture now contains our rendered image
    command_encoder.copy_texture_to_buffer(
        etalon::TexelCopyTextureInfo {
            texture: &render_target,
            mip_level: 0,
            origin: etalon::Origin3d::ZERO,
            aspect: etalon::TextureAspect::All,
        },
        etalon::TexelCopyBufferInfo {
            buffer: &output_staging_buffer,
            layout: etalon::TexelCopyBufferLayout {
                offset: 0,
                // This needs to be a multiple of 256. Normally we would need to pad
                // it but we here know it will work out anyways.
                bytes_per_row: Some((TEXTURE_DIMS.0 * 4) as u32),
                rows_per_image: Some(TEXTURE_DIMS.1 as u32),
            },
        },
        etalon::Extent3d {
            width: TEXTURE_DIMS.0 as u32,
            height: TEXTURE_DIMS.1 as u32,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(command_encoder.finish()));
    log::info!("Commands submitted.");

    //-----------------------------------------------

    // Time to get our image.
    let buffer_slice = output_staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(etalon::MapMode::Read, move |r| sender.send(r).unwrap());
    device.poll(etalon::PollType::wait_indefinitely()).unwrap();
    receiver.recv_async().await.unwrap().unwrap();
    log::info!("Output buffer mapped.");
    {
        let view = buffer_slice.get_mapped_range().unwrap();
        texture_data.extend_from_slice(&view[..]);
    }
    log::info!("Image data copied to local.");
    output_staging_buffer.unmap();

    #[cfg(not(target_arch = "wasm32"))]
    output_image_native(texture_data.to_vec(), TEXTURE_DIMS, _path.unwrap());
    #[cfg(target_arch = "wasm32")]
    output_image_wasm(texture_data.to_vec(), TEXTURE_DIMS);
    log::info!("Done.");
}

pub fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .format_timestamp_nanos()
            .init();

        let path = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "please_don't_git_push_me.png".to_string());
        pollster::block_on(run(Some(path)));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run(None));
    }
}
