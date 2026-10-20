//! This example demonstrates the basic usage of storage textures for the purpose of
//! creating a digital image of the Mandelbrot set
//! (<https://en.wikipedia.org/wiki/Mandelbrot_set>).
//!
//! Storage textures work like normal textures but they operate similar to storage buffers
//! in that they can be written to. The issue is that as it stands, write-only is the
//! only valid access mode for storage textures in WGSL and although there is a etalon feature
//! to allow for read-write access, this is unfortunately a native-only feature and thus
//! we won't be using it here. If we needed a reference texture, we would need to add a
//! second texture to act as a reference and attach that as well. Luckily, we don't need
//! to read anything in our shader except the dimensions of our texture, which we can
//! easily get via `textureDimensions`.
//!
//! A lot of things aren't explained here via comments. See hello-compute and
//! repeated-compute for code that is more thoroughly commented.

#[cfg(not(target_arch = "wasm32"))]
use crate::utils::output_image_native;
#[cfg(target_arch = "wasm32")]
use crate::utils::output_image_wasm;

const TEXTURE_DIMS: (usize, usize) = (512, 512);

async fn run(_path: Option<String>) {
    let mut texture_data = vec![0u8; TEXTURE_DIMS.0 * TEXTURE_DIMS.1 * 4];

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

    let storage_texture = device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size: etalon::Extent3d {
            width: TEXTURE_DIMS.0 as u32,
            height: TEXTURE_DIMS.1 as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Rgba8Unorm,
        usage: etalon::TextureUsages::STORAGE_BINDING | etalon::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let storage_texture_view = storage_texture.create_view(&etalon::TextureViewDescriptor::default());
    let output_staging_buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: size_of_val(&texture_data[..]) as u64,
        usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
        label: None,
        entries: &[etalon::BindGroupLayoutEntry {
            binding: 0,
            visibility: etalon::ShaderStages::COMPUTE,
            ty: etalon::BindingType::StorageTexture {
                access: etalon::StorageTextureAccess::WriteOnly,
                format: etalon::TextureFormat::Rgba8Unorm,
                view_dimension: etalon::TextureViewDimension::D2,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[etalon::BindGroupEntry {
            binding: 0,
            resource: etalon::BindingResource::TextureView(&storage_texture_view),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    log::info!("etalon context set up.");
    //----------------------------------------

    let mut command_encoder =
        device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.set_pipeline(&pipeline);
        compute_pass.dispatch_workgroups(TEXTURE_DIMS.0 as u32, TEXTURE_DIMS.1 as u32, 1);
    }
    command_encoder.copy_texture_to_buffer(
        etalon::TexelCopyTextureInfo {
            texture: &storage_texture,
            mip_level: 0,
            origin: etalon::Origin3d::ZERO,
            aspect: etalon::TextureAspect::All,
        },
        etalon::TexelCopyBufferInfo {
            buffer: &output_staging_buffer,
            layout: etalon::TexelCopyBufferLayout {
                offset: 0,
                // This needs to be padded to 256.
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

    let buffer_slice = output_staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(etalon::MapMode::Read, move |r| sender.send(r).unwrap());
    device.poll(etalon::PollType::wait_indefinitely()).unwrap();
    receiver.recv_async().await.unwrap().unwrap();
    log::info!("Output buffer mapped");
    {
        let view = buffer_slice.get_mapped_range().unwrap();
        texture_data.copy_from_slice(&view[..]);
    }
    log::info!("GPU data copied to local.");
    output_staging_buffer.unmap();

    #[cfg(not(target_arch = "wasm32"))]
    output_image_native(texture_data.to_vec(), TEXTURE_DIMS, _path.unwrap());
    #[cfg(target_arch = "wasm32")]
    output_image_wasm(texture_data.to_vec(), TEXTURE_DIMS);
    log::info!("Done.")
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
