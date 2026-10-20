//! This example assumes that you've seen hello-compute and or repeated-compute
//! and thus have a general understanding of what's going on here.
//!
//! There's an explainer on what this example does exactly and what workgroups
//! are and the meaning of `@workgroup(size_x, size_y, size_z)` in the
//! README. Also see commenting in shader.wgsl as well.
//!
//! Only parts specific to this example will be commented.

use etalon::util::DeviceExt;

async fn run() {
    let mut local_a = [0i32; 100];
    for (i, e) in local_a.iter_mut().enumerate() {
        *e = i as i32;
    }
    log::info!("Input a: {local_a:?}");
    let mut local_b = [0i32; 100];
    for (i, e) in local_b.iter_mut().enumerate() {
        *e = i as i32 * 2;
    }
    log::info!("Input b: {local_b:?}");

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

    let storage_buffer_a = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&local_a[..]),
        usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_SRC,
    });
    let storage_buffer_b = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&local_b[..]),
        usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_SRC,
    });
    let output_staging_buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: size_of_val(&local_a) as u64,
        usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            etalon::BindGroupLayoutEntry {
                binding: 0,
                visibility: etalon::ShaderStages::COMPUTE,
                ty: etalon::BindingType::Buffer {
                    ty: etalon::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            etalon::BindGroupLayoutEntry {
                binding: 1,
                visibility: etalon::ShaderStages::COMPUTE,
                ty: etalon::BindingType::Buffer {
                    ty: etalon::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            etalon::BindGroupEntry {
                binding: 0,
                resource: storage_buffer_a.as_entire_binding(),
            },
            etalon::BindGroupEntry {
                binding: 1,
                resource: storage_buffer_b.as_entire_binding(),
            },
        ],
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

    //----------------------------------------------------------

    let mut command_encoder =
        device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        /* Note that since each workgroup will cover both arrays, we only need to
        cover the length of one array. */
        compute_pass.dispatch_workgroups(local_a.len() as u32, 1, 1);
    }
    queue.submit(Some(command_encoder.finish()));

    //----------------------------------------------------------

    get_data(
        &mut local_a[..],
        &storage_buffer_a,
        &output_staging_buffer,
        &device,
        &queue,
    )
    .await;
    get_data(
        &mut local_b[..],
        &storage_buffer_b,
        &output_staging_buffer,
        &device,
        &queue,
    )
    .await;

    log::info!("Output in A: {local_a:?}");
    log::info!("Output in B: {local_b:?}");
}

async fn get_data<T: bytemuck::Pod>(
    output: &mut [T],
    storage_buffer: &etalon::Buffer,
    staging_buffer: &etalon::Buffer,
    device: &etalon::Device,
    queue: &etalon::Queue,
) {
    let mut command_encoder =
        device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
    command_encoder.copy_buffer_to_buffer(
        storage_buffer,
        0,
        staging_buffer,
        0,
        size_of_val(output) as u64,
    );
    queue.submit(Some(command_encoder.finish()));
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(etalon::MapMode::Read, move |r| sender.send(r).unwrap());
    device.poll(etalon::PollType::wait_indefinitely()).unwrap();
    receiver.recv_async().await.unwrap().unwrap();
    let data: Vec<T> =
        bytemuck::allocation::pod_collect_to_vec(&buffer_slice.get_mapped_range().unwrap()[..]);
    output.copy_from_slice(&data);
    staging_buffer.unmap();
}

pub fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .format_timestamp_nanos()
            .init();
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).expect("could not initialize logger");

        crate::utils::add_web_nothing_to_see_msg();

        wasm_bindgen_futures::spawn_local(run());
    }
}
