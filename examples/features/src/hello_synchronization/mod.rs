const ARR_SIZE: usize = 128;

struct ExecuteResults {
    patient_workgroup_results: Vec<u32>,
    hasty_workgroup_results: Vec<u32>,
}

async fn run() {
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
            memory_hints: etalon::MemoryHints::Performance,
            trace: etalon::Trace::Off,
        })
        .await
        .unwrap();

    let ExecuteResults {
        patient_workgroup_results,
        hasty_workgroup_results,
    } = execute(&device, &queue, ARR_SIZE).await;

    // Print data
    log::info!("Patient results: {patient_workgroup_results:?}");
    if !patient_workgroup_results.iter().any(|e| *e != 16) {
        log::info!("patient_main was patient.");
    } else {
        log::error!("patient_main was not patient!");
    }
    log::info!("Hasty results: {hasty_workgroup_results:?}");
    if hasty_workgroup_results.iter().any(|e| *e != 16) {
        log::info!("hasty_main was not patient.");
    } else {
        log::info!("hasty_main got lucky.");
    }
}

async fn execute(
    device: &etalon::Device,
    queue: &etalon::Queue,
    result_vec_size: usize,
) -> ExecuteResults {
    let mut local_patient_workgroup_results = vec![0u32; result_vec_size];
    let mut local_hasty_workgroup_results = local_patient_workgroup_results.clone();

    let shaders_module = device.create_shader_module(etalon::include_wgsl!("shaders.wgsl"));

    let storage_buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: size_of_val(local_patient_workgroup_results.as_slice()) as u64,
        usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_staging_buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: size_of_val(local_patient_workgroup_results.as_slice()) as u64,
        usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
        label: None,
        entries: &[etalon::BindGroupLayoutEntry {
            binding: 0,
            visibility: etalon::ShaderStages::COMPUTE,
            ty: etalon::BindingType::Buffer {
                ty: etalon::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[etalon::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let patient_pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shaders_module,
        entry_point: Some("patient_main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let hasty_pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shaders_module,
        entry_point: Some("hasty_main"),
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
        compute_pass.set_pipeline(&patient_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(local_patient_workgroup_results.len() as u32, 1, 1);
    }
    queue.submit(Some(command_encoder.finish()));

    get_data(
        local_patient_workgroup_results.as_mut_slice(),
        &storage_buffer,
        &output_staging_buffer,
        device,
        queue,
    )
    .await;

    let mut command_encoder =
        device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&hasty_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(local_patient_workgroup_results.len() as u32, 1, 1);
    }
    queue.submit(Some(command_encoder.finish()));

    get_data(
        local_hasty_workgroup_results.as_mut_slice(),
        &storage_buffer,
        &output_staging_buffer,
        device,
        queue,
    )
    .await;

    ExecuteResults {
        patient_workgroup_results: local_patient_workgroup_results,
        hasty_workgroup_results: local_hasty_workgroup_results,
    }
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

#[cfg(test)]
pub mod tests;
