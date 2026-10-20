//! This example shows you a potential course for when your 'data' is too large
//! for a single Buffer.
//!
//! A lot of things aren't explained here via comments. See hello-compute and
//! repeated-compute for code that is more thoroughly commented.

use std::num::{NonZeroU32, NonZeroU64};
use etalon::{util::DeviceExt, Features};

// These are set by the minimum required defaults for webgpu.
const MAX_BUFFER_SIZE: u64 = 1 << 27; // 134_217_728 // 134MB
const MAX_DISPATCH_SIZE: u32 = (1 << 16) - 1;

pub async fn execute_gpu(numbers: &[f32]) -> Vec<f32> {
    let instance = etalon::Instance::default();

    let adapter = instance
        .request_adapter(&etalon::RequestAdapterOptions::default())
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(&etalon::DeviceDescriptor {
            label: None,
            // These features are required to use `binding_array` in your wgsl.
            // Without them your shader may fail to compile.
            required_features: Features::BUFFER_BINDING_ARRAY
                | Features::STORAGE_RESOURCE_BINDING_ARRAY
                | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            memory_hints: etalon::MemoryHints::Performance,
            required_limits: etalon::Limits {
                max_buffer_size: MAX_BUFFER_SIZE,
                max_binding_array_elements_per_shader_stage: 8,
                ..Default::default()
            },
            ..Default::default()
        })
        .await
        .unwrap();

    execute_gpu_inner(&device, &queue, numbers).await
}

pub async fn execute_gpu_inner(
    device: &etalon::Device,
    queue: &etalon::Queue,
    numbers: &[f32],
) -> Vec<f32> {
    let (staging_buffers, storage_buffers, bind_group, compute_pipeline) = setup(device, numbers);

    let mut encoder =
        device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
            label: Some("compute pass descriptor"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, Some(&bind_group), &[]);

        cpass.dispatch_workgroups(MAX_DISPATCH_SIZE.min(numbers.len() as u32), 1, 1);
    }

    for (storage_buffer, staging_buffer) in storage_buffers.iter().zip(staging_buffers.iter()) {
        let stg_size = staging_buffer.size();

        encoder.copy_buffer_to_buffer(
            storage_buffer, // Source buffer
            0,
            staging_buffer, // Destination buffer
            0,
            stg_size,
        );
    }

    queue.submit(Some(encoder.finish()));

    for staging_buffer in &staging_buffers {
        let slice = staging_buffer.slice(..);
        slice.map_async(etalon::MapMode::Read, |_| {});
    }

    device.poll(etalon::PollType::wait_indefinitely()).unwrap();

    let mut data = Vec::new();
    for staging_buffer in &staging_buffers {
        let slice = staging_buffer.slice(..);
        let mapped = slice.get_mapped_range().unwrap();
        let chunk: Vec<f32> = bytemuck::allocation::pod_collect_to_vec(&mapped);
        data.extend_from_slice(&chunk);
        drop(mapped);
        staging_buffer.unmap();
    }

    data
}

fn setup(
    device: &etalon::Device,
    numbers: &[f32],
) -> (
    Vec<etalon::Buffer>,
    Vec<etalon::Buffer>,
    etalon::BindGroup,
    etalon::ComputePipeline,
) {
    let cs_module = device.create_shader_module(etalon::include_wgsl!("shader.wgsl"));

    let staging_buffers = create_staging_buffers(device, numbers);
    let storage_buffers = create_storage_buffers(device, numbers);

    let (bind_group_layout, bind_group) = setup_binds(&storage_buffers, device);

    let compute_pipeline = setup_pipeline(device, bind_group_layout, cs_module);
    (
        staging_buffers,
        storage_buffers,
        bind_group,
        compute_pipeline,
    )
}

fn setup_pipeline(
    device: &etalon::Device,
    bind_group_layout: etalon::BindGroupLayout,
    cs_module: etalon::ShaderModule,
) -> etalon::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn setup_binds(
    storage_buffers: &[etalon::Buffer],
    device: &etalon::Device,
) -> (etalon::BindGroupLayout, etalon::BindGroup) {
    let buffers: Vec<_> = storage_buffers
        .iter()
        .map(|b| b.as_entire_buffer_binding())
        .collect();

    let entry = etalon::BindGroupEntry {
        binding: 0,
        resource: etalon::BindingResource::BufferArray(&buffers),
    };

    let bgl_entry = etalon::BindGroupLayoutEntry {
        binding: 0,
        visibility: etalon::ShaderStages::COMPUTE,
        ty: etalon::BindingType::Buffer {
            ty: etalon::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: Some(NonZeroU64::new(4).unwrap()),
        },
        count: Some(NonZeroU32::new(buffers.len() as u32).unwrap()),
    };

    let bind_group_layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
        label: Some("Custom Storage Bind Group Layout"),
        entries: &[bgl_entry],
    });

    let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
        label: Some("Combined Storage Bind Group"),
        layout: &bind_group_layout,
        entries: &[entry],
    });

    (bind_group_layout, bind_group)
}

fn calculate_chunks(numbers: &[f32], max_buffer_size: u64) -> Vec<&[f32]> {
    let max_elements_per_chunk = max_buffer_size as usize / std::mem::size_of::<f32>();
    numbers.chunks(max_elements_per_chunk).collect()
}

fn create_storage_buffers(device: &etalon::Device, numbers: &[f32]) -> Vec<etalon::Buffer> {
    let chunks = calculate_chunks(numbers, MAX_BUFFER_SIZE);

    chunks
        .iter()
        .enumerate()
        .map(|(e, seg)| {
            device.create_buffer_init(&etalon::util::BufferInitDescriptor {
                label: Some(&format!("Storage Buffer-{e}")),
                contents: bytemuck::cast_slice(seg),
                usage: etalon::BufferUsages::STORAGE
                    | etalon::BufferUsages::COPY_DST
                    | etalon::BufferUsages::COPY_SRC,
            })
        })
        .collect()
}

fn create_staging_buffers(device: &etalon::Device, numbers: &[f32]) -> Vec<etalon::Buffer> {
    let chunks = calculate_chunks(numbers, MAX_BUFFER_SIZE);

    (0..chunks.len())
        .map(|e| {
            let size = std::mem::size_of_val(chunks[e]) as u64;

            device.create_buffer(&etalon::BufferDescriptor {
                label: Some(&format!("staging buffer-{e}")),
                size,
                usage: etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .collect()
}

#[cfg_attr(target_arch = "wasm32", allow(clippy::allow_attributes, dead_code))]
async fn run() {
    let numbers = {
        const BYTES_PER_GB: usize = 1024 * 1024 * 1024;
        // 4 bytes per f32
        let elements = (BYTES_PER_GB as f32 / 4.0) as usize;
        vec![0.0; elements]
    };
    assert!(numbers.iter().all(|n| *n == 0.0));
    log::info!("All 0.0s");
    let t1 = std::time::Instant::now();
    let results = execute_gpu(&numbers).await;
    log::info!("GPU RUNTIME: {}ms", t1.elapsed().as_millis());
    assert_eq!(numbers.len(), results.len());
    assert!(results.iter().all(|n| *n == 1.0));
    log::info!("All 1.0s");
}

pub fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run());
    }
}

#[cfg(test)]
#[cfg(not(wasm_test))]
pub mod tests;
