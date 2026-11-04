//! Caller-shader bitmask filter dispatch over borrowed record bytes.
use wgpu::util::DeviceExt;

/// 64-bit filter mask split into two u32 words — matches the lo/hi layout in sieve.wgsl.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FilterMask64 {
    pub lo: u32,
    pub hi: u32,
}

/// ceil(850 / 32) = 27 u32 words cover all 850 records per block (108 bytes total).
const BITMASK_WORDS: usize = 27;

pub struct BitmaskFilter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
}

impl BitmaskFilter {
    /// Creates `?` with the given state.
    pub async fn new(shader_src: &str) -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sieve Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sieve Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            device,
            queue,
            compute_pipeline,
        })
    }

    /// Dispatches the GPU sieve over `records`, returns matching record indices decoded from
    /// a 27-u32 bitmask (108 bytes readback vs. N×4 bytes for the old per-flag approach).
    /// The 64-bit `mask` is split lo/hi so the shader needs no shader-int64 capability.
    pub async fn execute_sieve(
        &self,
        records: &[u8],
        record_count: usize,
        mask: u64,
        on_submit: impl FnOnce(),
    ) -> Option<Vec<u32>> {
        if record_count == 0 {
            return Some(Vec::new());
        }
        if record_count > 850 || records.is_empty() {
            return None;
        }
        let filter = FilterMask64 {
            lo: mask as u32,
            hi: (mask >> 32) as u32,
        };

        // 1. Storage buffer: flat record array (12 × u32 per record = 48 bytes, bytemuck-safe)
        let record_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("record Buffer"),
                contents: records,
                usage: wgpu::BufferUsages::STORAGE,
            });

        // 2. Output bitmask: 27 u32s = 108 bytes (covers 850 records, zero-initialised by spec)
        let result_size = (BITMASK_WORDS * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bitmask Buffer"),
            size: result_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 3. Uniform: 64-bit mask as lo/hi u32 pair
        let filter_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Filter Buffer"),
                contents: bytemuck::cast_slice(&[filter]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout = self.compute_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sieve Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: record_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: result_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: filter_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            // Dispatch logic: 64 threads per workgroup
            let workgroups = ((record_count as f32) / 64.0).ceil() as u32;
            cpass.dispatch_workgroups(if workgroups == 0 { 1 } else { workgroups }, 1, 1);
        }

        // Copy results back to a staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: result_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, result_size);

        self.queue.submit(Some(encoder.finish()));

        on_submit();

        // Await completion
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        if matches!(receiver.recv(), Ok(Ok(()))) {
            let data = buffer_slice
                .get_mapped_range()
                .expect("wgpu buffer map_range failed");
            let bitmask: &[u32] = bytemuck::cast_slice(&data);

            // Trailing-zero bit-scan: extract each set bit as a record index
            let mut matching_indices: Vec<u32> = Vec::with_capacity(128);
            for (bucket_idx, &bucket_val) in bitmask.iter().enumerate() {
                let mut val = bucket_val;
                while val != 0 {
                    let bit_shift = val.trailing_zeros();
                    matching_indices.push(bucket_idx as u32 * 32 + bit_shift);
                    val &= val - 1; // clear lowest set bit
                }
            }

            drop(data);
            staging_buffer.unmap();
            return Some(matching_indices);
        }

        None
    }
}
