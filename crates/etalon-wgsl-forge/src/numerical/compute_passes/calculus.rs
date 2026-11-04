//! Borrowed-sample Simpson GPU reductions; file IO belongs to the caller.
use wgpu::util::DeviceExt;
pub const CALCULUS_WGSL: &str = include_str!("calculus.wgsl");

#[derive(Debug)]
pub enum GpuError {
    DirectStorageUnavailable(String),
    WebGPUUnavailable(String),
    ShaderCompilationFailed(String),
    BufferAllocationFailed(String),
    DispatchFailed(String),
    ReadbackFailed(String),
    InvalidOffset { offset: u64, required: u64 },
}
pub struct WebGpuIntegrator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    simpson_38_pipeline: wgpu::ComputePipeline,
}

impl WebGpuIntegrator {
    /// Creates a new WebGPU integrator.
    ///
    /// Initializes wgpu device, queue, and compiles the calculus compute shader.
    pub async fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                // wgpu 30 added `apply_limit_buckets`; take the crate default.
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError::WebGPUUnavailable(format!("No adapter found: {e}")))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| GpuError::WebGPUUnavailable(format!("Device request failed: {e}")))?;

        // Load pre-compiled compute shader (AOT compilation)
        let shader_src = CALCULUS_WGSL;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Calculus Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Calculus Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("simpsons_integration"),
            compilation_options: Default::default(),
            cache: None,
        });

        let simpson_38_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Simpson 3/8 Pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("simpson_38_integration"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            device,
            queue,
            compute_pipeline,
            simpson_38_pipeline,
        })
    }

    /// Executes a compute shader with the given input data.
    pub async fn execute_compute(
        &self,
        input_data: &[u8],
        step_size: f32,
    ) -> Result<f64, GpuError> {
        if input_data.len() % core::mem::size_of::<f32>() != 0 {
            return Err(GpuError::DispatchFailed(
                "calculus input must contain whole f32 values".to_string(),
            ));
        }
        let element_count = input_data.len() / core::mem::size_of::<f32>();
        if element_count < 3 || element_count % 2 == 0 {
            return Err(GpuError::DispatchFailed(
                "Simpson 1/3 requires an odd sample count of at least three".to_string(),
            ));
        }
        if !step_size.is_finite() || step_size == 0.0 {
            return Err(GpuError::DispatchFailed(
                "step size must be finite and non-zero".to_string(),
            ));
        }

        // Create storage buffer for input data
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input Buffer"),
                contents: input_data,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create workgroup reduction buffer (one f32 per workgroup)
        // For 5000 elements with 64-thread workgroups: ceil(5000/64) = 79 workgroups
        let num_workgroups = (element_count + 63) / 64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Workgroup Reduction Buffer"),
            size: (num_workgroups * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for step size and total element count
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
        struct Uniforms {
            step_size: f32,
            total_elements: u32,
        }

        let uniforms = Uniforms {
            step_size,
            total_elements: element_count as u32,
        };

        let step_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniforms Buffer"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group_layout = self.compute_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Calculus Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: step_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Calculus Encoder"),
            });

        // Dispatch compute shader
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Calculus Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        // Copy output to staging buffer for readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (num_workgroups * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (num_workgroups * 4) as u64,
        );

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Read back result
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.queue.submit(None);

        let _mapping_result = receiver.await.unwrap();

        // Get mapped slice
        let result_data = buffer_slice
            .get_mapped_range()
            .expect("wgpu buffer map_range failed");

        // Read workgroup results and sum using Kahan summation for precision
        let workgroup_results: &[f32] = bytemuck::cast_slice(&*result_data);
        let mut sum = 0.0f64;
        let mut compensation = 0.0f64;

        for &partial in workgroup_results {
            let y = partial as f64 - compensation;
            let t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }

        Ok(sum)
    }

    /// Composite Simpson 3/8 variant of the reduction pipeline.
    pub async fn execute_simpson_38_compute(
        &self,
        input_data: &[u8],
        step_size: f32,
    ) -> Result<f64, GpuError> {
        if input_data.len() % core::mem::size_of::<f32>() != 0 {
            return Err(GpuError::DispatchFailed(
                "calculus input must contain whole f32 values".to_string(),
            ));
        }
        let element_count = input_data.len() / core::mem::size_of::<f32>();
        if element_count < 4 || (element_count - 1) % 3 != 0 {
            return Err(GpuError::DispatchFailed(
                "Simpson 3/8 requires a positive panel count divisible by three".to_string(),
            ));
        }
        if !step_size.is_finite() || step_size == 0.0 {
            return Err(GpuError::DispatchFailed(
                "step size must be finite and non-zero".to_string(),
            ));
        }

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Simpson 3/8 Input Buffer"),
                contents: input_data,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let num_workgroups = (element_count + 63) / 64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Simpson 3/8 Reduction Buffer"),
            size: (num_workgroups * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
        struct Uniforms {
            step_size: f32,
            total_elements: u32,
        }

        let uniforms = Uniforms {
            step_size,
            total_elements: element_count as u32,
        };
        let step_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Simpson 3/8 Uniforms"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bgl = self.simpson_38_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Simpson 3/8 Bind Group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: step_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Simpson 3/8 Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Simpson 3/8 Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.simpson_38_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Simpson 3/8 Staging"),
            size: (num_workgroups * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, (num_workgroups * 4) as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.queue.submit(None);
        let _ = rx.await.unwrap();

        let mapped = slice
            .get_mapped_range()
            .expect("wgpu buffer map_range failed");
        let partials: &[f32] = bytemuck::cast_slice(&*mapped);

        // Kahan summation for numerical stability
        let mut sum = 0.0f64;
        let mut comp = 0.0f64;
        for &p in partials {
            let y = p as f64 - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        Ok(sum)
    }
}

impl WebGpuIntegrator {
    pub fn max_buffer_size(&self) -> u64 {
        self.device.limits().max_buffer_size
    }
}
