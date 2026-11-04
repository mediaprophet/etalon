#[derive(Clone, Copy, Debug)]
pub struct ResourceBudget {
    pub max_buffer_bytes: u64,
    pub max_single_buffer_bytes: u64,
    pub max_shader_source_bytes: u64,
    pub max_compute_pipelines: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ResourcePlan {
    /// New explicit buffers; hosts account for retained resources and driver-owned pipeline memory.
    pub buffer_bytes: u64,
    /// Conservative upper bound including generated shader source scratch.
    pub shader_source_bytes: u64,
    pub compute_pipelines: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceError {
    SizeOverflow,
    BufferBudget,
    DeviceBufferLimit,
    ShaderBudget,
    PipelineBudget,
    UnsupportedSubgroups,
    PolicyDenied,
}

impl ResourceBudget {
    /// Device limits alone impose no aggregate allocation policy. Hosts may tighten each cap.
    pub fn for_device(device: &wgpu::Device) -> Self {
        Self {
            max_buffer_bytes: u64::MAX,
            max_single_buffer_bytes: device.limits().max_buffer_size,
            max_shader_source_bytes: u64::MAX,
            max_compute_pipelines: u32::MAX,
        }
    }

    pub fn admit(self, plan: ResourcePlan) -> Result<(), ResourceError> {
        if plan.buffer_bytes > self.max_buffer_bytes {
            return Err(ResourceError::BufferBudget);
        }
        if plan.shader_source_bytes > self.max_shader_source_bytes {
            return Err(ResourceError::ShaderBudget);
        }
        if plan.compute_pipelines > self.max_compute_pipelines {
            return Err(ResourceError::PipelineBudget);
        }
        Ok(())
    }

    pub(super) fn check_buffer(
        self,
        device: &wgpu::Device,
        bytes: u64,
        _usage: wgpu::BufferUsages,
    ) -> Result<(), ResourceError> {
        let limits = device.limits();
        if bytes > self.max_single_buffer_bytes {
            return Err(ResourceError::BufferBudget);
        }
        if bytes > limits.max_buffer_size {
            return Err(ResourceError::DeviceBufferLimit);
        }
        Ok(())
    }
}
