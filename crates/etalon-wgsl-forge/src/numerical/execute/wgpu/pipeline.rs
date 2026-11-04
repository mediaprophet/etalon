use super::*;
use std::time::Instant;

impl<'a> WgpuPipeline<'a> {
    /// Compiles a compute pipeline from a WGSL module.
    pub fn compile(
        context: &'a WgpuComputeContext,
        source: &str,
        entry_point: &str,
    ) -> Result<Self, ForgeError> {
        let pipeline = context.compile_pipeline(source, entry_point)?;
        Ok(Self { context, pipeline })
    }

    /// Like `compile()` but accepts pre-compiled SPIR-V bytes (from HLSL→DXC).
    pub fn compile_spirv(
        context: &'a WgpuComputeContext,
        spirv: &[u8],
        entry_point: &str,
    ) -> Result<Self, ForgeError> {
        let pipeline = context.compile_pipeline_spirv(spirv, entry_point)?;
        Ok(Self { context, pipeline })
    }

    /// Compile pre-emitted MSL source through wgpu's Metal backend.
    ///
    /// wgpu does not expose `ShaderSource::Msl` — on macOS, wgpu transpiles
    /// WGSL→MSL internally via naga. For pre-emitted MSL, the caller should
    /// use a native Metal path (e.g. `metal-rs`). This method exists so the
    /// runtime can attempt MSL compilation on macOS; on other platforms it
    /// returns an error and the runtime falls back to WGSL.
    pub fn compile_msl(
        context: &'a WgpuComputeContext,
        _source: &str,
        _entry_point: &str,
    ) -> Result<Self, ForgeError> {
        // wgpu's ShaderSource enum does not have an Msl variant.
        // On macOS, the practical path is to transpile MSL→SPIR-V via
        // SPIRV-Cross and use compile_spirv, or use metal-rs directly.
        // For now, return an error so the runtime falls back to WGSL.
        let _ = context;
        Err(ForgeError::Emission(
            "MSL native compilation requires a metal-rs bridge (not yet implemented). \
             Falling back to WGSL."
                .to_string(),
        ))
    }

    /// Dispatch a ray-query kernel, binding `tlas` as the `acceleration_structure`
    /// at binding 0 and the supplied buffer views (rays at binding 1, hits at
    /// binding 2) at their own bindings. The generic [`QualiaCompute::dispatch`]
    /// only binds buffers, so this is the dedicated path for the acceleration-
    /// structure binding. Returns wall-clock nanoseconds (ray-query passes skip the
    /// timestamp path). The caller must keep the TLAS (and its BLAS) alive.
    pub fn dispatch_rayprobe(
        &self,
        tlas: &wgpu::Tlas,
        buffers: &[BufferView],
        schedule: &Schedule,
        element_count: usize,
    ) -> Result<u64, ForgeError> {
        self.context.ensure_live()?;
        let mut entries = Vec::with_capacity(buffers.len() + 1);
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: tlas.as_binding(),
        });
        for view in buffers {
            let size = wgpu::BufferSize::new(view.length_bytes as u64);
            entries.push(wgpu::BindGroupEntry {
                binding: view.binding,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: self.context.slab_for(view.usage),
                    offset: view.offset as wgpu::BufferAddress,
                    size,
                }),
            });
        }
        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("forge-rayprobe-bind-group"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
        let dispatch_x = schedule.dispatch_workgroups(element_count);

        let started = Instant::now();
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("forge-rayprobe-dispatch"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forge-rayprobe-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }
        self.context.submit_commands(encoder.finish())?;
        self.context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| {
                ForgeError::DeviceLost(format!("device poll failed after ray probe: {error:?}"))
            })?;
        self.context.ensure_live()?;
        Ok(started.elapsed().as_nanos().min(u64::MAX as u128) as u64)
    }

    /// Dispatch an attention kernel, binding buffer views according to the Forge attention layout
    /// and dispatching one workgroup per attention head.
    pub fn dispatch_attention(
        &self,
        buffers: &[BufferView],
        head_count: u32,
    ) -> Result<u64, ForgeError> {
        self.context.ensure_live()?;
        let bind_group = self
            .context
            .create_compute_bind_group(&self.pipeline, buffers);
        let started = Instant::now();
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("forge-attention-dispatch"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forge-attention-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(head_count, 1, 1);
        }
        self.context.submit_commands(encoder.finish())?;
        self.context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| {
                ForgeError::DeviceLost(format!("device poll failed after attention: {error:?}"))
            })?;
        self.context.ensure_live()?;
        Ok(started.elapsed().as_nanos().min(u64::MAX as u128) as u64)
    }
}

impl<'a> QualiaCompute for WgpuPipeline<'a> {
    fn dispatch(
        &self,
        buffers: &[BufferView],
        schedule: &Schedule,
        element_count: usize,
    ) -> Result<u64, ForgeError> {
        self.context.ensure_live()?;
        let bind_group = self
            .context
            .create_compute_bind_group(&self.pipeline, buffers);
        let dispatch_x = schedule.dispatch_workgroups(element_count);

        let started = Instant::now();
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("forge-dispatch"),
                });
        {
            let timestamp_writes = self
                .context
                .timestamp_resources
                .as_ref()
                .map(|resources| resources.pass_writes(true, true));
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forge-compute-pass"),
                timestamp_writes,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        if let Some(resources) = &self.context.timestamp_resources {
            resources.resolve(&mut encoder);
        }
        self.context.submit_commands(encoder.finish())?;

        if let Some(resources) = &self.context.timestamp_resources {
            let ticks = resources.read_ticks(&self.context.device)?;
            let elapsed = ticks[1].saturating_sub(ticks[0]);
            self.context.ensure_live()?;
            Ok((elapsed as f64 * self.context.timestamp_period_ns as f64) as u64)
        } else {
            self.context
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| ForgeError::DeviceLost(format!("device poll failed: {e:?}")))?;
            self.context.ensure_live()?;
            Ok(started.elapsed().as_nanos().min(u64::MAX as u128) as u64)
        }
    }
}
