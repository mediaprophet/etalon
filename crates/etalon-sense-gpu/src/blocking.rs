use super::*;

impl EtalonIqPowerProvider<'_> {
    /// Execute synchronously, preserving detailed Etalon failures.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn compute_into(
        &mut self,
        bytes: &[u8],
        byte_order: ByteOrder,
        scale: f32,
        workspace: &mut [f32],
        staged_output: &mut [f64],
    ) -> Result<usize, GpuError> {
        self.last_backend_error = None;
        if self.active.load(Ordering::Acquire) {
            return Err(GpuError::Busy);
        }
        if self.poisoned {
            return Err(GpuError::ProviderPoisoned);
        }
        if self.is_device_lost() {
            self.poisoned = true;
            return Err(GpuError::DeviceLost);
        }
        let count = validate_request(
            bytes,
            scale,
            workspace.len(),
            staged_output.len(),
            self.max_pairs,
        )?;
        let upper_bound = i16_iq_power_upper_bound(scale).map_err(|error| match error {
            SignalError::InvalidParameter => GpuError::InvalidScale,
            _ => GpuError::NumericRange,
        })?;

        self.queue.write_buffer(&self.input, 0, bytes);
        let params = make_params(scale, byte_order, count);
        self.queue.write_buffer(&self.params, 0, &params);

        let mut encoder = self
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor {
                label: Some("etalon-sense-iq-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
                label: Some("etalon-sense-iq-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(count.div_ceil(WORKGROUP_SIZE) as u32, 1, 1);
        }
        let output_bytes = checked_bytes(count, BYTES_PER_OUTPUT_PAIR)? as u64;
        encoder.copy_buffer_to_buffer(&self.power_values, 0, &self.readback, 0, output_bytes);
        let command_buffer = encoder.finish();
        let generation = self.completion.begin()?;
        let submission = match self.queue.submit([command_buffer]) {
            Some(submission) => submission,
            None => {
                self.completion.invalidate(generation);
                self.poisoned = true;
                return Err(GpuError::PollFailed);
            }
        };

        let completion = Arc::clone(&self.completion);
        self.readback
            .slice(0..output_bytes)
            .map_async(etalon::MapMode::Read, move |result| {
                completion.complete(generation, result.is_ok());
            });
        let poll = etalon::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(self.poll_timeout),
        };
        if let Err(error) = self.device.poll(poll) {
            self.completion.invalidate(generation);
            self.readback.unmap();
            self.poisoned = true;
            return Err(match error {
                etalon::PollError::Timeout => GpuError::PollTimedOut,
                etalon::PollError::CantWait => GpuError::SynchronousWaitUnsupported,
                _ => GpuError::PollFailed,
            });
        }
        match self.completion.result(generation) {
            Some(true) => {}
            Some(false) => {
                self.completion.invalidate(generation);
                self.readback.unmap();
                self.poisoned = true;
                return Err(GpuError::MapFailed);
            }
            None => {
                self.completion.invalidate(generation);
                self.readback.unmap();
                self.poisoned = true;
                return Err(GpuError::MapCallbackLost);
            }
        }

        let slice = self.readback.slice(0..output_bytes);
        let view = match slice.get_mapped_range() {
            Ok(view) => view,
            Err(_) => {
                self.readback.unmap();
                self.poisoned = true;
                return Err(GpuError::MappedRangeFailed);
            }
        };
        for (index, slot) in workspace.iter_mut().enumerate().take(count) {
            let start = index * 4;
            *slot = f32::from_le_bytes([
                view[start],
                view[start + 1],
                view[start + 2],
                view[start + 3],
            ]);
        }
        drop(view);
        self.readback.unmap();

        if self.is_device_lost() {
            workspace.fill(0.0);
            self.poisoned = true;
            return Err(GpuError::DeviceLost);
        }
        stage_power_values(&workspace[..count], staged_output, upper_bound)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl RegisteredSignalProvider for EtalonIqPowerProvider<'_> {
    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Persistent provider-owned GPU buffers were admitted against the
    /// constructor's explicit byte budget. This does not claim zero heap:
    /// Etalon command recording/submission and map callbacks still allocate.
    fn execution_allocation_class(&self) -> ExecutionAllocationClass {
        ExecutionAllocationClass::ColdBoundedBuffers
    }

    fn i16_iq_power_into(
        &mut self,
        bytes: &[u8],
        byte_order: ByteOrder,
        scale: f32,
        workspace: &mut [f32],
        staged_output: &mut [f64],
    ) -> Result<usize, SignalError> {
        match self.compute_into(bytes, byte_order, scale, workspace, staged_output) {
            Ok(count) => Ok(count),
            Err(error) => {
                self.last_backend_error = Some(error);
                Err(match error {
                    GpuError::InvalidScale => SignalError::InvalidParameter,
                    GpuError::CapacityExceeded
                    | GpuError::InvalidCapacity
                    | GpuError::LengthMismatch => SignalError::LengthMismatch,
                    GpuError::EmptyInput => SignalError::EmptyInput,
                    GpuError::InvalidInput => SignalError::InvalidParameter,
                    GpuError::NumericRange => SignalError::NumericRange,
                    _ => SignalError::InvalidParameter,
                })
            }
        }
    }
}
