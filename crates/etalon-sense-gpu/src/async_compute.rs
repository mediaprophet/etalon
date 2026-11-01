use crate::{
    checked_bytes, i16_iq_power_upper_bound, make_params, stage_power_values, validate_request,
    ByteOrder, EtalonIqPowerProvider, GpuError, MapCompletion, SignalError, BYTES_PER_OUTPUT_PAIR,
    WORKGROUP_SIZE,
};
use core::{future::Future, task::Poll};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[cfg(target_arch = "wasm32")]
mod deadline;
#[cfg(test)]
mod tests;

struct RequestGuard<'a, F: FnMut()> {
    active: &'a AtomicBool,
    completion: &'a MapCompletion,
    generation: u64,
    cleanup: F,
}
impl<'a, F: FnMut()> RequestGuard<'a, F> {
    fn new(
        active: &'a AtomicBool,
        completion: &'a MapCompletion,
        cleanup: F,
    ) -> Result<Self, GpuError> {
        active
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .map_err(|_| GpuError::Busy)?;
        let generation = match completion.begin() {
            Ok(generation) => generation,
            Err(error) => {
                active.store(false, Ordering::Release);
                return Err(error);
            }
        };
        Ok(Self {
            active,
            completion,
            generation,
            cleanup,
        })
    }
}
impl<F: FnMut()> Drop for RequestGuard<'_, F> {
    fn drop(&mut self) {
        self.completion.invalidate(self.generation);
        (self.cleanup)();
        self.active.store(false, Ordering::Release);
    }
}

fn scope_error(error: etalon::Error) -> GpuError {
    match error {
        etalon::Error::Validation { .. } => GpuError::ValidationFailed,
        etalon::Error::OutOfMemory { .. } => GpuError::OutOfMemory,
        etalon::Error::Internal { .. } => GpuError::InternalError,
    }
}

impl EtalonIqPowerProvider<'_> {
    /// Run the portable WGSL kernel with caller-owned staging/output. No CPU fallback occurs.
    /// Cancellation invalidates the map generation, unmaps staging and releases the busy state.
    /// Native futures drive nonblocking polling; browsers use callbacks and a cancellable timer.
    /// Commands/map callbacks may allocate; the constructor budget covers persistent buffers only.
    pub async fn compute_into_async(
        &mut self,
        bytes: &[u8],
        byte_order: ByteOrder,
        scale: f32,
        workspace: &mut [f32],
        staged_output: &mut [f64],
    ) -> Result<usize, GpuError> {
        self.last_backend_error = None;
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
        let output_bytes = checked_bytes(count, BYTES_PER_OUTPUT_PAIR)? as u64;
        let guard = RequestGuard::new(&self.active, &self.completion, || self.readback.unmap())?;
        let generation = guard.generation;
        let validation = self
            .device
            .push_error_scope(etalon::ErrorFilter::Validation);
        let internal = self.device.push_error_scope(etalon::ErrorFilter::Internal);
        let memory = self
            .device
            .push_error_scope(etalon::ErrorFilter::OutOfMemory);
        self.queue.write_buffer(&self.input, 0, bytes);
        self.queue
            .write_buffer(&self.params, 0, &make_params(scale, byte_order, count));
        let mut encoder = self
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor {
                label: Some("etalon-sense-iq-async"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(count.div_ceil(WORKGROUP_SIZE) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.power_values, 0, &self.readback, 0, output_bytes);
        let submission = self.queue.submit([encoder.finish()]);
        #[cfg(not(target_arch = "wasm32"))]
        if submission.is_none() {
            self.poisoned = true;
            return Err(GpuError::PollFailed);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = submission; // Browser submissions have no native waitable index.
        let completion = Arc::clone(&self.completion);
        self.readback
            .slice(..output_bytes)
            .map_async(etalon::MapMode::Read, move |result| {
                completion.complete(generation, result.is_ok())
            });

        // Remove every scope before yielding; browser pop futures attach Promise rejection handlers.
        let mut memory = core::pin::pin!(memory.pop());
        let mut internal = core::pin::pin!(internal.pop());
        let mut validation = core::pin::pin!(validation.pop());
        #[cfg(target_arch = "wasm32")]
        let _deadline =
            deadline::Deadline::new(Arc::clone(&self.completion), generation, self.poll_timeout)?;
        #[cfg(not(target_arch = "wasm32"))]
        let started = std::time::Instant::now();
        let mut scopes_done = [false; 3];
        let result = core::future::poll_fn(|cx| {
            #[cfg(not(target_arch = "wasm32"))]
            {
                if started.elapsed() >= self.poll_timeout {
                    self.completion.fail(generation, GpuError::PollTimedOut);
                } else if self.device.poll(etalon::PollType::Poll).is_err() {
                    self.completion.fail(generation, GpuError::PollFailed);
                }
            }
            if self.is_device_lost() {
                self.completion.fail(generation, GpuError::DeviceLost);
            }
            for (done, future) in scopes_done.iter_mut().zip([
                memory.as_mut(),
                internal.as_mut(),
                validation.as_mut(),
            ]) {
                if !*done {
                    if let Poll::Ready(error) = future.poll(cx) {
                        *done = true;
                        if let Some(error) = error {
                            self.completion.fail(generation, scope_error(error));
                        }
                    }
                }
            }
            match self.completion.status(generation, cx) {
                Some(Err(error)) => Poll::Ready(Err(error)),
                Some(Ok(())) if scopes_done.iter().all(|done| *done) => Poll::Ready(Ok(())),
                _ => {
                    #[cfg(not(target_arch = "wasm32"))]
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
        })
        .await;
        if let Err(error) = result {
            self.poisoned = true;
            return Err(error);
        }
        let view = match self.readback.slice(..output_bytes).get_mapped_range() {
            Ok(view) => view,
            Err(_) => {
                self.poisoned = true;
                return Err(GpuError::MappedRangeFailed);
            }
        };
        if view.len() != count * 4 {
            self.poisoned = true;
            return Err(GpuError::MappedRangeFailed);
        }
        for (slot, word) in workspace[..count].iter_mut().zip(view.chunks_exact(4)) {
            *slot = f32::from_le_bytes(word.try_into().unwrap());
        }
        drop(view);
        drop(guard);
        if self.is_device_lost() {
            workspace.fill(0.0);
            self.poisoned = true;
            return Err(GpuError::DeviceLost);
        }
        stage_power_values(&workspace[..count], staged_output, upper_bound)
    }
}
