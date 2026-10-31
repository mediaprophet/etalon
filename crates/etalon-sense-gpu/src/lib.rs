//! Explicit, caller-selected native GPU execution for bounded sensing kernels.
//!
//! This crate accepts an existing Etalon device and queue. It does not discover
//! devices, authorize capture, attest a provider, qualify hardware, or fall back
//! to CPU execution. Readback is blocking with a bounded poll timeout and is
//! available only on native Etalon backends; WebGPU cannot wait synchronously.
//! This provider is explicitly `ColdBoundedBuffers`, not a Tier-1 zero-heap hot path.
//! Etalon's `BufferSlice::map_async` boxes its callback, and its command encoder,
//! command buffer, and queue submission APIs do not provide a caller-owned
//! preallocated recording/submission arena. Strict `HotZeroHeap` admission is
//! therefore rejected before resources are created. Persistent GPU buffer
//! capacities are admitted against a caller byte budget; backend/pipeline and
//! transient command allocations are outside that buffer-only budget. Buffers
//! retain samples and powers until overwritten or destroyed; secure erasure is
//! not guaranteed.
#![forbid(unsafe_code)]
#![cfg(not(target_arch = "wasm32"))]

mod allocation;
pub use allocation::GpuAllocationClass;
pub mod live;
mod numeric;
mod pair;
use numeric::{
    checked_bytes, make_params, persistent_buffer_bytes, stage_power_values, validate_power_values,
    validate_request,
};
pub use pair::{DeviceQueuePair, DeviceRequestGate, DeviceRequestStage, PairRequestError};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use etalon::{BindGroup, Buffer, ComputePipeline, Device, Queue};
use etalon_signal::{
    decode::ByteOrder,
    execution::{
        i16_iq_power_upper_bound, DeviceId, ExecutionAllocationClass, RegisteredSignalProvider,
    },
    SignalError,
};

pub const MAX_IQ_PAIRS: usize = 1_048_576;
pub const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_secs(5);
pub const MAX_POLL_TIMEOUT: Duration = Duration::from_secs(60);
const WORKGROUP_SIZE: usize = 64;
const PARAM_BYTES: u64 = 16;
const BYTES_PER_INPUT_PAIR: usize = 4;
const BYTES_PER_OUTPUT_PAIR: usize = 4;

/// Failures from shape validation or Etalon command/readback execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuError {
    InvalidCapacity,
    CapacityExceeded,
    DeviceLimitExceeded,
    EmptyInput,
    LengthMismatch,
    InvalidInput,
    InvalidScale,
    InvalidTimeout,
    NumericRange,
    PollFailed,
    PollTimedOut,
    SynchronousWaitUnsupported,
    MapFailed,
    MapCallbackLost,
    MappedRangeFailed,
    ProviderPoisoned,
    HotZeroHeapUnsupported,
    BufferBudgetExceeded,
    DeviceLost,
}

/// Caller-selected Etalon device provider for signed-i16 I/Q power.
///
/// Each call uploads only the provided bytes, runs WGSL to decode the requested
/// byte order, apply `scale`, and compute I² + Q² in f32. Results are converted
/// to f64 on readback; precision is therefore f32 and can differ from the
/// reference CPU pipeline's f64 sum. This is blocking, bounded by `max_pairs`,
/// and native-only. An Etalon device/ID match does not constitute GPU attestation
/// or hardware qualification. Its host path may allocate transient command
/// resources, and Etalon boxes the map callback, so it does not satisfy a strict
/// hot zero-heap contract. The completion cell is allocated once per provider.
/// The persistent input, power, and readback buffers retain sample-derived data
/// until overwritten or destroyed. This API makes no secure-erasure guarantee;
/// the host must control provider reuse, lifetime, and trust-domain separation.
pub struct EtalonIqPowerProvider<'a> {
    device: &'a Device,
    queue: &'a Queue,
    device_id: DeviceId,
    allocation_class: GpuAllocationClass,
    persistent_buffer_bytes: u64,
    max_pairs: usize,
    poll_timeout: Duration,
    input: Buffer,
    power_values: Buffer,
    readback: Buffer,
    params: Buffer,
    bind_group: BindGroup,
    pipeline: ComputePipeline,
    completion: Arc<MapCompletion>,
    poisoned: bool,
    last_backend_error: Option<GpuError>,
    loss_state: Option<&'a AtomicBool>,
}

/// Reusable provider-owned map completion state. Etalon requires a `'static`
/// callback, so each callback holds an `Arc` clone. Generation checks make a
/// late callback after cancellation unable to complete a later request.
struct MapCompletion {
    state: Mutex<MapCompletionState>,
}

struct MapCompletionState {
    generation: u64,
    result: Option<bool>,
    retired: bool,
}

impl MapCompletion {
    fn new() -> Self {
        Self {
            state: Mutex::new(MapCompletionState {
                generation: 0,
                result: None,
                retired: false,
            }),
        }
    }

    fn begin(&self) -> Result<u64, GpuError> {
        let mut state = self.lock();
        if state.retired {
            return Err(GpuError::ProviderPoisoned);
        }
        let Some(generation) = state.generation.checked_add(1) else {
            state.retired = true;
            return Err(GpuError::ProviderPoisoned);
        };
        state.generation = generation;
        state.result = None;
        Ok(generation)
    }

    fn complete(&self, generation: u64, mapped: bool) {
        let mut state = self.lock();
        if !state.retired && state.generation == generation {
            state.result = Some(mapped);
        }
    }

    fn result(&self, generation: u64) -> Option<bool> {
        let state = self.lock();
        if !state.retired && state.generation == generation {
            state.result
        } else {
            None
        }
    }

    /// Invalidate callbacks before unmapping after a failed or timed-out wait.
    fn invalidate(&self, generation: u64) {
        let mut state = self.lock();
        if state.generation != generation {
            return;
        }
        state.result = None;
        if let Some(next) = state.generation.checked_add(1) {
            state.generation = next;
        } else {
            // No future request is safe at wraparound. Existing callbacks are
            // ignored even if their generation matches the terminal value.
            state.retired = true;
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, MapCompletionState> {
        // A poisoned lock still contains bounded state; recover it without
        // panicking in the completion callback.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl<'a> EtalonIqPowerProvider<'a> {
    /// Build a `ColdBoundedBuffers` provider using its maximum fixed-buffer
    /// size as the budget. Prefer `new_with_allocation_class` when the host has
    /// an explicit byte budget or requires strict zero-heap admission.
    /// Resource sizes are capped and checked against declared device limits.
    ///
    /// The caller must pass a queue created with this device. Etalon exposes no
    /// portable safe API to inspect a queue's owning device; a mismatched pair
    /// may surface as an Etalon validation error outside this result type.
    pub fn new(
        device: &'a Device,
        queue: &'a Queue,
        device_id: DeviceId,
        max_pairs: usize,
    ) -> Result<Self, GpuError> {
        Self::new_with_timeout(device, queue, device_id, max_pairs, DEFAULT_POLL_TIMEOUT)
    }

    /// Build fixed-capacity resources with a bounded native readback wait.
    /// Timeout cancels the pending readback map where possible and poisons this
    /// provider; callers must discard it and arrange device cleanup. The timeout
    /// cannot cancel already submitted GPU work or synchronize external policy.
    pub fn new_with_timeout(
        device: &'a Device,
        queue: &'a Queue,
        device_id: DeviceId,
        max_pairs: usize,
        poll_timeout: Duration,
    ) -> Result<Self, GpuError> {
        if max_pairs == 0 || max_pairs > MAX_IQ_PAIRS {
            return Err(GpuError::InvalidCapacity);
        }
        let budget = persistent_buffer_bytes(max_pairs)?;
        Self::new_with_allocation_class(
            device,
            queue,
            device_id,
            max_pairs,
            GpuAllocationClass::ColdBoundedBuffers,
            budget,
            poll_timeout,
        )
    }

    /// Build resources only when the requested class is supported and
    /// persistent GPU buffers fit the caller's explicit byte budget.
    /// `HotZeroHeap` always fails closed because this Etalon path cannot meet a
    /// strict zero-heap contract. The budget counts the fixed input, power,
    /// readback, and params buffers; it does not bound driver, pipeline, or
    /// transient command memory.
    pub fn new_with_allocation_class(
        device: &'a Device,
        queue: &'a Queue,
        device_id: DeviceId,
        max_pairs: usize,
        allocation_class: GpuAllocationClass,
        gpu_buffer_budget_bytes: u64,
        poll_timeout: Duration,
    ) -> Result<Self, GpuError> {
        if max_pairs == 0 || max_pairs > MAX_IQ_PAIRS {
            return Err(GpuError::InvalidCapacity);
        }
        let persistent_buffer_bytes = persistent_buffer_bytes(max_pairs)?;
        allocation::admit(
            allocation_class,
            gpu_buffer_budget_bytes,
            persistent_buffer_bytes,
        )
        .map_err(|error| match error {
            allocation::AdmissionError::HotPathUnsupported => GpuError::HotZeroHeapUnsupported,
            allocation::AdmissionError::BudgetExceeded => GpuError::BufferBudgetExceeded,
        })?;
        if poll_timeout.is_zero() || poll_timeout > MAX_POLL_TIMEOUT {
            return Err(GpuError::InvalidTimeout);
        }
        let input_bytes = checked_bytes(max_pairs, BYTES_PER_INPUT_PAIR)?;
        let output_bytes = checked_bytes(max_pairs, BYTES_PER_OUTPUT_PAIR)?;
        let limits = device.limits();
        if input_bytes as u64 > u64::from(limits.max_storage_buffer_binding_size)
            || output_bytes as u64 > u64::from(limits.max_storage_buffer_binding_size)
            || input_bytes as u64 > limits.max_buffer_size
            || output_bytes as u64 > limits.max_buffer_size
            || PARAM_BYTES > u64::from(limits.max_storage_buffer_binding_size)
            || PARAM_BYTES > limits.max_buffer_size
            || limits.max_bind_groups < 1
            || limits.max_bindings_per_bind_group < 3
            || limits.max_storage_buffers_per_shader_stage < 3
            || limits.max_compute_invocations_per_workgroup < WORKGROUP_SIZE as u32
            || limits.max_compute_workgroup_size_x < WORKGROUP_SIZE as u32
            || limits.max_compute_workgroup_size_y < 1
            || limits.max_compute_workgroup_size_z < 1
            || max_pairs.div_ceil(WORKGROUP_SIZE)
                > limits.max_compute_workgroups_per_dimension as usize
        {
            return Err(GpuError::DeviceLimitExceeded);
        }

        let input = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-sense-iq-input"),
            size: input_bytes as u64,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let power_values = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-sense-iq-power"),
            size: output_bytes as u64,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-sense-iq-readback"),
            size: output_bytes as u64,
            usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-sense-iq-params"),
            size: PARAM_BYTES,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout =
            device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                label: Some("etalon-sense-iq-layout"),
                entries: &[
                    storage_entry(0, true, 4),
                    storage_entry(1, false, 4),
                    storage_entry(2, true, PARAM_BYTES),
                ],
            });
        let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("etalon-sense-iq-bindings"),
            layout: &bind_group_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: power_values.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("etalon-sense-iq-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(etalon::include_wgsl!("iq_power.wgsl"));
        let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("etalon-sense-iq-pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: etalon::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            device_id,
            allocation_class,
            persistent_buffer_bytes,
            max_pairs,
            poll_timeout,
            input,
            power_values,
            readback,
            params,
            bind_group,
            pipeline,
            completion: Arc::new(MapCompletion::new()),
            poisoned: false,
            last_backend_error: None,
            loss_state: None,
        })
    }

    pub const fn max_pairs(&self) -> usize {
        self.max_pairs
    }

    pub const fn allocation_class(&self) -> GpuAllocationClass {
        self.allocation_class
    }

    /// Capacity of persistent GPU buffers admitted at construction. This is
    /// not an estimate of total device or driver memory use.
    pub const fn persistent_buffer_bytes(&self) -> u64 {
        self.persistent_buffer_bytes
    }

    /// A detailed backend failure from the last failed `RegisteredSignalProvider`
    /// call. The shared signal trait has no backend-error variant, so its adapter
    /// maps these failures to `SignalError::InvalidParameter`; callers needing
    /// faithful errors should call `compute_into` directly.
    pub const fn last_backend_error(&self) -> Option<GpuError> {
        self.last_backend_error
    }

    pub(crate) fn is_device_lost(&self) -> bool {
        self.loss_state
            .is_some_and(|state| state.load(Ordering::Acquire))
    }

    /// Execute synchronously, preserving detailed Etalon failures.
    pub fn compute_into(
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
        for index in 0..count {
            let start = index * 4;
            workspace[index] = f32::from_le_bytes([
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

fn storage_entry(binding: u32, read_only: bool, min_size: u64) -> etalon::BindGroupLayoutEntry {
    etalon::BindGroupLayoutEntry {
        binding,
        visibility: etalon::ShaderStages::COMPUTE,
        ty: etalon::BindingType::Buffer {
            ty: etalon::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(min_size),
        },
        count: None,
    }
}

#[cfg(test)]
mod tests;
