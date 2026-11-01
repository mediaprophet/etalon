//! Explicit, caller-selected portable GPU execution for bounded sensing kernels.
//!
//! The ordinary provider accepts an existing Etalon device and queue. It does not discover
//! devices, authorize capture, attest a provider, qualify hardware, or fall back
//! to CPU execution. Async readback supports native and Browser WebGPU; the
//! blocking API and prepared Vulkan path remain native-only.
//! This provider is explicitly `ColdBoundedBuffers`, not a Tier-1 zero-heap hot path.
//! Etalon's `BufferSlice::map_async` boxes its callback, and its command encoder,
//! command buffer, and queue submission APIs do not provide a caller-owned
//! preallocated recording/submission arena. Strict `HotZeroHeap` admission is
//! therefore rejected before resources are created. Persistent GPU buffer
//! capacities are admitted against a caller byte budget; backend/pipeline and
//! transient command allocations are outside that buffer-only budget. Buffers
//! retain samples and powers until overwritten or destroyed; secure erasure is
//! not guaranteed.
//!
//! The `prepared` module instead prepares a private Vulkan lane on a selected
//! adapter. Its fixed resources avoid ordinary per-run command recording, but
//! its live descriptor remains `ColdUnbounded` pending target qualification.
#![forbid(unsafe_code)]

mod allocation;
pub use allocation::GpuAllocationClass;
mod async_compute;
#[cfg(not(target_arch = "wasm32"))]
mod blocking;
mod completion;
#[cfg(not(target_arch = "wasm32"))]
pub mod live;
mod numeric;
mod pair;
#[cfg(not(target_arch = "wasm32"))]
pub mod prepared;
use completion::MapCompletion;
use numeric::{
    checked_bytes, make_params, persistent_buffer_bytes, stage_power_values, validate_request,
};
pub use pair::{DeviceQueuePair, DeviceRequestGate, DeviceRequestStage, PairRequestError};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use etalon::{BindGroup, Buffer, ComputePipeline, Device, Queue};
pub use etalon_signal::{decode::ByteOrder, execution::DeviceId};
use etalon_signal::{execution::i16_iq_power_upper_bound, SignalError};

#[cfg(not(target_arch = "wasm32"))]
use etalon_signal::execution::{ExecutionAllocationClass, RegisteredSignalProvider};

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
    Busy,
    ValidationFailed,
    OutOfMemory,
    InternalError,
    AsyncTimerUnavailable,
}

/// Caller-selected Etalon device provider for signed-i16 I/Q power.
///
/// Each call uploads only the provided bytes, runs WGSL to decode the requested
/// byte order, apply `scale`, and compute IÂ² + QÂ² in f32. Results are converted
/// to f64 on readback; precision is therefore f32 and can differ from the
/// reference CPU pipeline's f64 sum. Calls are bounded by `max_pairs`; async
/// execution supports Browser WebGPU and blocking execution remains native-only. An Etalon device/ID match does not constitute GPU attestation
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
    active: AtomicBool,
    poisoned: bool,
    last_backend_error: Option<GpuError>,
    loss_state: Option<&'a AtomicBool>,
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
        if input_bytes as u64 > limits.max_storage_buffer_binding_size
            || output_bytes as u64 > limits.max_storage_buffer_binding_size
            || input_bytes as u64 > limits.max_buffer_size
            || output_bytes as u64 > limits.max_buffer_size
            || PARAM_BYTES > limits.max_storage_buffer_binding_size
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
            active: AtomicBool::new(false),
            poisoned: false,
            last_backend_error: None,
            loss_state: None,
        })
    }

    /// Caller registry identity supplied at construction; this is not physical-device attestation.
    pub const fn device_id(&self) -> DeviceId {
        self.device_id
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

impl Drop for EtalonIqPowerProvider<'_> {
    fn drop(&mut self) {
        self.readback.unmap();
        for buffer in [
            &self.input,
            &self.power_values,
            &self.readback,
            &self.params,
        ] {
            buffer.destroy();
        }
    }
}
