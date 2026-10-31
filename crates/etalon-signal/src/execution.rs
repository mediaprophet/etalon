//! Explicit execution routing for the caller-buffered IQ power pipeline.
//!
//! The CPU route calls the crate's existing integer decode and I/Q power
//! implementation. An accelerator route can only call a provider instance
//! supplied by the host and matched to an opaque registered device ID. This
//! module does not discover, qualify, attest, or claim hardware execution.
use core::num::NonZeroU64;

use crate::{decode::ByteOrder, pipeline::i16_iq_power_into, SignalError};

/// Opaque nonzero device handle in a host-owned registry namespace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(NonZeroU64);

impl DeviceId {
    pub const fn new(value: u64) -> Result<Self, ExecutionError> {
        match NonZeroU64::new(value) {
            Some(value) => Ok(Self(value)),
            None => Err(ExecutionError::InvalidDeviceId),
        }
    }

    pub const fn value(self) -> u64 {
        self.0.get()
    }
}

/// Exact execution selection. Device routing never silently falls back to CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionTarget {
    Cpu,
    RegisteredDevice(DeviceId),
}

/// Allocation class claimed for the selected execution path. `Unknown` is the
/// default for external providers and cannot satisfy a strict admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionAllocationClass {
    Unknown,
    HotZeroHeap,
    /// Persistent buffers have a caller-selected bound; transient/backend
    /// allocations may remain outside that budget.
    ColdBoundedBuffers,
}

/// Allocation policy selected by the caller for one execution request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationAdmission {
    /// Do not make an allocation-class claim; this is the compatibility default.
    Unrestricted,
    /// Require a path declared `HotZeroHeap` before any provider dispatch.
    RequireHotZeroHeap,
    /// Allow code-declared hot paths or paths with explicitly bounded persistent buffers.
    AllowColdBoundedBuffers,
}

fn allocation_admitted(class: ExecutionAllocationClass, admission: AllocationAdmission) -> bool {
    match admission {
        AllocationAdmission::Unrestricted => true,
        AllocationAdmission::RequireHotZeroHeap => class == ExecutionAllocationClass::HotZeroHeap,
        AllocationAdmission::AllowColdBoundedBuffers => matches!(
            class,
            ExecutionAllocationClass::HotZeroHeap | ExecutionAllocationClass::ColdBoundedBuffers
        ),
    }
}

/// Host-supplied processor implementation for the IQ-power operation.
///
/// A host registry may return an implementation for `device_id`. The ID match
/// is only routing consistency; implementing this trait does not prove that
/// the provider is hardware backed, qualified, or honestly reports execution.
pub trait RegisteredSignalProvider {
    fn device_id(&self) -> DeviceId;

    /// Declare the allocation class of this provider path. The default is
    /// `Unknown`, which fails strict admissions. Declarations are not measured
    /// or independently verified by this trait.
    fn execution_allocation_class(&self) -> ExecutionAllocationClass {
        ExecutionAllocationClass::Unknown
    }

    /// Write one power value per I/Q pair into `staged_output`.
    ///
    /// The output uses caller-provided buffers, but this does not prohibit
    /// provider-internal allocations. The outer executor validates output count
    /// and numeric range before it copies results to the public output buffer.
    fn i16_iq_power_into(
        &mut self,
        bytes: &[u8],
        byte_order: ByteOrder,
        scale: f32,
        workspace: &mut [f32],
        staged_output: &mut [f64],
    ) -> Result<usize, SignalError>;
}

/// Execution failures distinguish signal errors from provider/route failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionError {
    InvalidDeviceId,
    EmptyInput,
    Capacity,
    UnexpectedProvider,
    DeviceUnavailable,
    DeviceIdMismatch,
    AllocationClassRejected,
    InvalidProviderResult,
    Signal(SignalError),
}

/// Conservative upper bound for uncalibrated power of one signed-i16 I/Q pair.
///
/// The bound includes four f32 epsilons of relative rounding allowance for
/// amplitude products, squaring, and addition, plus four smallest f32
/// subnormals for underflow rounding. This admits valid CPU/GPU f32 rounding
/// while still rejecting values outside the declared signed-i16 domain.
pub fn i16_iq_power_upper_bound(scale: f32) -> Result<f64, SignalError> {
    if !scale.is_finite() {
        return Err(SignalError::InvalidParameter);
    }
    if scale == 0.0 {
        return Ok(0.0);
    }
    let max_component = 32768.0f64 * f64::from(scale.abs());
    let mathematical_max = 2.0 * max_component * max_component;
    let relative_rounding = 1.0 + 4.0 * f64::from(f32::EPSILON);
    let subnormal_slack = 4.0 * f64::from(f32::from_bits(1));
    let bound = mathematical_max * relative_rounding + subnormal_slack;
    if !bound.is_finite() {
        return Err(SignalError::NumericRange);
    }
    Ok(bound)
}

/// Execute signed-i16 interleaved I/Q decoding and uncalibrated power without
/// enforcing an allocation-class requirement. For compatibility, provider
/// class defaults to `Unknown`; callers needing a zero-heap guarantee must use
/// [`execute_i16_iq_power_into_with_admission`] with `RequireHotZeroHeap`.
#[allow(clippy::too_many_arguments)]
pub fn execute_i16_iq_power_into(
    target: ExecutionTarget,
    provider: Option<&mut dyn RegisteredSignalProvider>,
    bytes: &[u8],
    byte_order: ByteOrder,
    scale: f32,
    workspace: &mut [f32],
    staged_output: &mut [f64],
    output: &mut [f64],
) -> Result<usize, ExecutionError> {
    execute_i16_iq_power_into_with_admission(
        target,
        provider,
        bytes,
        byte_order,
        scale,
        workspace,
        staged_output,
        output,
        AllocationAdmission::Unrestricted,
    )
}

/// Execute signed-i16 interleaved I/Q decoding subject to explicit allocation
/// admission. Admission is checked after shape preflight but before clearing
/// staging buffers or invoking the provider, so rejection is transactional.
///
/// Buffer contracts are exact: `workspace` must hold two f32 scalars per I/Q
/// pair, while `staged_output` and `output` must each hold one f64 per pair.
/// All results are staged and checked before `output` is modified. On success
/// only `output[..count]` is written; this API requires exact output capacity,
/// so no tail is present. The CPU route uses the existing reference pipeline.
/// A registered-device route requires a caller-supplied provider and never
/// substitutes CPU computation on lookup or provider failure.
pub fn execute_i16_iq_power_into_with_admission(
    target: ExecutionTarget,
    provider: Option<&mut dyn RegisteredSignalProvider>,
    bytes: &[u8],
    byte_order: ByteOrder,
    scale: f32,
    workspace: &mut [f32],
    staged_output: &mut [f64],
    output: &mut [f64],
    admission: AllocationAdmission,
) -> Result<usize, ExecutionError> {
    if bytes.is_empty() {
        return Err(ExecutionError::EmptyInput);
    }
    if bytes.len() % 4 != 0 {
        return Err(ExecutionError::Signal(SignalError::LengthMismatch));
    }
    let count = bytes.len() / 4;
    if workspace.len() != bytes.len() / 2 || staged_output.len() != count || output.len() != count {
        return Err(ExecutionError::Capacity);
    }
    let allocation_class = match target {
        ExecutionTarget::Cpu => ExecutionAllocationClass::HotZeroHeap,
        ExecutionTarget::RegisteredDevice(expected_id) => {
            let provider = provider
                .as_deref()
                .ok_or(ExecutionError::DeviceUnavailable)?;
            if provider.device_id() != expected_id {
                return Err(ExecutionError::DeviceIdMismatch);
            }
            provider.execution_allocation_class()
        }
    };
    if !allocation_admitted(allocation_class, admission) {
        return Err(ExecutionError::AllocationClassRejected);
    }
    let max_power = i16_iq_power_upper_bound(scale).map_err(ExecutionError::Signal)?;

    // A provider that returns a full count without writing every slot must not
    // accidentally publish values left over from a previous invocation.
    staged_output.fill(f64::NAN);
    let written = match target {
        ExecutionTarget::Cpu => {
            if provider.is_some() {
                return Err(ExecutionError::UnexpectedProvider);
            }
            i16_iq_power_into(bytes, byte_order, scale, workspace, staged_output)
                .map_err(ExecutionError::Signal)?
        }
        ExecutionTarget::RegisteredDevice(expected_id) => {
            let provider = provider.ok_or(ExecutionError::DeviceUnavailable)?;
            if provider.device_id() != expected_id {
                return Err(ExecutionError::DeviceIdMismatch);
            }
            provider
                .i16_iq_power_into(bytes, byte_order, scale, workspace, staged_output)
                .map_err(ExecutionError::Signal)?
        }
    };

    // Provider values are declarations until checked against the physical
    // i16 input domain and the same f32-rounding allowance as the CPU route.
    if written != count
        || staged_output[..count]
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0 || *value > max_power)
    {
        return Err(ExecutionError::InvalidProviderResult);
    }

    output.copy_from_slice(&staged_output[..count]);
    Ok(count)
}

#[cfg(test)]
#[path = "execution_tests.rs"]
mod tests;
