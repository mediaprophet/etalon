//! Explicit CUDA execution of shared Forge IQ-power kernels.

use crate::GpuAllocationClass;
use etalon_cuda::CudaIqPowerPlan;
use etalon_cuda::CudaKernels;
pub use etalon_cuda::{CudaContext, CudaDeviceInfo, CudaError};
use etalon_signal::{
    decode::ByteOrder,
    execution::{
        i16_iq_power_upper_bound, DeviceId, ExecutionAllocationClass, RegisteredSignalProvider,
    },
    SignalError,
};

/// Bounded IQ input with per-call allocations, GPU dispatch, and synchronous readback.
#[derive(Debug)]
pub struct CudaIqPowerProvider {
    kernels: CudaKernels,
    max_pairs: usize,
}

#[derive(Debug)]
pub enum PreparedCudaPowerError {
    InvalidCapacity,
    InvalidInput,
    InvalidScale,
    Capacity,
    NumericRange,
    HotZeroHeapUnsupported,
    BudgetExceeded,
    Backend(CudaError),
}

impl std::fmt::Display for PreparedCudaPowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for PreparedCudaPowerError {}

/// Reusable signed-i16 IQ decoding and FP32 power with caller-owned staging/output.
/// The device ID is a host routing declaration, not attestation or authorization.
/// Persistent buffers retain samples; driver allocations and waits are not bounded.
#[derive(Debug)]
pub struct PreparedCudaIqPower {
    plan: CudaIqPowerPlan,
    device_id: DeviceId,
}

fn admit_prepared(
    max_pairs: usize,
    class: GpuAllocationClass,
    budget: u64,
) -> Result<(), PreparedCudaPowerError> {
    if max_pairs == 0 || max_pairs > crate::MAX_IQ_PAIRS {
        return Err(PreparedCudaPowerError::InvalidCapacity);
    }
    if class == GpuAllocationClass::HotZeroHeap {
        return Err(PreparedCudaPowerError::HotZeroHeapUnsupported);
    }
    if max_pairs as u64 * 8 > budget {
        return Err(PreparedCudaPowerError::BudgetExceeded);
    }
    Ok(())
}

impl PreparedCudaIqPower {
    pub fn new(
        context: CudaContext,
        device_id: DeviceId,
        max_pairs: usize,
        buffer_budget: u64,
    ) -> Result<Self, PreparedCudaPowerError> {
        Self::new_with_allocation_class(
            context,
            device_id,
            max_pairs,
            GpuAllocationClass::ColdBoundedBuffers,
            buffer_budget,
        )
    }

    pub fn new_with_allocation_class(
        context: CudaContext,
        device_id: DeviceId,
        max_pairs: usize,
        class: GpuAllocationClass,
        buffer_budget: u64,
    ) -> Result<Self, PreparedCudaPowerError> {
        admit_prepared(max_pairs, class, buffer_budget)?;
        let plan = CudaIqPowerPlan::new(context, max_pairs, buffer_budget)
            .map_err(PreparedCudaPowerError::Backend)?;
        Ok(Self { plan, device_id })
    }

    pub fn device_info(&self) -> &CudaDeviceInfo {
        self.plan.device_info()
    }
    pub fn max_pairs(&self) -> usize {
        self.plan.max_pairs()
    }
    pub fn logical_buffer_bytes(&self) -> u64 {
        self.plan.logical_buffer_bytes()
    }
    pub fn completed_runs(&self) -> u64 {
        self.plan.completed_runs()
    }
    pub fn is_poisoned(&self) -> bool {
        self.plan.is_poisoned()
    }

    /// Workspace has two f32 slots per pair, matching the portable signal executor.
    /// Output is widened FP32 power. Workspace is wiped and output cleared on failure.
    pub fn compute_into(
        &mut self,
        bytes: &[u8],
        order: ByteOrder,
        scale: f32,
        workspace: &mut [f32],
        output: &mut [f64],
    ) -> Result<usize, PreparedCudaPowerError> {
        output.fill(0.0);
        let result = (|| {
            let count = crate::numeric::validate_request(
                bytes,
                scale,
                workspace.len(),
                output.len(),
                self.max_pairs(),
            )
            .map_err(|error| match error {
                crate::GpuError::InvalidScale => PreparedCudaPowerError::InvalidScale,
                crate::GpuError::CapacityExceeded | crate::GpuError::LengthMismatch => {
                    PreparedCudaPowerError::Capacity
                }
                _ => PreparedCudaPowerError::InvalidInput,
            })?;
            let upper = i16_iq_power_upper_bound(scale)
                .map_err(|_| PreparedCudaPowerError::NumericRange)?;
            self.plan
                .execute(
                    bytes,
                    matches!(order, ByteOrder::BigEndian),
                    scale,
                    &mut workspace[..count],
                )
                .map_err(PreparedCudaPowerError::Backend)?;
            crate::numeric::stage_power_values(&workspace[..count], output, upper)
                .map_err(|_| PreparedCudaPowerError::NumericRange)
        })();
        workspace.fill(0.0);
        if result.is_err() {
            output.fill(0.0);
        }
        result
    }
}

impl RegisteredSignalProvider for PreparedCudaIqPower {
    fn device_id(&self) -> DeviceId {
        self.device_id
    }
    fn execution_allocation_class(&self) -> ExecutionAllocationClass {
        ExecutionAllocationClass::ColdBoundedBuffers
    }
    fn i16_iq_power_into(
        &mut self,
        bytes: &[u8],
        order: ByteOrder,
        scale: f32,
        workspace: &mut [f32],
        output: &mut [f64],
    ) -> Result<usize, SignalError> {
        self.compute_into(bytes, order, scale, workspace, output)
            .map_err(|error| match error {
                PreparedCudaPowerError::Capacity | PreparedCudaPowerError::InvalidCapacity => {
                    SignalError::LengthMismatch
                }
                PreparedCudaPowerError::NumericRange => SignalError::NumericRange,
                _ => SignalError::InvalidParameter,
            })
    }
}

impl CudaIqPowerProvider {
    pub fn new(context: CudaContext, max_pairs: usize) -> Result<Self, CudaError> {
        if max_pairs == 0 || max_pairs > crate::MAX_IQ_PAIRS {
            return Err(CudaError::InvalidArgument("invalid CUDA IQ pair capacity"));
        }
        Ok(Self {
            kernels: CudaKernels::new(context)?,
            max_pairs,
        })
    }

    pub fn device_info(&self) -> &CudaDeviceInfo {
        self.kernels.context().device_info()
    }

    /// Compute I² + Q² for interleaved FP32 samples on the selected CUDA device.
    pub fn powers(&self, interleaved_iq: &[f32]) -> Result<Vec<f32>, CudaError> {
        if interleaved_iq.len() > self.max_pairs * 2 {
            return Err(CudaError::InvalidArgument(
                "CUDA IQ input exceeds pair capacity",
            ));
        }
        self.kernels.iq_power_f32(interleaved_iq)
    }
}

#[cfg(test)]
mod tests;
