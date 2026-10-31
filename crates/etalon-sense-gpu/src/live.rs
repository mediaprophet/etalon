//! Explicit live dispatch adapter. Internal GPU allocations are unbounded.
use std::panic::{catch_unwind, AssertUnwindSafe};

use etalon_sense_core::{
    buffer::SampleLayout,
    processing::{OperationId, ProcessorKind},
    processing_runtime::{
        AllocationClass, LiveProcessingProvider, LiveProviderDescriptor, ProviderDispatchGrant,
        ProviderLifecycle, ProviderReport,
    },
    quantity::SampleEncoding,
    ContractError,
};
use etalon_signal::decode::ByteOrder;

use super::{EtalonIqPowerProvider, GpuError};

/// Revision 1 maps little-endian complex i16 to little-endian f64, with the
/// calculation performed in f32. It is uncalibrated digital power.
pub const GPU_IQ_POWER_F32_OPERATION_ID: u64 = 0x4951_5046_3332_3031;
pub const GPU_IQ_POWER_F32_REVISION: u32 = 1;

/// Staging slices remain caller-owned and must outlive the adapter. The
/// staging budget covers the required portions of these slices only. GPU
/// buffers, commands, callback and driver allocations are not covered.
/// The provider must not be used in a Sentinel or hard-budget context.
/// Staging slices retain sample-derived values until the caller overwrites them.
pub struct LiveGpuIqPower<'a, 'device> {
    provider: &'a mut EtalonIqPowerProvider<'device>,
    descriptor: LiveProviderDescriptor,
    float_staging: &'a mut [f32],
    result_staging: &'a mut [f64],
    staging_bytes: u64,
    faulted: bool,
    last_error: Option<GpuError>,
}

impl<'a, 'device> LiveGpuIqPower<'a, 'device> {
    pub fn new(
        provider: &'a mut EtalonIqPowerProvider<'device>,
        generation: u32,
        float_staging: &'a mut [f32],
        result_staging: &'a mut [f64],
        staging_budget_bytes: u64,
    ) -> Result<Self, ContractError> {
        let count = provider.max_pairs();
        let float_len = count
            .checked_mul(2)
            .ok_or(ContractError::ArithmeticOverflow)?;
        let staging_bytes = count
            .checked_mul(16)
            .ok_or(ContractError::ArithmeticOverflow)?;
        let staging_bytes =
            u64::try_from(staging_bytes).map_err(|_| ContractError::ArithmeticOverflow)?;
        if float_staging.len() < float_len
            || result_staging.len() < count
            || staging_bytes > staging_budget_bytes
        {
            return Err(ContractError::OutOfBounds);
        }
        if provider.poisoned {
            return Err(ContractError::InvalidSource);
        }
        let count = u32::try_from(count).map_err(|_| ContractError::InvalidExtent)?;
        let operation = OperationId::new(GPU_IQ_POWER_F32_OPERATION_ID, GPU_IQ_POWER_F32_REVISION)?;
        let descriptor = LiveProviderDescriptor::new(
            provider.device_id.value(),
            generation,
            ProcessorKind::Gpu,
            operation,
            SampleEncoding::ComplexI16,
            SampleEncoding::F64,
            1,
            [count, 0, 0, 0],
            // Staging is attached at construction, not supplied as per-call workspace.
            0,
            AllocationClass::ColdUnbounded,
            None,
        )?;
        Ok(Self {
            provider,
            descriptor,
            float_staging,
            result_staging,
            staging_bytes,
            faulted: false,
            last_error: None,
        })
    }

    pub const fn staging_bytes(&self) -> u64 {
        self.staging_bytes
    }
    pub const fn last_error(&self) -> Option<GpuError> {
        self.last_error
    }
}

impl LiveProcessingProvider for LiveGpuIqPower<'_, '_> {
    fn descriptor(&self) -> LiveProviderDescriptor {
        self.descriptor
    }
    fn lifecycle(&self) -> ProviderLifecycle {
        if self.faulted || self.provider.poisoned || self.provider.is_device_lost() {
            ProviderLifecycle::Faulted
        } else {
            ProviderLifecycle::Ready
        }
    }

    fn dispatch(
        &mut self,
        grant: ProviderDispatchGrant,
        input_layout: &SampleLayout,
        input: &[u8],
        output_layout: &SampleLayout,
        output: &mut [u8],
        _workspace: &mut [u8],
    ) -> Result<ProviderReport, ContractError> {
        if self.lifecycle() != ProviderLifecycle::Ready
            || grant.provider_id() != self.descriptor.provider_id()
            || grant.generation() != self.descriptor.generation()
            || grant.operation() != self.descriptor.operation()
        {
            return Err(ContractError::InvalidSource);
        }
        if input_layout.rank() != 1
            || output_layout.rank() != 1
            || *input_layout.encoding() != SampleEncoding::ComplexI16
            || *output_layout.encoding() != SampleEncoding::F64
            || input_layout.extents()[0] != output_layout.extents()[0]
            || (input_layout.extents()[0] > 1 && input_layout.strides_bytes()[0] != 4)
            || (output_layout.extents()[0] > 1 && output_layout.strides_bytes()[0] != 8)
        {
            return Err(ContractError::IncompatibleEncoding);
        }
        let count = input_layout.extents()[0] as usize;
        if count > self.provider.max_pairs() {
            return Err(ContractError::OutOfBounds);
        }
        let in_len = count
            .checked_mul(4)
            .ok_or(ContractError::ArithmeticOverflow)?;
        let out_len = count
            .checked_mul(8)
            .ok_or(ContractError::ArithmeticOverflow)?;
        if input.len() != in_len || output.len() != out_len {
            return Err(ContractError::OutOfBounds);
        }
        self.last_error = None;
        let result = catch_unwind(AssertUnwindSafe(|| {
            self.provider.compute_into(
                input,
                ByteOrder::LittleEndian,
                1.0,
                &mut self.float_staging[..count * 2],
                &mut self.result_staging[..count],
            )
        }));
        match result {
            Ok(Ok(written)) if written == count => {}
            Ok(Err(error)) => {
                self.last_error = Some(error);
                self.faulted = true;
                self.provider.poisoned = true;
                return Err(ContractError::InvalidSource);
            }
            _ => {
                self.faulted = true;
                self.provider.poisoned = true;
                return Err(ContractError::InvalidSource);
            }
        }
        for (result, bytes) in self.result_staging[..count]
            .iter()
            .zip(output.chunks_exact_mut(8))
        {
            bytes.copy_from_slice(&result.to_le_bytes());
        }
        Ok(ProviderReport::new(
            self.descriptor.provider_id(),
            self.descriptor.generation(),
            self.descriptor.operation(),
            in_len as u64,
            out_len as u64,
        ))
    }
}
