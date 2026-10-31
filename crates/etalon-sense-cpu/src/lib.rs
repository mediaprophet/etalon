//! Zero-allocation CPU reference provider for raw complex-i16 power.
#![no_std]
#![forbid(unsafe_code)]

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

/// Revision 1: little-endian complex i16 pairs to little-endian f64 I²+Q².
/// Results are unscaled digital power, not a calibrated physical quantity.
pub const IQ_POWER_OPERATION_ID: u64 = 0x4951_504f_5745_5231;
pub const IQ_POWER_REVISION: u32 = 1;
pub const MAX_PAIRS: u32 = 1_048_576;

pub struct CpuIqPowerProvider {
    descriptor: LiveProviderDescriptor,
    lifecycle: ProviderLifecycle,
    max_pairs: u32,
}

impl CpuIqPowerProvider {
    pub fn new(provider_id: u64, generation: u32, max_pairs: u32) -> Result<Self, ContractError> {
        if max_pairs == 0 || max_pairs > MAX_PAIRS {
            return Err(ContractError::InvalidExtent);
        }
        let operation = OperationId::new(IQ_POWER_OPERATION_ID, IQ_POWER_REVISION)?;
        let descriptor = LiveProviderDescriptor::new(
            provider_id,
            generation,
            ProcessorKind::Cpu,
            operation,
            SampleEncoding::ComplexI16,
            SampleEncoding::F64,
            1,
            [max_pairs, 0, 0, 0],
            0,
            AllocationClass::HotZeroHeap,
            None,
        )?;
        Ok(Self {
            descriptor,
            lifecycle: ProviderLifecycle::Ready,
            max_pairs,
        })
    }

    pub fn stop(&mut self) {
        self.lifecycle = ProviderLifecycle::Stopped;
    }
}

impl LiveProcessingProvider for CpuIqPowerProvider {
    fn descriptor(&self) -> LiveProviderDescriptor {
        self.descriptor
    }
    fn lifecycle(&self) -> ProviderLifecycle {
        self.lifecycle
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
        if self.lifecycle != ProviderLifecycle::Ready
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
        if input_layout.extents()[0] > self.max_pairs {
            return Err(ContractError::OutOfBounds);
        }
        let count = input_layout.extents()[0] as usize;
        let in_len = count
            .checked_mul(4)
            .ok_or(ContractError::ArithmeticOverflow)?;
        let out_len = count
            .checked_mul(8)
            .ok_or(ContractError::ArithmeticOverflow)?;
        if input.len() != in_len || output.len() != out_len {
            return Err(ContractError::OutOfBounds);
        }
        for (pair, result) in input.chunks_exact(4).zip(output.chunks_exact_mut(8)) {
            let i = i64::from(i16::from_le_bytes([pair[0], pair[1]]));
            let q = i64::from(i16::from_le_bytes([pair[2], pair[3]]));
            result.copy_from_slice(&((i * i + q * q) as f64).to_le_bytes());
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

#[cfg(test)]
mod tests;
