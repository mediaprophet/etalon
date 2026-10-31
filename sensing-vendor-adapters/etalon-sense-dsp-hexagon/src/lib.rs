//! Native Qualcomm Hexagon DSP execution with caller-owned buffers.
//!
//! This library executes on a Hexagon target. It does not load a remote DSP,
//! provide host-to-DSP RPC, discover hardware, or attest a deployment. Host
//! architectures fail closed; no CPU or alternate-provider fallback exists.
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
use etalon_sense_dsp::{DspBinding, DspError, DspTarget};

/// Shared revision with Etalon's exact CPU reference: LE complex i16 input,
/// LE f64 I²+Q² output, without scale or physical-unit/calibration claims.
pub const IQ_POWER_OPERATION_ID: u64 = 0x4951_504f_5745_5231;
pub const IQ_POWER_REVISION: u32 = 1;
pub const MAX_PAIRS: u32 = 1_048_576;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HexagonError {
    UnsupportedPlatform,
    Contract(ContractError),
}

pub struct HexagonIqPowerProvider {
    descriptor: LiveProviderDescriptor,
    max_pairs: u32,
    lifecycle: ProviderLifecycle,
}

impl HexagonIqPowerProvider {
    /// Target architecture is necessary but does not authenticate a physical
    /// device: emulators may also execute a Hexagon artifact. The caller owns
    /// firmware, ABI, toolchain, process/interrupt isolation and device policy.
    pub fn new(provider_id: u64, generation: u32, max_pairs: u32) -> Result<Self, HexagonError> {
        if !cfg!(target_arch = "hexagon") {
            return Err(HexagonError::UnsupportedPlatform);
        }
        if max_pairs == 0 || max_pairs > MAX_PAIRS {
            return Err(HexagonError::Contract(ContractError::InvalidExtent));
        }
        let operation = OperationId::new(IQ_POWER_OPERATION_ID, IQ_POWER_REVISION)
            .map_err(HexagonError::Contract)?;
        let descriptor = LiveProviderDescriptor::new(
            provider_id,
            generation,
            ProcessorKind::Dsp,
            operation,
            SampleEncoding::ComplexI16,
            SampleEncoding::F64,
            1,
            [max_pairs, 0, 0, 0],
            0,
            AllocationClass::HotZeroHeap,
            None,
        )
        .map_err(HexagonError::Contract)?;
        Ok(Self {
            descriptor,
            max_pairs,
            lifecycle: ProviderLifecycle::Ready,
        })
    }

    /// Registry IDs are host assertions. Device ID must agree with the bound
    /// provider ID; vendor/model IDs require the host's registry authority.
    pub fn bind(&mut self, target: DspTarget) -> Result<DspBinding<'_>, DspError> {
        if target.device() != self.descriptor.provider_id() {
            return Err(DspError::InvalidTarget);
        }
        DspBinding::new(target, self)
    }

    pub fn stop(&mut self) {
        self.lifecycle = ProviderLifecycle::Stopped;
    }
}

impl LiveProcessingProvider for HexagonIqPowerProvider {
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
        if !cfg!(target_arch = "hexagon") {
            return Err(ContractError::UnsupportedOperation);
        }
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
        write_power(input, output);
        Ok(ProviderReport::new(
            self.descriptor.provider_id(),
            self.descriptor.generation(),
            self.descriptor.operation(),
            in_len as u64,
            out_len as u64,
        ))
    }
}

fn write_power(input: &[u8], output: &mut [u8]) {
    for (pair, result) in input.chunks_exact(4).zip(output.chunks_exact_mut(8)) {
        let i = i64::from(i16::from_le_bytes([pair[0], pair[1]]));
        let q = i64::from(i16::from_le_bytes([pair[2], pair[3]]));
        result.copy_from_slice(&((i * i + q * q) as f64).to_le_bytes());
    }
}

#[cfg(test)]
mod tests;
