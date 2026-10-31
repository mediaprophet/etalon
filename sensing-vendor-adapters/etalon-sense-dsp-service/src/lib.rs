//! Native DSP service for the original Etalon byte-buffer ABI.
#![no_std]
#![deny(unsafe_code)]

use etalon_sense_core::{
    buffer::SampleLayout,
    processing::{OperationId, ProcessingRequest, ProcessorKind, ProcessorPreference},
    processing_runtime::{dispatch_live, AllocationClass, DispatchConstraints, LiveProcessingProvider},
    quantity::SampleEncoding,
};
use etalon_sense_dsp_hexagon::HexagonIqPowerProvider;
use etalon_sense_dsp_wire::{decode_request, finish_response, response_bytes,
    IQ_POWER_OPERATION_ID, IQ_POWER_REVISION, HEADER_BYTES, MAX_PAIRS};

#[cfg(feature = "ffi")]
#[allow(unsafe_code)]
mod ffi;
#[cfg(any(feature = "ffi", test))]
mod native_context;
#[cfg(all(feature = "ffi", not(target_has_atomic = "32")))]
compile_error!("The native DSP context requires lock-free 32-bit target atomics");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServiceError { UnsupportedPlatform, Configuration, Request, Identity, Capacity, Dispatch }

/// Immutable provider configuration for one admitted native context.
/// Identity comes from deployment and native open, not request data.
pub struct HexagonIqPowerService {
    provider: HexagonIqPowerProvider,
    provider_id: u64,
    generation: u32,
    max_pairs: u32,
}

impl HexagonIqPowerService {
    pub fn new(provider_id: u64, generation: u32, max_pairs: u32) -> Result<Self, ServiceError> {
        if !cfg!(target_arch = "hexagon") { return Err(ServiceError::UnsupportedPlatform); }
        if provider_id == 0 || generation == 0 || max_pairs == 0 || max_pairs > MAX_PAIRS {
            return Err(ServiceError::Configuration);
        }
        let provider = HexagonIqPowerProvider::new(provider_id, generation, max_pairs)
            .map_err(|_| ServiceError::Configuration)?;
        Ok(Self { provider, provider_id, generation, max_pairs })
    }

    /// A failed call clears the bounded caller response; no success header survives.
    pub fn process(&mut self, input: &[u8], output: &mut [u8]) -> Result<(), ServiceError> {
        let header_len = HEADER_BYTES.min(output.len());
        output[..header_len].fill(0);
        // Do not scan an arbitrarily large malformed output before enforcing the budget.
        if output.len() > response_bytes(self.max_pairs).map_err(|_| ServiceError::Capacity)? {
            return Err(ServiceError::Capacity);
        }
        output.fill(0);
        let result = self.process_inner(input, output);
        if result.is_err() { output.fill(0); }
        result
    }

    fn process_inner(&mut self, input: &[u8], output: &mut [u8]) -> Result<(), ServiceError> {
        let (identity, pairs, iq) = decode_request(input).map_err(|_| ServiceError::Request)?;
        if identity.provider_id() != self.provider_id || identity.generation() != self.generation {
            return Err(ServiceError::Identity);
        }
        if pairs > self.max_pairs || output.len() != response_bytes(pairs).map_err(|_| ServiceError::Capacity)? {
            return Err(ServiceError::Capacity);
        }
        let input_layout = SampleLayout::new(1, [pairs, 0, 0, 0], [4, 0, 0, 0], SampleEncoding::ComplexI16)
            .map_err(|_| ServiceError::Request)?;
        let output_layout = SampleLayout::new(1, [pairs, 0, 0, 0], [8, 0, 0, 0], SampleEncoding::F64)
            .map_err(|_| ServiceError::Request)?;
        let operation = OperationId::new(IQ_POWER_OPERATION_ID, IQ_POWER_REVISION)
            .map_err(|_| ServiceError::Configuration)?;
        let request = ProcessingRequest::new(operation, ProcessorPreference::Require(ProcessorKind::Dsp), 0, None)
            .map_err(|_| ServiceError::Configuration)?;
        let mut providers: [&mut dyn LiveProcessingProvider; 1] = [&mut self.provider];
        dispatch_live(request, DispatchConstraints::new(AllocationClass::HotZeroHeap, 0),
            &mut providers, &input_layout, iq, &output_layout, &mut output[HEADER_BYTES..], &mut [])
            .map_err(|_| ServiceError::Dispatch)?;
        finish_response(identity, pairs, output).map_err(|_| ServiceError::Capacity)
    }
}
