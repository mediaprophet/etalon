use crate::{FastRpcError, FastRpcSession, SessionState};
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
use etalon_sense_deployment::fastrpc::AdmittedFastRpcDeployment;
use etalon_sense_dsp::{DspBinding, DspError, DspTarget};
use etalon_sense_dsp_wire::{self as wire, RequestIdentity};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Concrete remote IQ-power operation through the original Etalon wire ABI.
/// Results assert completion by the trusted service, not physical attestation.
pub struct FastRpcIqPowerProvider<'lease> {
    session: FastRpcSession<'lease>,
    descriptor: LiveProviderDescriptor,
    lifecycle: ProviderLifecycle,
    next_request_id: u64,
    last_error: Option<FastRpcError>,
}

impl<'lease> FastRpcIqPowerProvider<'lease> {
    /// # Safety
    /// All safety obligations of [`FastRpcSession::open`] apply. The caller
    /// owns the provider-generation namespace and must never reuse a generation
    /// for this target after reset, restart or constructing another provider.
    pub unsafe fn open(
        deployment: &'lease AdmittedFastRpcDeployment<'_, '_, '_>,
        generation: u32,
        max_pairs: u32,
    ) -> Result<Self, FastRpcError> {
        if generation != deployment.provider_generation() {
            return Err(FastRpcError::DeploymentAdmission);
        }
        let descriptor = make_descriptor(deployment.target_id(), generation, max_pairs)?;
        let session = unsafe { FastRpcSession::open(deployment, max_pairs) }?;
        Ok(Self {
            session,
            descriptor,
            lifecycle: ProviderLifecycle::Ready,
            next_request_id: 1,
            last_error: None,
        })
    }

    pub fn bind(&mut self, target: DspTarget) -> Result<DspBinding<'_>, DspError> {
        if target.device() != self.session.target_id() {
            return Err(DspError::InvalidTarget);
        }
        DspBinding::new(target, self)
    }

    pub fn last_error(&self) -> Option<FastRpcError> {
        self.last_error
    }
    pub fn domain(&self) -> u32 {
        self.session.domain()
    }
    pub fn session_id(&self) -> u32 {
        self.session.session()
    }

    pub fn stop(&mut self) -> Result<(), FastRpcError> {
        self.lifecycle = ProviderLifecycle::Stopping;
        match self.session.close() {
            Ok(()) => {
                self.lifecycle = ProviderLifecycle::Stopped;
                Ok(())
            }
            Err(error) => {
                self.last_error = Some(error);
                self.lifecycle = ProviderLifecycle::Faulted;
                Err(error)
            }
        }
    }

    fn validate_call(
        &self,
        grant: &ProviderDispatchGrant,
        input_layout: &SampleLayout,
        input: &[u8],
        output_layout: &SampleLayout,
        output: &[u8],
        workspace: &[u8],
    ) -> Result<u32, FastRpcError> {
        if self.lifecycle != ProviderLifecycle::Ready
            || self.session.state() != SessionState::Open
            || grant.provider_id() != self.descriptor.provider_id()
            || grant.generation() != self.descriptor.generation()
            || grant.operation() != self.descriptor.operation()
        {
            return Err(FastRpcError::InvalidState);
        }
        if !self.session.authority_current() {
            return Err(FastRpcError::DeploymentAdmission);
        }
        if input_layout.rank() != 1
            || output_layout.rank() != 1
            || *input_layout.encoding() != SampleEncoding::ComplexI16
            || *output_layout.encoding() != SampleEncoding::F64
            || input_layout.extents()[0] != output_layout.extents()[0]
            || (input_layout.extents()[0] > 1 && input_layout.strides_bytes()[0] != 4)
            || (output_layout.extents()[0] > 1 && output_layout.strides_bytes()[0] != 8)
        {
            return Err(FastRpcError::Contract(ContractError::IncompatibleEncoding));
        }
        let pairs = input_layout.extents()[0];
        if pairs == 0 || pairs > self.session.max_pairs() {
            return Err(FastRpcError::BufferSize);
        }
        let request_bytes = wire::request_bytes(pairs).map_err(|_| FastRpcError::BufferSize)?;
        let response_bytes = wire::response_bytes(pairs).map_err(|_| FastRpcError::BufferSize)?;
        if input.len() != request_bytes - wire::HEADER_BYTES
            || output.len() != response_bytes - wire::HEADER_BYTES
            || workspace.len() < request_bytes + response_bytes
        {
            return Err(FastRpcError::BufferSize);
        }
        Ok(pairs)
    }

    fn run(
        &mut self,
        pairs: u32,
        input: &[u8],
        output: &mut [u8],
        workspace: &mut [u8],
    ) -> Result<ProviderReport, FastRpcError> {
        let request_id = self.next_request_id;
        self.next_request_id = request_id
            .checked_add(1)
            .ok_or(FastRpcError::RequestExhausted)?;
        let identity = RequestIdentity::new(
            self.descriptor.provider_id(),
            self.descriptor.generation(),
            request_id,
        )
        .map_err(|_| FastRpcError::Wire)?;
        let request_bytes = wire::request_bytes(pairs).map_err(|_| FastRpcError::BufferSize)?;
        let response_bytes = wire::response_bytes(pairs).map_err(|_| FastRpcError::BufferSize)?;
        let scratch = Scratch(&mut workspace[..request_bytes + response_bytes]);
        let (request, response) = scratch.0.split_at_mut(request_bytes);
        wire::encode_request(identity, pairs, input, request).map_err(|_| FastRpcError::Wire)?;
        self.session.invoke_buffers(request, response)?;
        let payload =
            wire::decode_response(identity, pairs, response).map_err(|_| FastRpcError::Wire)?;
        if payload.len() != output.len() {
            return Err(FastRpcError::BufferSize);
        }
        if !self.session.authority_current() {
            return Err(FastRpcError::DeploymentAdmission);
        }
        output.copy_from_slice(payload);
        Ok(ProviderReport::new(
            self.descriptor.provider_id(),
            self.descriptor.generation(),
            self.descriptor.operation(),
            input.len() as u64,
            output.len() as u64,
        ))
    }
}

impl LiveProcessingProvider for FastRpcIqPowerProvider<'_> {
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
        workspace: &mut [u8],
    ) -> Result<ProviderReport, ContractError> {
        output.fill(0);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let pairs = self.validate_call(
                &grant,
                input_layout,
                input,
                output_layout,
                output,
                workspace,
            )?;
            self.lifecycle = ProviderLifecycle::Busy;
            self.run(pairs, input, output, workspace)
        }))
        .unwrap_or(Err(FastRpcError::InvalidState));
        match result {
            Ok(report) => {
                self.lifecycle = ProviderLifecycle::Ready;
                self.last_error = None;
                Ok(report)
            }
            Err(error) => {
                output.fill(0);
                self.last_error = Some(error);
                // Stopping/stopped providers must never become Ready again.
                if self.lifecycle != ProviderLifecycle::Stopped {
                    self.lifecycle = ProviderLifecycle::Faulted;
                }
                Err(match error {
                    FastRpcError::Contract(error) => error,
                    FastRpcError::BufferSize => ContractError::OutOfBounds,
                    _ => ContractError::InvalidSource,
                })
            }
        }
    }
}

fn make_descriptor(
    provider_id: u64,
    generation: u32,
    max_pairs: u32,
) -> Result<LiveProviderDescriptor, FastRpcError> {
    if max_pairs == 0 || max_pairs > wire::MAX_PAIRS {
        return Err(FastRpcError::InvalidLimits);
    }
    let workspace = wire::request_bytes(max_pairs)
        .map_err(|_| FastRpcError::InvalidLimits)?
        .checked_add(wire::response_bytes(max_pairs).map_err(|_| FastRpcError::InvalidLimits)?)
        .ok_or(FastRpcError::InvalidLimits)?;
    let operation = OperationId::new(wire::IQ_POWER_OPERATION_ID, wire::IQ_POWER_REVISION)
        .map_err(FastRpcError::Contract)?;
    LiveProviderDescriptor::new(
        provider_id,
        generation,
        ProcessorKind::Dsp,
        operation,
        SampleEncoding::ComplexI16,
        SampleEncoding::F64,
        1,
        [max_pairs, 0, 0, 0],
        workspace as u64,
        AllocationClass::ColdUnbounded,
        None,
    )
    .map_err(FastRpcError::Contract)
}

struct Scratch<'a>(&'a mut [u8]);
impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        self.0.fill(0);
    }
}

#[cfg(test)]
#[path = "provider_tests.rs"]
mod tests;
