//! Exact DSP target routing; no CPU or alternate-device fallback.
//!
//! Target identifiers and backend declarations are host registry assertions,
//! not device discovery or attestation. Native SDK providers implement the
//! existing live-processing trait and must execute on their bound DSP target.
#![no_std]
#![forbid(unsafe_code)]

use core::num::NonZeroU64;
use etalon_sense_core::{
    buffer::SampleLayout,
    processing::{ProcessingRequest, ProcessorKind, ProcessorPreference},
    processing_runtime::{
        dispatch_live, DispatchConstraints, LiveDispatchReceipt, LiveProcessingProvider,
        LiveProviderDescriptor, ProviderDispatchGrant, ProviderLifecycle, ProviderReport,
    },
    ContractError,
};

pub const MAX_DSP_BINDINGS: usize = 16;

/// Opaque host-registry vendor, processor model and device-instance IDs.
/// Equal IDs only mean an equal declaration within that host namespace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DspTarget {
    vendor: NonZeroU64,
    model: NonZeroU64,
    device: NonZeroU64,
}

impl DspTarget {
    pub const fn new(vendor: u64, model: u64, device: u64) -> Result<Self, DspError> {
        let Some(vendor) = NonZeroU64::new(vendor) else {
            return Err(DspError::InvalidTarget);
        };
        let Some(model) = NonZeroU64::new(model) else {
            return Err(DspError::InvalidTarget);
        };
        let Some(device) = NonZeroU64::new(device) else {
            return Err(DspError::InvalidTarget);
        };
        Ok(Self {
            vendor,
            model,
            device,
        })
    }
    pub const fn vendor(self) -> u64 {
        self.vendor.get()
    }
    pub const fn model(self) -> u64 {
        self.model.get()
    }
    pub const fn device(self) -> u64 {
        self.device.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DspError {
    InvalidTarget,
    Capacity,
    DuplicateBinding,
    WrongProcessor,
    Unavailable,
    StaleBinding,
    Contract(ContractError),
}

/// Borrow a live SDK provider without allocating or taking device ownership.
/// Restarting a provider requires a new binding with a fresh generation.
pub struct DspBinding<'a> {
    target: DspTarget,
    descriptor: LiveProviderDescriptor,
    provider: &'a mut dyn LiveProcessingProvider,
}

impl<'a> DspBinding<'a> {
    pub fn new(
        target: DspTarget,
        provider: &'a mut dyn LiveProcessingProvider,
    ) -> Result<Self, DspError> {
        let descriptor = provider.descriptor();
        if descriptor.processor() != ProcessorKind::Dsp {
            return Err(DspError::WrongProcessor);
        }
        if provider.lifecycle() != ProviderLifecycle::Ready {
            return Err(DspError::Unavailable);
        }
        Ok(Self {
            target,
            descriptor,
            provider,
        })
    }
    pub const fn target(&self) -> DspTarget {
        self.target
    }
}

/// Checked provider assertion only; not an execution or device attestation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DspDispatchReceipt {
    pub target: DspTarget,
    pub dispatch: LiveDispatchReceipt,
}

/// Dispatch one operation to exactly one selected target and generation.
/// Operation revision, encoding, shape, allocation class and workspace are
/// admitted by the existing core dispatcher. SDKs remain responsible for
/// device capability truth, bounded blocking, cancellation and no unwinding.
/// The SDK must atomically recheck the grant and hold its lifecycle lease
/// throughout native dispatch; successive wrapper checks cannot provide it.
/// Every returned error clears the entire output buffer.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_dsp(
    target: DspTarget,
    request: ProcessingRequest,
    constraints: DispatchConstraints,
    bindings: &mut [DspBinding<'_>],
    input_layout: &SampleLayout,
    input: &[u8],
    output_layout: &SampleLayout,
    output: &mut [u8],
    workspace: &mut [u8],
) -> Result<DspDispatchReceipt, DspError> {
    let result = dispatch_inner(
        target,
        request,
        constraints,
        bindings,
        input_layout,
        input,
        output_layout,
        output,
        workspace,
    );
    if result.is_err() {
        output.fill(0);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn dispatch_inner(
    target: DspTarget,
    request: ProcessingRequest,
    constraints: DispatchConstraints,
    bindings: &mut [DspBinding<'_>],
    input_layout: &SampleLayout,
    input: &[u8],
    output_layout: &SampleLayout,
    output: &mut [u8],
    workspace: &mut [u8],
) -> Result<DspDispatchReceipt, DspError> {
    if bindings.len() > MAX_DSP_BINDINGS {
        return Err(DspError::Capacity);
    }
    for (index, binding) in bindings.iter().enumerate() {
        if bindings[..index].iter().any(|prior| {
            prior.target == binding.target
                && prior.descriptor.operation() == binding.descriptor.operation()
        }) {
            return Err(DspError::DuplicateBinding);
        }
    }
    if !request.preference().permits(ProcessorKind::Dsp) {
        return Err(DspError::WrongProcessor);
    }
    let binding = bindings
        .iter_mut()
        .find(|binding| {
            binding.target == target && binding.descriptor.operation() == request.operation()
        })
        .ok_or(DspError::Unavailable)?;
    let descriptor = binding.provider.descriptor();
    if descriptor != binding.descriptor {
        return Err(DspError::StaleBinding);
    }
    if descriptor.processor() != ProcessorKind::Dsp {
        return Err(DspError::WrongProcessor);
    }
    let request = ProcessingRequest::new(
        request.operation(),
        ProcessorPreference::Require(ProcessorKind::Dsp),
        request.max_workspace_bytes(),
        request.latency_budget_ns(),
    )
    .map_err(DspError::Contract)?;
    let mut pinned = PinnedProvider {
        descriptor: binding.descriptor,
        provider: &mut *binding.provider,
    };
    let mut providers: [&mut dyn LiveProcessingProvider; 1] = [&mut pinned];
    let dispatch = dispatch_live(
        request,
        constraints,
        &mut providers,
        input_layout,
        input,
        output_layout,
        output,
        workspace,
    )
    .map_err(DspError::Contract)?;
    Ok(DspDispatchReceipt { target, dispatch })
}

/// The core dispatcher must never issue a grant from a newly observed
/// generation after the binding was checked. Forward only the frozen contract.
struct PinnedProvider<'a> {
    descriptor: LiveProviderDescriptor,
    provider: &'a mut dyn LiveProcessingProvider,
}

impl PinnedProvider<'_> {
    fn matches_ready(&self) -> bool {
        self.provider.lifecycle() == ProviderLifecycle::Ready
            && self.provider.descriptor() == self.descriptor
    }
}

impl LiveProcessingProvider for PinnedProvider<'_> {
    fn descriptor(&self) -> LiveProviderDescriptor {
        self.descriptor
    }
    fn lifecycle(&self) -> ProviderLifecycle {
        if self.matches_ready() {
            ProviderLifecycle::Ready
        } else {
            ProviderLifecycle::Faulted
        }
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
        if grant.provider_id() != self.descriptor.provider_id()
            || grant.generation() != self.descriptor.generation()
            || grant.operation() != self.descriptor.operation()
            || !self.matches_ready()
        {
            output.fill(0);
            return Err(ContractError::InvalidSource);
        }
        let report =
            self.provider
                .dispatch(grant, input_layout, input, output_layout, output, workspace)?;
        if !self.matches_ready() {
            output.fill(0);
            return Err(ContractError::InvalidSource);
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests;
