//! Bounded dispatch through caller-supplied live providers.
//!
//! This module is deliberately separate from `processing_route`: planner
//! candidates are declarations and cannot be converted into executable
//! providers. A provider is an integration supplied by the host. The core
//! validates its declarations and synchronous completion report, but does not
//! discover devices, allocate memory, enforce an OS deadline, or attest that
//! provider claims are truthful.
use crate::{
    buffer::SampleLayout,
    processing::{OperationId, ProcessingRequest, ProcessorKind, ProcessorPreference},
    quantity::SampleEncoding,
    ContractError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderLifecycle {
    Ready,
    Busy,
    Lost,
    Stopping,
    Stopped,
    Faulted,
}

/// Allocation behavior declared for this operation by a provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationClass {
    /// Provider promises no heap allocation during dispatch.
    HotZeroHeap,
    /// Cold provider may allocate only under a separately enforced provider
    /// bound. The caller workspace is not an allocator-accounting mechanism.
    ColdBounded,
    /// Explicit opt-in for execution whose internal allocations are not
    /// bounded by this contract. Not suitable for a hard memory budget or
    /// a Sentinel arena. Host isolation and resource controls remain required.
    ColdUnbounded,
}

/// Immutable declaration of one provider operation and its accepted shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveProviderDescriptor {
    provider_id: u64,
    generation: u32,
    processor: ProcessorKind,
    operation: OperationId,
    input_encoding: SampleEncoding,
    output_encoding: SampleEncoding,
    max_rank: u8,
    max_extents: [u32; 4],
    workspace_bytes: u64,
    allocation_class: AllocationClass,
    /// Provider-declared upper bound. It is not independently measured.
    latency_bound_ns: Option<u64>,
}

impl LiveProviderDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        provider_id: u64,
        generation: u32,
        processor: ProcessorKind,
        operation: OperationId,
        input_encoding: SampleEncoding,
        output_encoding: SampleEncoding,
        max_rank: u8,
        max_extents: [u32; 4],
        workspace_bytes: u64,
        allocation_class: AllocationClass,
        latency_bound_ns: Option<u64>,
    ) -> Result<Self, ContractError> {
        if provider_id == 0 || generation == 0 {
            return Err(ContractError::InvalidIdentifier);
        }
        if max_rank == 0 || max_rank > 4 {
            return Err(ContractError::InvalidExtent);
        }
        let mut axis = 0usize;
        while axis < 4 {
            if (axis < max_rank as usize && max_extents[axis] == 0)
                || (axis >= max_rank as usize && max_extents[axis] != 0)
            {
                return Err(ContractError::InvalidExtent);
            }
            axis += 1;
        }
        if let Some(0) = latency_bound_ns {
            return Err(ContractError::InvalidTime);
        }
        Ok(Self {
            provider_id,
            generation,
            processor,
            operation,
            input_encoding,
            output_encoding,
            max_rank,
            max_extents,
            workspace_bytes,
            allocation_class,
            latency_bound_ns,
        })
    }

    pub const fn provider_id(self) -> u64 {
        self.provider_id
    }
    pub const fn generation(self) -> u32 {
        self.generation
    }
    pub const fn processor(self) -> ProcessorKind {
        self.processor
    }
    pub const fn operation(self) -> OperationId {
        self.operation
    }
    pub const fn workspace_bytes(self) -> u64 {
        self.workspace_bytes
    }
    pub const fn allocation_class(self) -> AllocationClass {
        self.allocation_class
    }
    pub const fn latency_bound_ns(self) -> Option<u64> {
        self.latency_bound_ns
    }
}

/// Runtime constraints beyond the operation and processor preference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchConstraints {
    allocation_class: AllocationClass,
    max_workspace_bytes: u64,
}

impl DispatchConstraints {
    pub const fn new(allocation_class: AllocationClass, max_workspace_bytes: u64) -> Self {
        Self {
            allocation_class,
            max_workspace_bytes,
        }
    }
}

/// Opaque, per-call grant. It is not policy authorization or a reusable plan.
pub struct ProviderDispatchGrant {
    provider_id: u64,
    generation: u32,
    operation: OperationId,
}

impl ProviderDispatchGrant {
    pub const fn provider_id(&self) -> u64 {
        self.provider_id
    }
    pub const fn generation(&self) -> u32 {
        self.generation
    }
    pub const fn operation(&self) -> OperationId {
        self.operation
    }
}

/// Provider-declared completion data, checked against the admitted call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderReport {
    provider_id: u64,
    generation: u32,
    operation: OperationId,
    bytes_read: u64,
    bytes_written: u64,
}

impl ProviderReport {
    pub const fn new(
        provider_id: u64,
        generation: u32,
        operation: OperationId,
        bytes_read: u64,
        bytes_written: u64,
    ) -> Self {
        Self {
            provider_id,
            generation,
            operation,
            bytes_read,
            bytes_written,
        }
    }
}

/// A synchronous live implementation supplied by the host.
///
/// A generation must advance on restart and must never be reused. `dispatch`
/// must recheck the grant against its current generation and
/// lifecycle atomically before reading inputs, hold a lifecycle lease until
/// return, serialize against stop/reset, honor the declared allocation and
/// latency bounds, and never unwind across this boundary. Those properties
/// cannot be enforced by this no-std contract. Asynchronous providers need a
/// separate completion/cancellation contract and are not admitted here.
pub trait LiveProcessingProvider {
    fn descriptor(&self) -> LiveProviderDescriptor;
    fn lifecycle(&self) -> ProviderLifecycle;
    fn dispatch(
        &mut self,
        grant: ProviderDispatchGrant,
        input_layout: &SampleLayout,
        input: &[u8],
        output_layout: &SampleLayout,
        output: &mut [u8],
        workspace: &mut [u8],
    ) -> Result<ProviderReport, ContractError>;
}

/// Successful synchronous dispatch. This is a checked provider assertion,
/// not a cryptographic execution receipt or hardware attestation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveDispatchReceipt {
    pub provider_id: u64,
    pub generation: u32,
    pub processor: ProcessorKind,
    pub operation: OperationId,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub allocation_class: AllocationClass,
}

/// Dispatches through a currently supplied provider after rechecking live
/// descriptor, lifecycle, operation, shape, workspace, and latency claims.
/// On any failure after provider invocation, the entire output slice is zeroed.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_live(
    request: ProcessingRequest,
    constraints: DispatchConstraints,
    providers: &mut [&mut dyn LiveProcessingProvider],
    input_layout: &SampleLayout,
    input: &[u8],
    output_layout: &SampleLayout,
    output: &mut [u8],
    workspace: &mut [u8],
) -> Result<LiveDispatchReceipt, ContractError> {
    let input_span = input_layout.required_byte_span()?;
    let output_span = output_layout.required_byte_span()?;
    if (input.len() as u128) < (input_span as u128)
        || (output.len() as u128) < (output_span as u128)
    {
        output.fill(0);
        return Err(ContractError::OutOfBounds);
    }

    let preferred = match request.preference() {
        ProcessorPreference::Prefer(kind) | ProcessorPreference::Require(kind) => Some(kind),
        ProcessorPreference::AnySupported => None,
    };
    let mut selected = None;
    if let Some(kind) = preferred {
        selected = find_provider(
            Some(kind),
            request,
            constraints,
            providers,
            input_layout,
            output_layout,
            workspace.len(),
        );
    }
    if selected.is_none() && !matches!(request.preference(), ProcessorPreference::Require(_)) {
        selected = find_provider(
            None,
            request,
            constraints,
            providers,
            input_layout,
            output_layout,
            workspace.len(),
        );
    }
    let Some(index) = selected else {
        output.fill(0);
        return Err(ContractError::UnsupportedOperation);
    };

    let provider = &mut providers[index];
    let descriptor = provider.descriptor();
    if provider.lifecycle() != ProviderLifecycle::Ready
        || !eligible(
            descriptor,
            request,
            constraints,
            input_layout,
            output_layout,
            workspace.len(),
            Some(descriptor.processor),
        )
    {
        output.fill(0);
        return Err(ContractError::InvalidSource);
    }
    let grant = ProviderDispatchGrant {
        provider_id: descriptor.provider_id,
        generation: descriptor.generation,
        operation: descriptor.operation,
    };
    let report = match provider.dispatch(
        grant,
        input_layout,
        &input[..input_span as usize],
        output_layout,
        &mut output[..output_span as usize],
        workspace,
    ) {
        Ok(report) => report,
        Err(error) => {
            output.fill(0);
            return Err(error);
        }
    };
    let current = provider.descriptor();
    if provider.lifecycle() != ProviderLifecycle::Ready
        || current != descriptor
        || report.provider_id != descriptor.provider_id
        || report.generation != descriptor.generation
        || report.operation != request.operation()
        || report.bytes_read != input_span
        || report.bytes_written != output_span
    {
        output.fill(0);
        return Err(ContractError::InvalidSource);
    }
    Ok(LiveDispatchReceipt {
        provider_id: descriptor.provider_id,
        generation: descriptor.generation,
        processor: descriptor.processor,
        operation: descriptor.operation,
        bytes_read: report.bytes_read,
        bytes_written: report.bytes_written,
        allocation_class: descriptor.allocation_class,
    })
}

fn find_provider(
    processor: Option<ProcessorKind>,
    request: ProcessingRequest,
    constraints: DispatchConstraints,
    providers: &[&mut dyn LiveProcessingProvider],
    input: &SampleLayout,
    output: &SampleLayout,
    workspace_len: usize,
) -> Option<usize> {
    providers.iter().position(|provider| {
        let descriptor = provider.descriptor();
        provider.lifecycle() == ProviderLifecycle::Ready
            && eligible(
                descriptor,
                request,
                constraints,
                input,
                output,
                workspace_len,
                processor,
            )
    })
}

fn eligible(
    d: LiveProviderDescriptor,
    request: ProcessingRequest,
    constraints: DispatchConstraints,
    input: &SampleLayout,
    output: &SampleLayout,
    workspace_len: usize,
    processor: Option<ProcessorKind>,
) -> bool {
    if d.provider_id == 0
        || d.generation == 0
        || d.operation != request.operation()
        || processor.is_some_and(|kind| d.processor != kind)
        || !request.preference().permits(d.processor)
        || d.allocation_class != constraints.allocation_class
        || d.workspace_bytes > constraints.max_workspace_bytes
        || d.workspace_bytes > request.max_workspace_bytes()
        || d.workspace_bytes > workspace_len as u64
    {
        return false;
    }
    if let Some(budget) = request.latency_budget_ns() {
        if !matches!(d.latency_bound_ns, Some(bound) if bound <= budget) {
            return false;
        }
    }
    layout_supported(input, d.input_encoding, d.max_rank, d.max_extents)
        && layout_supported(output, d.output_encoding, d.max_rank, d.max_extents)
}

fn layout_supported(
    layout: &SampleLayout,
    encoding: SampleEncoding,
    max_rank: u8,
    max_extents: [u32; 4],
) -> bool {
    if *layout.encoding() != encoding || layout.rank() > max_rank {
        return false;
    }
    (0..usize::from(layout.rank())).all(|axis| layout.extents()[axis] <= max_extents[axis])
}

#[cfg(test)]
#[path = "processing_runtime_tests.rs"]
mod tests;
