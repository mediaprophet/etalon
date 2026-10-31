//! Caller-buffered replay of one immutable sample block.
//!
//! This adapter has no operating-system, network, or hardware path. It copies
//! only caller-supplied bytes and metadata and never issues capture authority.

use crate::{
    adapter::{
        AdapterDescriptor, AdapterError, AdapterId, AuthorizedCapture, AuthorizedCaptureBuffer,
        CaptureSession, CapturedSamples, DiscoveredSource, SensorAdapter,
    },
    buffer::SampleLayout,
    capability::SensorCapability,
    evidence::EvidenceRecord,
    time::ClockStamp,
    ContractError,
};

#[path = "extended_replay.rs"]
mod extended;
pub use extended::*;

#[path = "replay_stream.rs"]
mod streaming;
pub use streaming::*;

/// Explicit classification for the supplied data; neither variant is live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayKind {
    Synthetic,
    Prerecorded,
}

/// Caller-supplied descriptive metadata. Provenance and timestamp are copied
/// verbatim and not authenticated or mapped to the authorization clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayMetadata {
    kind: ReplayKind,
    evidence: EvidenceRecord,
    recorded_at: Option<ClockStamp>,
}

impl ReplayMetadata {
    pub const fn new(
        kind: ReplayKind,
        evidence: EvidenceRecord,
        recorded_at: Option<ClockStamp>,
    ) -> Self {
        Self {
            kind,
            evidence,
            recorded_at,
        }
    }

    pub const fn kind(self) -> ReplayKind {
        self.kind
    }
    pub const fn evidence(self) -> EvidenceRecord {
        self.evidence
    }
    pub const fn recorded_at(self) -> Option<ClockStamp> {
        self.recorded_at
    }
}

/// External trusted authorization-time source required by the adapter trait.
/// Replay does not establish the trustworthiness of this implementation.
pub trait ReplayAuthorizationClock {
    fn trusted_now(&mut self) -> Result<ClockStamp, AdapterError>;
}

/// A validated, immutable sample block. The byte slice must contain exactly
/// one layout span; padding bytes within that span are preserved on replay.
pub struct ReplayBlock<'a> {
    source: DiscoveredSource,
    layout: SampleLayout,
    bytes: &'a [u8],
    metadata: ReplayMetadata,
}

impl<'a> ReplayBlock<'a> {
    pub fn new(
        source_id: u64,
        capability: SensorCapability,
        layout: SampleLayout,
        bytes: &'a [u8],
        metadata: ReplayMetadata,
    ) -> Result<Self, AdapterError> {
        if layout.encoding() != &capability.encoding() {
            return Err(AdapterError::Buffer(ContractError::IncompatibleEncoding));
        }
        if layout.rank() != capability.max_rank() {
            return Err(AdapterError::Buffer(ContractError::InvalidExtent));
        }
        let extents = layout.extents();
        let limits = capability.max_extents();
        let mut element_count = 1u64;
        for axis in 0..usize::from(layout.rank()) {
            if extents[axis] > limits[axis] {
                return Err(AdapterError::Buffer(ContractError::InvalidExtent));
            }
            element_count = element_count
                .checked_mul(u64::from(extents[axis]))
                .ok_or(AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
        }
        let payload = element_count
            .checked_mul(u64::from(capability.encoding().element_bytes()))
            .ok_or(AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
        if payload > capability.max_payload_bytes() {
            return Err(AdapterError::Buffer(ContractError::OutOfBounds));
        }
        let span = layout.required_byte_span()?;
        let span = usize::try_from(span)
            .map_err(|_| AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
        if span != bytes.len() {
            return Err(AdapterError::Buffer(ContractError::OutOfBounds));
        }
        let source = DiscoveredSource::new(source_id, capability)?;
        Ok(Self {
            source,
            layout,
            bytes,
            metadata,
        })
    }

    pub const fn source(&self) -> DiscoveredSource {
        self.source
    }
    pub const fn layout(&self) -> &SampleLayout {
        &self.layout
    }
    pub const fn metadata(&self) -> ReplayMetadata {
        self.metadata
    }
}

/// In-memory one-shot replay source. Authorization is always supplied by the
/// existing crate authority through `AuthorizedCapture`; this type cannot mint it.
pub struct ReplayAdapter<'data, 'clock> {
    descriptor: AdapterDescriptor,
    block: ReplayBlock<'data>,
    clock: &'clock mut dyn ReplayAuthorizationClock,
    active: Option<CaptureSession>,
    next_session_id: u64,
    consumed: bool,
}

impl<'data, 'clock> ReplayAdapter<'data, 'clock> {
    pub fn new(
        adapter_id: AdapterId,
        block: ReplayBlock<'data>,
        clock: &'clock mut dyn ReplayAuthorizationClock,
    ) -> Self {
        let descriptor = AdapterDescriptor::new(adapter_id, block.source.capability().modality());
        Self {
            descriptor,
            block,
            clock,
            active: None,
            next_session_id: 1,
            consumed: false,
        }
    }

    pub const fn metadata(&self) -> ReplayMetadata {
        self.block.metadata
    }
    pub const fn is_consumed(&self) -> bool {
        self.consumed
    }
}

impl SensorAdapter for ReplayAdapter<'_, '_> {
    fn descriptor(&self) -> AdapterDescriptor {
        self.descriptor
    }

    fn trusted_now(&mut self) -> Result<ClockStamp, AdapterError> {
        self.clock.trusted_now()
    }

    fn discover_backend(&mut self, out: &mut [DiscoveredSource]) -> Result<usize, AdapterError> {
        if out.is_empty() {
            return Err(AdapterError::DiscoveryBufferTooSmall);
        }
        out[0] = self.block.source;
        Ok(1)
    }

    fn start_backend(
        &mut self,
        request: AuthorizedCapture,
    ) -> Result<CaptureSession, AdapterError> {
        if request.source() != self.block.source {
            return Err(AdapterError::AuthorizationDenied);
        }
        if self.active.is_some() {
            return Err(AdapterError::Unavailable);
        }
        if self.consumed {
            return Err(AdapterError::Unavailable);
        }
        let id = self.next_session_id;
        let next = id.checked_add(1).ok_or(AdapterError::InvalidIdentifier)?;
        let session = CaptureSession::new(self.descriptor.id(), self.block.source, id, 1)?;
        self.next_session_id = next;
        self.active = Some(session);
        Ok(session)
    }

    fn capture_backend<'a>(
        &mut self,
        mut output: AuthorizedCaptureBuffer<'a>,
    ) -> Result<CapturedSamples<'a>, AdapterError> {
        if self.active != Some(output.session())
            || output.source() != self.block.source
            || output.layout() != &self.block.layout
        {
            return Err(AdapterError::InvalidSession);
        }
        if self.consumed {
            return Err(AdapterError::Unavailable);
        }
        let dst = output.writable_bytes();
        if dst.len() != self.block.bytes.len() {
            return Err(AdapterError::Buffer(ContractError::OutOfBounds));
        }
        dst.copy_from_slice(self.block.bytes);
        self.consumed = true;
        Ok(output.complete())
    }

    fn stop_backend(&mut self, session: CaptureSession) -> Result<(), AdapterError> {
        if self.active != Some(session) {
            return Err(AdapterError::InvalidSession);
        }
        self.active = None;
        Ok(())
    }
}

#[cfg(test)]
#[path = "replay_tests.rs"]
mod tests;
