//! In-memory replay adapter for host-defined sensor capabilities. Replay is
//! restricted by construction to synthetic or prerecorded metadata and cannot
//! mint an authorization permit.

use super::{ReplayAuthorizationClock, ReplayMetadata};
use crate::{
    adapter::{
        AdapterError, AdapterId, AuthorizedExtendedCapture, AuthorizedExtendedCaptureBuffer,
        ExtendedAdapterDescriptor, ExtendedCaptureSession, ExtendedCapturedSamples,
        ExtendedDiscoveredSource, ExtendedSensorAdapter,
    },
    buffer::SampleLayout,
    extended_capability::ExtendedCapability,
    time::ClockStamp,
    ContractError,
};

/// Validated immutable extended-capability block. The bytes and metadata are
/// caller supplied; replay verifies shape and classification, not provenance.
pub struct ExtendedReplayBlock<'a> {
    source: ExtendedDiscoveredSource,
    layout: SampleLayout,
    bytes: &'a [u8],
    metadata: ReplayMetadata,
}

impl<'a> ExtendedReplayBlock<'a> {
    pub fn new(
        source_id: u64,
        capability: ExtendedCapability,
        layout: SampleLayout,
        bytes: &'a [u8],
        metadata: ReplayMetadata,
    ) -> Result<Self, AdapterError> {
        capability.validate_layout(&layout, None)?;
        let span = usize::try_from(layout.required_byte_span()?)
            .map_err(|_| AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
        if span != bytes.len() {
            return Err(AdapterError::Buffer(ContractError::OutOfBounds));
        }
        let source = ExtendedDiscoveredSource::new(source_id, capability)?;
        Ok(Self {
            source,
            layout,
            bytes,
            metadata,
        })
    }

    pub const fn source(&self) -> ExtendedDiscoveredSource {
        self.source
    }
    pub const fn layout(&self) -> &SampleLayout {
        &self.layout
    }
    pub const fn metadata(&self) -> ReplayMetadata {
        self.metadata
    }
    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

/// One-shot in-memory replay using the same per-source authorization and
/// caller-buffer lifecycle as other extended adapters.
pub struct ExtendedReplayAdapter<'data, 'clock> {
    descriptor: ExtendedAdapterDescriptor,
    block: ExtendedReplayBlock<'data>,
    clock: &'clock mut dyn ReplayAuthorizationClock,
    active: Option<ExtendedCaptureSession>,
    next_session_id: u64,
    consumed: bool,
}

impl<'data, 'clock> ExtendedReplayAdapter<'data, 'clock> {
    pub fn new(
        adapter_id: AdapterId,
        block: ExtendedReplayBlock<'data>,
        clock: &'clock mut dyn ReplayAuthorizationClock,
    ) -> Result<Self, AdapterError> {
        let descriptor =
            ExtendedAdapterDescriptor::new(adapter_id, block.source.capability().modality_id())?;
        Ok(Self {
            descriptor,
            block,
            clock,
            active: None,
            next_session_id: 1,
            consumed: false,
        })
    }

    pub const fn metadata(&self) -> ReplayMetadata {
        self.block.metadata
    }
    pub const fn is_consumed(&self) -> bool {
        self.consumed
    }
    pub const fn source(&self) -> ExtendedDiscoveredSource {
        self.block.source
    }
}

impl ExtendedSensorAdapter for ExtendedReplayAdapter<'_, '_> {
    fn descriptor(&self) -> ExtendedAdapterDescriptor {
        self.descriptor
    }

    fn trusted_now(&mut self) -> Result<ClockStamp, AdapterError> {
        self.clock.trusted_now()
    }

    fn discover_backend(
        &mut self,
        out: &mut [ExtendedDiscoveredSource],
    ) -> Result<usize, AdapterError> {
        if out.is_empty() {
            return Err(AdapterError::DiscoveryBufferTooSmall);
        }
        out[0] = self.block.source;
        Ok(1)
    }

    fn start_backend(
        &mut self,
        request: AuthorizedExtendedCapture,
    ) -> Result<ExtendedCaptureSession, AdapterError> {
        if request.source() != self.block.source {
            return Err(AdapterError::AuthorizationDenied);
        }
        if self.active.is_some() || self.consumed {
            return Err(AdapterError::Unavailable);
        }
        let id = self.next_session_id;
        let next = id.checked_add(1).ok_or(AdapterError::InvalidIdentifier)?;
        let session = ExtendedCaptureSession::new(self.descriptor.id(), self.block.source, id, 1)?;
        self.next_session_id = next;
        self.active = Some(session);
        Ok(session)
    }

    fn capture_backend<'a>(
        &mut self,
        mut output: AuthorizedExtendedCaptureBuffer<'a>,
    ) -> Result<ExtendedCapturedSamples<'a>, AdapterError> {
        if self.active != Some(output.session())
            || output.source() != self.block.source
            || output.layout() != &self.block.layout
        {
            return Err(AdapterError::InvalidSession);
        }
        if self.consumed {
            return Err(AdapterError::Unavailable);
        }
        let destination = output.writable_bytes();
        if destination.len() != self.block.bytes.len() {
            return Err(AdapterError::Buffer(ContractError::OutOfBounds));
        }
        destination.copy_from_slice(self.block.bytes);
        self.consumed = true;
        Ok(output.complete())
    }

    fn stop_backend(&mut self, session: ExtendedCaptureSession) -> Result<(), AdapterError> {
        if self.active != Some(session) {
            return Err(AdapterError::InvalidSession);
        }
        self.active = None;
        Ok(())
    }
}
