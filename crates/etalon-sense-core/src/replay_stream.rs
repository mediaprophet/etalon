//! Bounded deterministic replay over caller-owned frame bytes and descriptors.
//!
//! This is an offline data source, not a live device adapter. It retains no
//! frame payloads and allocates nothing. Every output frame still requires the
//! same exact-source/capability authorization buffer and trusted clock checks
//! as an extended adapter.

use super::{ReplayAuthorizationClock, ReplayMetadata};
use crate::{
    adapter::{
        AdapterError, AdapterId, AuthorizedExtendedCapture, AuthorizedExtendedCaptureBuffer,
        ExtendedAdapterDescriptor, ExtendedCaptureSession, ExtendedCapturedSamples,
        ExtendedDiscoveredSource, ExtendedSensorAdapter,
    },
    buffer::SampleLayout,
    extended_capability::ExtendedCapability,
    quality::{CalibrationRecord, QualityRecord},
    quantity::SampleRate,
    stream_order::{SequenceEvent, StreamOrder},
    time::ClockStamp,
    ContractError,
};

/// Caller-selected bounds for a replay stream. Bounds are checked before any
/// session can be opened; this type does not allocate its frame table or bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayStreamBudget {
    max_frames: usize,
    max_payload_bytes: usize,
}

impl ReplayStreamBudget {
    pub const fn new(max_frames: usize, max_payload_bytes: usize) -> Result<Self, ContractError> {
        if max_frames == 0 || max_payload_bytes == 0 {
            return Err(ContractError::InvalidExtent);
        }
        Ok(Self {
            max_frames,
            max_payload_bytes,
        })
    }

    pub const fn max_frames(self) -> usize {
        self.max_frames
    }
    pub const fn max_payload_bytes(self) -> usize {
        self.max_payload_bytes
    }
}

/// One frame's position and caller-declared measurement metadata.
/// `offset_bytes` addresses the supplied replay payload; it does not establish
/// provenance, acquisition, calibration truth, or a wall-clock timestamp.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayFrameDescriptor {
    offset_bytes: u64,
    layout: SampleLayout,
    timestamp: ClockStamp,
    sequence: u64,
    sample_rate: Option<SampleRate>,
    calibration: CalibrationRecord,
    quality: QualityRecord,
}

impl ReplayFrameDescriptor {
    pub const fn new(
        offset_bytes: u64,
        layout: SampleLayout,
        timestamp: ClockStamp,
        sequence: u64,
        sample_rate: Option<SampleRate>,
        calibration: CalibrationRecord,
        quality: QualityRecord,
    ) -> Self {
        Self {
            offset_bytes,
            layout,
            timestamp,
            sequence,
            sample_rate,
            calibration,
            quality,
        }
    }

    pub const fn offset_bytes(&self) -> u64 {
        self.offset_bytes
    }
    pub const fn layout(&self) -> &SampleLayout {
        &self.layout
    }
    pub const fn timestamp(&self) -> ClockStamp {
        self.timestamp
    }
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }
    pub const fn sample_rate(&self) -> Option<SampleRate> {
        self.sample_rate
    }
    pub const fn calibration(&self) -> CalibrationRecord {
        self.calibration
    }
    pub const fn quality(&self) -> QualityRecord {
        self.quality
    }
}

/// Metadata returned alongside one successfully copied replay frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplayFrameInfo {
    timestamp: ClockStamp,
    sequence: u64,
    sequence_event: SequenceEvent,
    sample_rate: Option<SampleRate>,
    calibration: CalibrationRecord,
    quality: QualityRecord,
}

impl ReplayFrameInfo {
    pub const fn timestamp(self) -> ClockStamp {
        self.timestamp
    }
    pub const fn sequence(self) -> u64 {
        self.sequence
    }
    pub const fn sequence_event(self) -> SequenceEvent {
        self.sequence_event
    }
    pub const fn sample_rate(self) -> Option<SampleRate> {
        self.sample_rate
    }
    pub const fn calibration(self) -> CalibrationRecord {
        self.calibration
    }
    pub const fn quality(self) -> QualityRecord {
        self.quality
    }
}

/// Frame data and its caller-declared timing/quality metadata, returned only
/// after authorization, validation, and the caller-buffer copy succeeded.
#[derive(Debug, PartialEq)]
pub struct ReplayCapturedFrame<'a> {
    source: ExtendedDiscoveredSource,
    samples: ExtendedCapturedSamples<'a>,
    info: ReplayFrameInfo,
    replay_metadata: ReplayMetadata,
}

impl<'a> ReplayCapturedFrame<'a> {
    pub const fn source(&self) -> ExtendedDiscoveredSource {
        self.source
    }
    pub const fn samples(&self) -> &ExtendedCapturedSamples<'a> {
        &self.samples
    }
    pub const fn info(&self) -> ReplayFrameInfo {
        self.info
    }
    pub const fn replay_metadata(&self) -> ReplayMetadata {
        self.replay_metadata
    }
}

/// Multi-frame, caller-buffered deterministic replay adapter for one extended
/// capability. The stream is prerecorded or synthetic by `ReplayMetadata`;
/// this type has no live mode and cannot issue a capture permit.
pub struct ReplayStreamAdapter<'bytes, 'frames, 'clock> {
    descriptor: ExtendedAdapterDescriptor,
    source: ExtendedDiscoveredSource,
    payload: &'bytes [u8],
    frames: &'frames [ReplayFrameDescriptor],
    metadata: ReplayMetadata,
    clock: &'clock mut dyn ReplayAuthorizationClock,
    stream_id: u64,
    generation: u32,
    order: StreamOrder,
    frame_index: usize,
    active: Option<ExtendedCaptureSession>,
    next_session_id: u64,
    last_frame_info: Option<ReplayFrameInfo>,
}

impl<'bytes, 'frames, 'clock> ReplayStreamAdapter<'bytes, 'frames, 'clock> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        adapter_id: AdapterId,
        source_id: u64,
        capability: ExtendedCapability,
        stream_id: u64,
        generation: u32,
        payload: &'bytes [u8],
        frames: &'frames [ReplayFrameDescriptor],
        metadata: ReplayMetadata,
        budget: ReplayStreamBudget,
        clock: &'clock mut dyn ReplayAuthorizationClock,
    ) -> Result<Self, AdapterError> {
        if frames.is_empty() || frames.len() > budget.max_frames {
            return Err(AdapterError::Buffer(ContractError::InvalidExtent));
        }
        if payload.is_empty() || payload.len() > budget.max_payload_bytes {
            return Err(AdapterError::Buffer(ContractError::OutOfBounds));
        }
        let source = ExtendedDiscoveredSource::new(source_id, capability)?;
        let descriptor = ExtendedAdapterDescriptor::new(adapter_id, capability.modality_id())?;
        let mut validation_order = StreamOrder::new(stream_id, generation)?;
        for frame in frames {
            capability.validate_layout(&frame.layout, frame.sample_rate)?;
            let start = usize::try_from(frame.offset_bytes)
                .map_err(|_| AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
            let span = usize::try_from(frame.layout.required_byte_span()?)
                .map_err(|_| AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
            let end = start
                .checked_add(span)
                .ok_or(AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
            if end > payload.len() {
                return Err(AdapterError::Buffer(ContractError::OutOfBounds));
            }
            validation_order.observe(stream_id, generation, frame.sequence, frame.timestamp)?;
        }
        Ok(Self {
            descriptor,
            source,
            payload,
            frames,
            metadata,
            clock,
            stream_id,
            generation,
            order: StreamOrder::new(stream_id, generation)?,
            frame_index: 0,
            active: None,
            next_session_id: 1,
            last_frame_info: None,
        })
    }

    pub const fn metadata(&self) -> ReplayMetadata {
        self.metadata
    }
    pub const fn source(&self) -> ExtendedDiscoveredSource {
        self.source
    }
    pub const fn stream_id(&self) -> u64 {
        self.stream_id
    }
    pub const fn generation(&self) -> u32 {
        self.generation
    }
    pub const fn next_frame_index(&self) -> usize {
        self.frame_index
    }
    pub const fn is_exhausted(&self) -> bool {
        self.frame_index >= self.frames.len()
    }

    /// Authorized frame-level capture. Each frame consumes one fresh
    /// `AuthorizedExtendedCaptureBuffer`; the output includes recorded time,
    /// sequence and gap metadata so downstream ingest can preserve them.
    pub fn capture_frame<'out>(
        &mut self,
        output: AuthorizedExtendedCaptureBuffer<'out>,
    ) -> Result<ReplayCapturedFrame<'out>, AdapterError> {
        self.last_frame_info = None;
        let samples = <Self as ExtendedSensorAdapter>::capture(self, output)?;
        let info = self
            .last_frame_info
            .take()
            .ok_or(AdapterError::BackendFailure)?;
        Ok(ReplayCapturedFrame {
            source: self.source,
            samples,
            info,
            replay_metadata: self.metadata,
        })
    }

    fn frame_slice(&self, frame: &ReplayFrameDescriptor) -> Result<&'bytes [u8], AdapterError> {
        let payload: &'bytes [u8] = self.payload;
        let start = usize::try_from(frame.offset_bytes)
            .map_err(|_| AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
        let span = usize::try_from(frame.layout.required_byte_span()?)
            .map_err(|_| AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
        let end = start
            .checked_add(span)
            .ok_or(AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
        payload
            .get(start..end)
            .ok_or(AdapterError::Buffer(ContractError::OutOfBounds))
    }
}

impl ExtendedSensorAdapter for ReplayStreamAdapter<'_, '_, '_> {
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
        out[0] = self.source;
        Ok(1)
    }

    fn start_backend(
        &mut self,
        request: AuthorizedExtendedCapture,
    ) -> Result<ExtendedCaptureSession, AdapterError> {
        if request.source() != self.source {
            return Err(AdapterError::AuthorizationDenied);
        }
        request.validate_at(self.clock.trusted_now()?)?;
        if self.active.is_some() || self.is_exhausted() {
            return Err(AdapterError::Unavailable);
        }
        let id = self.next_session_id;
        self.next_session_id = id.checked_add(1).ok_or(AdapterError::InvalidIdentifier)?;
        let session =
            ExtendedCaptureSession::new(self.descriptor.id(), self.source, id, self.generation)?;
        self.active = Some(session);
        Ok(session)
    }

    fn capture_backend<'out>(
        &mut self,
        mut output: AuthorizedExtendedCaptureBuffer<'out>,
    ) -> Result<ExtendedCapturedSamples<'out>, AdapterError> {
        self.last_frame_info = None;
        if self.active != Some(output.session())
            || output.source() != self.source
            || !output.session().matches(self.descriptor, self.source)
        {
            return Err(AdapterError::InvalidSession);
        }
        let frame = self
            .frames
            .get(self.frame_index)
            .ok_or(AdapterError::Unavailable)?;
        if output.layout() != frame.layout() {
            return Err(AdapterError::Buffer(ContractError::InvalidExtent));
        }
        output.validate_at(self.clock.trusted_now()?)?;
        let source_bytes = self.frame_slice(frame)?;
        let destination = output.writable_bytes();
        if destination.len() != source_bytes.len() {
            return Err(AdapterError::Buffer(ContractError::OutOfBounds));
        }
        let event = self.order.observe(
            self.stream_id,
            self.generation,
            frame.sequence,
            frame.timestamp,
        )?;
        destination.copy_from_slice(source_bytes);
        let info = ReplayFrameInfo {
            timestamp: frame.timestamp,
            sequence: frame.sequence,
            sequence_event: event,
            sample_rate: frame.sample_rate,
            calibration: frame.calibration,
            quality: frame.quality,
        };
        self.frame_index += 1;
        self.last_frame_info = Some(info);
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

#[cfg(test)]
#[path = "replay_stream_tests.rs"]
mod tests;
