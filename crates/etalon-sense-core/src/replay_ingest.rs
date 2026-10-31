//! Replay-specific capture-to-ingest bridge.
//!
//! Frame time, sequence, rate, calibration, and quality come from the selected
//! replay descriptor. The frame's replay classification and evidence reference
//! remain caller declarations; this module performs no provenance verification.

use crate::{
    adapter::{
        AdapterError, AuthorizedExtendedCapture, AuthorizedExtendedCaptureBuffer,
        ExtendedCaptureBuffer, ExtendedCaptureSession, ExtendedSensorAdapter,
    },
    buffer::SampleRegion,
    ingest::{ExtendedIngestStream, ExtendedIngestedFrame},
    quality::{SourceDescriptor, SourceKind},
    replay::{ReplayCapturedFrame, ReplayFrameInfo, ReplayMetadata, ReplayStreamAdapter},
    stream_order::SequenceEvent,
    ContractError,
};

/// One replay frame admitted with the metadata from its own descriptor.
#[derive(Debug, PartialEq)]
pub struct ReplayIngestedFrame<'a> {
    ingested: ExtendedIngestedFrame<'a>,
    replay_info: ReplayFrameInfo,
    replay_metadata: ReplayMetadata,
}

impl<'a> ReplayIngestedFrame<'a> {
    pub const fn ingested(&self) -> &ExtendedIngestedFrame<'a> { &self.ingested }
    pub const fn replay_info(&self) -> ReplayFrameInfo { self.replay_info }
    pub const fn replay_metadata(&self) -> ReplayMetadata { self.replay_metadata }
}

/// Failure classification. Post-capture failures retain the frame so a caller
/// can correct the ingest stream or region and retry admission without losing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayIngestErrorKind {
    Adapter(AdapterError),
    Contract(ContractError),
    SourceMismatch,
    StreamMismatch,
    BackendFrameMismatch,
    ReplayOrderMismatch,
}

impl From<AdapterError> for ReplayIngestErrorKind {
    fn from(error: AdapterError) -> Self { Self::Adapter(error) }
}

impl From<ContractError> for ReplayIngestErrorKind {
    fn from(error: ContractError) -> Self { Self::Contract(error) }
}

/// A pre-capture error has no consumed replay frame. A post-capture error
/// returns the still-owned frame; ingest validates before updating its tracker.
#[derive(Debug, PartialEq)]
pub enum ReplayIngestFailure<'a> {
    BeforeCapture(ReplayIngestErrorKind),
    Captured {
        error: ReplayIngestErrorKind,
        frame: ReplayCapturedFrame<'a>,
    },
}

/// Capture and ingest exactly the next replay frame.
///
/// The replay adapter and ingest stream must have the same caller-selected
/// stream ID and generation. Their source and capability declarations must also
/// match. Region identity remains a caller declaration and sample provenance is
/// always reported as unverified by `SourceDescriptor`.
pub fn capture_replay_and_ingest<'out>(
    adapter: &mut ReplayStreamAdapter<'_, '_, '_>,
    session: ExtendedCaptureSession,
    request: AuthorizedExtendedCapture,
    buffer: ExtendedCaptureBuffer<'out>,
    stream: &mut ExtendedIngestStream,
    region: SampleRegion,
) -> Result<ReplayIngestedFrame<'out>, ReplayIngestFailure<'out>> {
    let source = adapter.source();
    let capability = source.capability();
    if stream.source_id() != source.source_id() {
        return Err(ReplayIngestFailure::BeforeCapture(ReplayIngestErrorKind::SourceMismatch));
    }
    if stream.stream_id() != adapter.stream_id()
        || stream.generation() != adapter.generation()
        || stream.capability() != capability
    {
        return Err(ReplayIngestFailure::BeforeCapture(ReplayIngestErrorKind::StreamMismatch));
    }
    if buffer.capability() != capability {
        return Err(ReplayIngestFailure::BeforeCapture(ReplayIngestErrorKind::BackendFrameMismatch));
    }
    let layout = buffer.layout().clone();
    if let Err(error) = capability.validate_layout(&layout, None)
        .and_then(|()| region.validate_layout(&layout))
    {
        return Err(ReplayIngestFailure::BeforeCapture(error.into()));
    }

    let output = match AuthorizedExtendedCaptureBuffer::new(
        adapter.descriptor(), session, request, buffer,
    ) {
        Ok(output) => output,
        Err(error) => return Err(ReplayIngestFailure::BeforeCapture(error.into())),
    };
    let frame = match adapter.capture_frame(output) {
        Ok(frame) => frame,
        Err(error) => return Err(ReplayIngestFailure::BeforeCapture(error.into())),
    };

    let info = frame.info();
    let replay_metadata = frame.replay_metadata();
    let expected_source_kind = match replay_metadata.kind() {
        crate::replay::ReplayKind::Synthetic => SourceKind::Synthetic,
        crate::replay::ReplayKind::Prerecorded => SourceKind::Recorded,
    };
    let source_descriptor = match SourceDescriptor::new(source.source_id(), expected_source_kind) {
        Ok(source) => source,
        Err(error) => return Err(captured(error.into(), frame)),
    };
    let expected_bytes = match layout.required_byte_span()
        .and_then(|span| usize::try_from(span).map_err(|_| ContractError::ArithmeticOverflow))
    {
        Ok(bytes) => bytes,
        Err(error) => return Err(captured(error.into(), frame)),
    };
    if frame.source() != source
        || frame.samples().layout() != &layout
        || frame.samples().bytes().len() != expected_bytes
    {
        return Err(captured(ReplayIngestErrorKind::BackendFrameMismatch, frame));
    }

    let expected_event = match expected_event(
        stream.last_observation(),
        stream.last_known_timestamp(),
        info.sequence(),
        info.timestamp(),
    ) {
        Ok(event) => event,
        Err(error) => return Err(captured(error.into(), frame)),
    };
    if expected_event != info.sequence_event() {
        return Err(captured(ReplayIngestErrorKind::ReplayOrderMismatch, frame));
    }

    match stream.ingest(
        frame.samples().bytes(),
        layout,
        region,
        source_descriptor,
        info.timestamp(),
        info.sample_rate(),
        info.calibration(),
        info.quality(),
        info.sequence(),
    ) {
        Ok(ingested) => Ok(ReplayIngestedFrame { ingested, replay_info: info, replay_metadata }),
        Err(error) => Err(captured(error.into(), frame)),
    }
}

fn captured<'a>(error: ReplayIngestErrorKind, frame: ReplayCapturedFrame<'a>) -> ReplayIngestFailure<'a> {
    ReplayIngestFailure::Captured { error, frame }
}

fn expected_event(
    last: Option<(u64, Option<crate::time::ClockStamp>)>,
    last_known_timestamp: Option<crate::time::ClockStamp>,
    sequence: u64,
    timestamp: crate::time::ClockStamp,
) -> Result<SequenceEvent, ContractError> {
    let Some((previous_sequence, _)) = last else {
        return Ok(SequenceEvent::First);
    };
    let delta = sequence.checked_sub(previous_sequence).ok_or(ContractError::InvalidSource)?;
    if delta == 0 { return Err(ContractError::InvalidSource); }
    if let Some(previous_timestamp) = last_known_timestamp {
        if timestamp.clock_id() != previous_timestamp.clock_id()
            || timestamp.restart_epoch() != previous_timestamp.restart_epoch()
        {
            return Err(ContractError::ClockDomainMismatch);
        }
        if timestamp.ticks_ns() < previous_timestamp.ticks_ns() {
            return Err(ContractError::InvalidTime);
        }
    }
    Ok(if delta == 1 { SequenceEvent::Contiguous } else { SequenceEvent::Gap { missing: delta - 1 } })
}

#[cfg(test)]
#[path = "replay_ingest_tests.rs"]
mod tests;
