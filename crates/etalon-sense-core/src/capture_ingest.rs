//! One-shot capture-to-ingest composition for built-in and host-registered adapters.
//!
//! The bridge requires the existing crate-issued authorized capture request and
//! goes through the adapter's authorization recheck. It preflights metadata and
//! stream ordering before device I/O, then admits exactly the returned byte span
//! into the matching ingest tracker. All source, time, quality, calibration,
//! buffer-token, and sequence values remain declarations supplied by the host;
//! this bridge creates no hardware, identity, provenance, or policy receipt.

use crate::{
    adapter::{
        AdapterError, AuthorizedCapture, AuthorizedCaptureBuffer, AuthorizedExtendedCapture,
        AuthorizedExtendedCaptureBuffer, CaptureBuffer, CaptureSession, CapturedSamples,
        ExtendedCaptureBuffer, ExtendedCaptureSession, ExtendedCapturedSamples,
        ExtendedSensorAdapter, SensorAdapter,
    },
    buffer::{SampleLayout, SampleRegion},
    capability::SensorCapability,
    extended_capability::ExtendedCapability,
    ingest::{ExtendedIngestStream, ExtendedIngestedFrame, IngestStream, IngestedFrame},
    measurement::{ExtendedMeasurement, Measurement},
    quality::{CalibrationRecord, QualityRecord, SourceDescriptor},
    quantity::{SampleEncoding, SampleRate},
    time::ClockStamp,
    ContractError,
};

/// Caller declarations needed to admit one captured frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IngestMetadata {
    region: SampleRegion,
    source: SourceDescriptor,
    timestamp: Option<ClockStamp>,
    sample_rate: Option<SampleRate>,
    calibration: CalibrationRecord,
    quality: QualityRecord,
    sequence: u64,
}

impl IngestMetadata {
    pub const fn new(
        region: SampleRegion,
        source: SourceDescriptor,
        timestamp: ClockStamp,
        sample_rate: Option<SampleRate>,
        calibration: CalibrationRecord,
        quality: QualityRecord,
        sequence: u64,
    ) -> Self {
        Self::new_with_optional_timestamp(region, source, Some(timestamp), sample_rate, calibration, quality, sequence)
    }

    pub const fn new_with_optional_timestamp(
        region: SampleRegion,
        source: SourceDescriptor,
        timestamp: Option<ClockStamp>,
        sample_rate: Option<SampleRate>,
        calibration: CalibrationRecord,
        quality: QualityRecord,
        sequence: u64,
    ) -> Self {
        Self { region, source, timestamp, sample_rate, calibration, quality, sequence }
    }
}

/// Execute one built-in adapter capture and admit its bytes to one stream.
///
/// `session.generation()` must match the tracker generation. The stream ID is
/// independent of the adapter's capture-session ID; sequence ordering remains
/// scoped to the caller-created `IngestStream`.
pub fn capture_and_ingest<'a, A: SensorAdapter>(
    adapter: &mut A,
    session: CaptureSession,
    request: AuthorizedCapture,
    buffer: CaptureBuffer<'a>,
    stream: &mut IngestStream,
    metadata: IngestMetadata,
) -> Result<IngestedFrame<'a>, CaptureIngestError> {
    let source = session.source();
    check_binding(
        stream.source_id(),
        stream.generation(),
        source.source_id(),
        session.generation(),
        metadata.source,
    )?;
    let layout = buffer.layout().clone();
    let capability = source.capability();
    preflight_builtin(stream, layout.clone(), capability, metadata)?;

    let output = AuthorizedCaptureBuffer::new(adapter.descriptor(), session, request, buffer)?;
    let samples = adapter.capture(output)?;
    admit_builtin(stream, samples, layout, capability, metadata)
}

/// Execute one host-registered adapter capture and admit its bytes to an
/// extended stream. Registry IDs and unit semantics remain host assertions.
pub fn capture_extended_and_ingest<'a, A: ExtendedSensorAdapter>(
    adapter: &mut A,
    session: ExtendedCaptureSession,
    request: AuthorizedExtendedCapture,
    buffer: ExtendedCaptureBuffer<'a>,
    stream: &mut ExtendedIngestStream,
    metadata: IngestMetadata,
) -> Result<ExtendedIngestedFrame<'a>, CaptureIngestError> {
    let source = session.source();
    check_binding(
        stream.source_id(),
        stream.generation(),
        source.source_id(),
        session.generation(),
        metadata.source,
    )?;
    let layout = buffer.layout().clone();
    let capability = source.capability();
    preflight_extended(stream, layout.clone(), capability, metadata)?;

    let output = AuthorizedExtendedCaptureBuffer::new(
        adapter.descriptor(), session, request, buffer,
    )?;
    let samples = adapter.capture(output)?;
    admit_extended(stream, samples, layout, capability, metadata)
}

fn check_binding(
    stream_source: u64,
    stream_generation: u32,
    capture_source: u64,
    capture_generation: u32,
    metadata_source: SourceDescriptor,
) -> Result<(), CaptureIngestError> {
    if stream_source != capture_source || metadata_source.source_id() != capture_source {
        return Err(CaptureIngestError::SourceMismatch);
    }
    if stream_generation != capture_generation {
        return Err(CaptureIngestError::GenerationMismatch);
    }
    Ok(())
}

fn preflight_builtin(
    stream: &IngestStream,
    layout: SampleLayout,
    capability: SensorCapability,
    metadata: IngestMetadata,
) -> Result<(), CaptureIngestError> {
    metadata.region.validate_layout(&layout)?;
    Measurement::new_with_optional_timestamp(
        capability,
        layout,
        metadata.region,
        metadata.source,
        metadata.timestamp,
        metadata.sample_rate,
        metadata.calibration,
        metadata.quality,
    )?;
    preflight_sequence(stream.last_observation(), stream.last_known_timestamp(), metadata.sequence, metadata.timestamp)
}

fn preflight_extended(
    stream: &ExtendedIngestStream,
    layout: SampleLayout,
    capability: ExtendedCapability,
    metadata: IngestMetadata,
) -> Result<(), CaptureIngestError> {
    metadata.region.validate_layout(&layout)?;
    ExtendedMeasurement::new_with_optional_timestamp(
        capability,
        layout.clone(),
        metadata.region,
        metadata.source,
        metadata.timestamp,
        metadata.sample_rate,
        metadata.calibration,
        metadata.quality,
    )?;
    validate_payload_ceiling(&layout, capability.encoding(), capability.max_payload_bytes())?;
    preflight_sequence(stream.last_observation(), stream.last_known_timestamp(), metadata.sequence, metadata.timestamp)
}

fn validate_payload_ceiling(
    layout: &SampleLayout,
    encoding: SampleEncoding,
    max_payload_bytes: u64,
) -> Result<(), CaptureIngestError> {
    let mut elements = 1u64;
    for extent in layout.extents().iter().take(usize::from(layout.rank())) {
        elements = elements
            .checked_mul(u64::from(*extent))
            .ok_or(ContractError::ArithmeticOverflow)?;
    }
    let packed_bytes = elements
        .checked_mul(u64::from(encoding.element_bytes()))
        .ok_or(ContractError::ArithmeticOverflow)?;
    if packed_bytes > max_payload_bytes {
        return Err(ContractError::OutOfBounds.into());
    }
    Ok(())
}

fn preflight_sequence(
    last: Option<(u64, Option<ClockStamp>)>,
    last_known_timestamp: Option<ClockStamp>,
    sequence: u64,
    timestamp: Option<ClockStamp>,
) -> Result<(), CaptureIngestError> {
    if let Some((previous_sequence, _)) = last {
        let delta = sequence
            .checked_sub(previous_sequence)
            .ok_or(ContractError::InvalidSource)?;
        if delta == 0 {
            return Err(ContractError::InvalidSource.into());
        }
    }
    if let (Some(timestamp), Some(previous_timestamp)) = (timestamp, last_known_timestamp) {
        if timestamp.clock_id() != previous_timestamp.clock_id()
            || timestamp.restart_epoch() != previous_timestamp.restart_epoch()
        {
            return Err(ContractError::ClockDomainMismatch.into());
        }
        if timestamp.ticks_ns() < previous_timestamp.ticks_ns() {
            return Err(ContractError::InvalidTime.into());
        }
    }
    Ok(())
}

fn admit_builtin<'a>(
    stream: &mut IngestStream,
    samples: CapturedSamples<'a>,
    expected_layout: SampleLayout,
    capability: SensorCapability,
    metadata: IngestMetadata,
) -> Result<IngestedFrame<'a>, CaptureIngestError> {
    if samples.layout() != &expected_layout
        || samples.bytes().len() != expected_span(&expected_layout)?
    {
        return Err(CaptureIngestError::BackendFrameMismatch);
    }
    Ok(stream.ingest_with_optional_timestamp(
        samples.bytes(),
        expected_layout,
        metadata.region,
        metadata.source,
        capability,
        metadata.timestamp,
        metadata.sample_rate,
        metadata.calibration,
        metadata.quality,
        metadata.sequence,
    )?)
}

fn admit_extended<'a>(
    stream: &mut ExtendedIngestStream,
    samples: ExtendedCapturedSamples<'a>,
    expected_layout: SampleLayout,
    capability: ExtendedCapability,
    metadata: IngestMetadata,
) -> Result<ExtendedIngestedFrame<'a>, CaptureIngestError> {
    if samples.layout() != &expected_layout
        || samples.bytes().len() != expected_span(&expected_layout)?
    {
        return Err(CaptureIngestError::BackendFrameMismatch);
    }
    Ok(stream.ingest_with_optional_timestamp(
        samples.bytes(),
        expected_layout,
        metadata.region,
        metadata.source,
        metadata.timestamp,
        metadata.sample_rate,
        metadata.calibration,
        metadata.quality,
        metadata.sequence,
    )?)
}

fn expected_span(layout: &SampleLayout) -> Result<usize, ContractError> {
    usize::try_from(layout.required_byte_span()?).map_err(|_| ContractError::ArithmeticOverflow)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureIngestError {
    Adapter(AdapterError),
    Contract(ContractError),
    SourceMismatch,
    GenerationMismatch,
    BackendFrameMismatch,
}

impl From<AdapterError> for CaptureIngestError {
    fn from(error: AdapterError) -> Self { Self::Adapter(error) }
}

impl From<ContractError> for CaptureIngestError {
    fn from(error: ContractError) -> Self { Self::Contract(error) }
}

#[cfg(test)]
#[path = "capture_ingest_tests.rs"]
mod tests;
