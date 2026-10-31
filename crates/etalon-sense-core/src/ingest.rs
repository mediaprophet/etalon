//! Bounded bridge from captured or replayed bytes to declared measurements.
//!
//! This module validates caller metadata, payload extent, source identity, and
//! stream sequence. It does not inspect or authenticate sample contents.
//! Pairing the borrowed payload with a `SampleRegion` is a caller declaration;
//! a `BufferToken` does not prove that these bytes came from that live allocation
//! or belong to the claimed source. Unknown rate, calibration, and quality
//! remain explicit.

use crate::buffer::{SampleLayout, SampleRegion};
use crate::capability::SensorCapability;
use crate::extended_capability::ExtendedCapability;
use crate::measurement::{ExtendedMeasurement, Measurement};
use crate::quality::{CalibrationRecord, QualityRecord, SourceDescriptor};
use crate::quantity::SampleRate;
use crate::stream_order::{SequenceEvent, StreamOrder};
use crate::time::ClockStamp;
use crate::ContractError;

/// Stateful admission boundary for one declared source and stream generation.
/// The tracker records ordering only; it does not authenticate identifiers.
#[derive(Debug)]
pub struct IngestStream {
    source_id: u64,
    order: StreamOrder,
}

/// Stateful admission boundary for one stream described by host-defined
/// modality/quantity IDs. The IDs and mapping to `source_id` remain caller or
/// registry declarations; this tracker does not authenticate them.
#[derive(Debug)]
pub struct ExtendedIngestStream {
    source_id: u64,
    capability: ExtendedCapability,
    order: StreamOrder,
}

impl IngestStream {
    pub fn new(
        source_id: u64,
        stream_id: u64,
        generation: u32,
    ) -> Result<Self, ContractError> {
        if source_id == 0 {
            return Err(ContractError::InvalidIdentifier);
        }
        Ok(Self {
            source_id,
            order: StreamOrder::new(stream_id, generation)?,
        })
    }

    /// Validate and admit exactly one byte view.
    ///
    /// `payload` must cover exactly the layout's byte span beginning at the
    /// sample region's declared offset. It is not the entire backing allocation.
    /// All measurement and byte checks happen before sequence state advances.
    /// A reported gap is returned with this frame and is never silently reset or
    /// bridged; temporal filters must apply their own explicit gap policy.
    #[allow(clippy::too_many_arguments)]
    pub fn ingest<'a>(
        &mut self,
        payload: &'a [u8],
        layout: SampleLayout,
        region: SampleRegion,
        source: SourceDescriptor,
        capability: SensorCapability,
        timestamp: ClockStamp,
        sample_rate: Option<SampleRate>,
        calibration: CalibrationRecord,
        quality: QualityRecord,
        sequence: u64,
    ) -> Result<IngestedFrame<'a>, ContractError> {
        self.ingest_with_optional_timestamp(
            payload, layout, region, source, capability, Some(timestamp), sample_rate,
            calibration, quality, sequence,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ingest_with_optional_timestamp<'a>(
        &mut self,
        payload: &'a [u8],
        layout: SampleLayout,
        region: SampleRegion,
        source: SourceDescriptor,
        capability: SensorCapability,
        timestamp: Option<ClockStamp>,
        sample_rate: Option<SampleRate>,
        calibration: CalibrationRecord,
        quality: QualityRecord,
        sequence: u64,
    ) -> Result<IngestedFrame<'a>, ContractError> {
        if source.source_id() != self.source_id {
            return Err(ContractError::InvalidSource);
        }
        let expected_bytes = usize::try_from(layout.required_byte_span()?)
            .map_err(|_| ContractError::ArithmeticOverflow)?;
        if payload.len() != expected_bytes {
            return Err(ContractError::OutOfBounds);
        }
        let measurement = Measurement::new_with_optional_timestamp(
            capability,
            layout,
            region,
            source,
            timestamp,
            sample_rate,
            calibration,
            quality,
        )?;
        let stream_id = self.order.stream_id();
        let generation = self.order.generation();
        let event = self
            .order
            .observe_optional(stream_id, generation, sequence, timestamp)?;
        Ok(IngestedFrame {
            payload,
            measurement,
            sequence_event: event,
            sequence,
            stream_id,
            generation,
        })
    }

    pub const fn source_id(&self) -> u64 {
        self.source_id
    }

    pub const fn stream_id(&self) -> u64 {
        self.order.stream_id()
    }

    pub const fn generation(&self) -> u32 {
        self.order.generation()
    }

    /// Latest sequence paired with a known timestamp. Use `last_observation`
    /// for sequence preflight when the latest timestamp may be unknown.
    pub const fn last_sequence(&self) -> Option<(u64, ClockStamp)> {
        self.order.last()
    }

    pub const fn last_observation(&self) -> Option<(u64, Option<ClockStamp>)> {
        self.order.last_observation()
    }

    pub const fn last_known_timestamp(&self) -> Option<ClockStamp> {
        self.order.last_known_timestamp()
    }
}

impl ExtendedIngestStream {
    pub fn new(
        source_id: u64,
        stream_id: u64,
        generation: u32,
        capability: ExtendedCapability,
    ) -> Result<Self, ContractError> {
        if source_id == 0 {
            return Err(ContractError::InvalidIdentifier);
        }
        Ok(Self { source_id, capability, order: StreamOrder::new(stream_id, generation)? })
    }

    /// Validate a byte view against an extended capability and admit one
    /// strictly ordered observation. The payload/region association remains a
    /// caller declaration and does not prove buffer provenance or hardware use.
    #[allow(clippy::too_many_arguments)]
    pub fn ingest<'a>(
        &mut self,
        payload: &'a [u8],
        layout: SampleLayout,
        region: SampleRegion,
        source: SourceDescriptor,
        timestamp: ClockStamp,
        sample_rate: Option<SampleRate>,
        calibration: CalibrationRecord,
        quality: QualityRecord,
        sequence: u64,
    ) -> Result<ExtendedIngestedFrame<'a>, ContractError> {
        self.ingest_with_optional_timestamp(
            payload, layout, region, source, Some(timestamp), sample_rate, calibration, quality,
            sequence,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ingest_with_optional_timestamp<'a>(
        &mut self,
        payload: &'a [u8],
        layout: SampleLayout,
        region: SampleRegion,
        source: SourceDescriptor,
        timestamp: Option<ClockStamp>,
        sample_rate: Option<SampleRate>,
        calibration: CalibrationRecord,
        quality: QualityRecord,
        sequence: u64,
    ) -> Result<ExtendedIngestedFrame<'a>, ContractError> {
        if source.source_id() != self.source_id {
            return Err(ContractError::InvalidSource);
        }
        let expected_bytes = usize::try_from(layout.required_byte_span()?)
            .map_err(|_| ContractError::ArithmeticOverflow)?;
        if payload.len() != expected_bytes {
            return Err(ContractError::OutOfBounds);
        }
        let measurement = ExtendedMeasurement::new_with_optional_timestamp(
            self.capability, layout, region, source, timestamp, sample_rate, calibration, quality,
        )?;
        let stream_id = self.order.stream_id();
        let generation = self.order.generation();
        let event = self.order.observe_optional(stream_id, generation, sequence, timestamp)?;
        Ok(ExtendedIngestedFrame { payload, measurement, sequence_event: event, sequence, stream_id, generation })
    }

    pub const fn source_id(&self) -> u64 { self.source_id }
    pub const fn capability(&self) -> ExtendedCapability { self.capability }
    pub const fn stream_id(&self) -> u64 { self.order.stream_id() }
    pub const fn generation(&self) -> u32 { self.order.generation() }
    /// Latest sequence paired with a known timestamp. Use `last_observation`
    /// for sequence preflight when the latest timestamp may be unknown.
    pub const fn last_sequence(&self) -> Option<(u64, ClockStamp)> { self.order.last() }
    pub const fn last_observation(&self) -> Option<(u64, Option<ClockStamp>)> { self.order.last_observation() }
    pub const fn last_known_timestamp(&self) -> Option<ClockStamp> { self.order.last_known_timestamp() }
}

/// A validated declaration paired with its borrowed bytes and sequence event.
#[derive(Debug, PartialEq)]
pub struct IngestedFrame<'a> {
    payload: &'a [u8],
    measurement: Measurement,
    sequence_event: SequenceEvent,
    sequence: u64,
    stream_id: u64,
    generation: u32,
}

/// Extended measurement declaration paired with its caller-owned bytes and
/// stream-order event.
#[derive(Debug, PartialEq)]
pub struct ExtendedIngestedFrame<'a> {
    payload: &'a [u8],
    measurement: ExtendedMeasurement,
    sequence_event: SequenceEvent,
    sequence: u64,
    stream_id: u64,
    generation: u32,
}

impl<'a> IngestedFrame<'a> {
    pub const fn payload(&self) -> &'a [u8] {
        self.payload
    }

    pub const fn measurement(&self) -> &Measurement {
        &self.measurement
    }

    pub const fn sequence_event(&self) -> SequenceEvent {
        self.sequence_event
    }

    pub const fn sequence(&self) -> u64 { self.sequence }

    pub const fn stream_id(&self) -> u64 { self.stream_id }

    pub const fn generation(&self) -> u32 { self.generation }
}

impl<'a> ExtendedIngestedFrame<'a> {
    pub const fn payload(&self) -> &'a [u8] { self.payload }
    pub const fn measurement(&self) -> &ExtendedMeasurement { &self.measurement }
    pub const fn sequence_event(&self) -> SequenceEvent { self.sequence_event }
    pub const fn sequence(&self) -> u64 { self.sequence }
    pub const fn stream_id(&self) -> u64 { self.stream_id }
    pub const fn generation(&self) -> u32 { self.generation }
}

#[cfg(test)]
#[path = "ingest_tests.rs"]
mod tests;
