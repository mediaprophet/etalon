//! Caller-buffered bridge from validated host SDR frames to the shared
//! observation envelope. Reader and tuning fields remain host assertions.

use etalon_sense_core::{
    evidence::EvidenceOrigin,
    observation::{HostPolicyDecision, MeasurementUncertainty, ObservationEnvelope, ObservationKind, ObservationTimeAnchor, QuantityUnitIds},
    quality::{CalibrationRecord, CalibrationState, QualityRecord, SourceDescriptor, SourceKind},
    ContractError,
};

use crate::{
    AppliedTuning, CaptureTrust, IqByteOrder, OverrunReport, RfTuning, SdrFrame,
    SdrPowerFrame, TimestampAnchor,
};

/// Host-supplied semantic mapping. Source ID comes exactly from the frame's
/// reader declaration; stream position is copied only from the reader report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SdrObservationMapping {
    pub source_kind: SourceKind,
    /// Host declaration of whether the frame is live, cached, or derived.
    pub observation_kind: ObservationKind,
    pub quantity_unit: QuantityUnitIds,
}

/// Payload is borrowed from the frame's caller-owned storage; no sample copy
/// or allocation is performed by the adapter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SdrPayload<'a> {
    InterleavedI16Iq(&'a [u8]),
    IqPowerF64(&'a [f64]),
}

/// Tuning and continuity values preserved as host/reader assertions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SdrCaptureMetadata {
    pub reader_id: u64,
    pub requested_tuning: RfTuning,
    pub applied_tuning: AppliedTuning,
    pub sample_count: usize,
    pub byte_order: IqByteOrder,
    pub timestamp_anchor: TimestampAnchor,
    pub overrun: OverrunReport,
    pub trust: CaptureTrust,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdrObservation<'a> {
    pub envelope: ObservationEnvelope,
    pub metadata: SdrCaptureMetadata,
    pub payload: SdrPayload<'a>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptError {
    InvalidMapping(ContractError),
}

/// Adapt one validated IQ frame into caller-owned metadata output. Timestamp
/// remains exactly the optional frame stamp; tuning, sample count, byte order,
/// overrun and unverified trust are preserved separately. The adapter does
/// not supply physical units, policy approval, calibration, or identity proof.
pub fn adapt_sdr_frame<'a>(
    frame: SdrFrame<'a>,
    mapping: SdrObservationMapping,
    output: &mut Option<SdrObservation<'a>>,
) -> Result<(), AdaptError> {
    *output = None;
    let envelope = make_envelope(
        frame.reader_id(), frame.stream_id(), frame.generation(), frame.sequence(),
        frame.timestamp(), frame.timestamp_anchor(), mapping,
    )?;
    *output = Some(SdrObservation {
        envelope,
        metadata: SdrCaptureMetadata {
            reader_id: frame.reader_id(),
            requested_tuning: frame.requested_tuning(),
            applied_tuning: frame.applied_tuning(),
            sample_count: frame.sample_count(),
            byte_order: frame.byte_order(),
            timestamp_anchor: frame.timestamp_anchor(),
            overrun: frame.overrun(),
            trust: frame.trust(),
        },
        payload: SdrPayload::InterleavedI16Iq(frame.bytes()),
    });
    Ok(())
}

/// Adapt a derived I/Q power frame. Power values retain their declared numeric
/// representation; no watt/dBm mapping is inferred. Metadata is still the
/// original reader's report, not evidence that processing or tuning was valid.
pub fn adapt_sdr_power_frame<'a>(
    frame: SdrPowerFrame<'a>,
    mapping: SdrObservationMapping,
    output: &mut Option<SdrObservation<'a>>,
) -> Result<(), AdaptError> {
    *output = None;
    let envelope = make_envelope(
        frame.reader_id(), frame.stream_id(), frame.generation(), frame.sequence(),
        frame.timestamp(), frame.timestamp_anchor(), mapping,
    )?;
    *output = Some(SdrObservation {
        envelope,
        metadata: SdrCaptureMetadata {
            reader_id: frame.reader_id(),
            requested_tuning: frame.requested_tuning(),
            applied_tuning: frame.applied_tuning(),
            sample_count: frame.sample_count(),
            byte_order: frame.byte_order(),
            timestamp_anchor: frame.timestamp_anchor(),
            overrun: frame.overrun(),
            trust: frame.trust(),
        },
        payload: SdrPayload::IqPowerF64(frame.power_values()),
    });
    Ok(())
}

fn make_envelope(
    reader_id: u64,
    stream_id: Option<u64>,
    generation: Option<u32>,
    sequence: Option<u64>,
    timestamp: Option<etalon_sense_core::time::ClockStamp>,
    anchor: TimestampAnchor,
    mapping: SdrObservationMapping,
) -> Result<ObservationEnvelope, AdaptError> {
    let source = SourceDescriptor::new(reader_id, mapping.source_kind)
        .map_err(AdaptError::InvalidMapping)?;
    let calibration = CalibrationRecord::new(CalibrationState::Missing)
        .map_err(AdaptError::InvalidMapping)?;
    ObservationEnvelope::new(
        mapping.observation_kind,
        source,
        stream_id,
        generation,
        sequence,
        timestamp,
        mapping.quantity_unit,
        MeasurementUncertainty::Unknown,
        QualityRecord::Unknown,
        calibration,
        EvidenceOrigin::Unknown,
        None,
        HostPolicyDecision::NotEvaluated,
    ).and_then(|envelope| envelope.with_time_anchor(match anchor {
        TimestampAnchor::Unknown => ObservationTimeAnchor::Unknown,
        TimestampAnchor::FirstSample => ObservationTimeAnchor::FirstSample,
        TimestampAnchor::LastSample => ObservationTimeAnchor::LastSample,
        TimestampAnchor::ReadCompletion => ObservationTimeAnchor::ReadCompletion,
    })).map_err(AdaptError::InvalidMapping)
}

#[cfg(test)]
#[path = "observation_tests.rs"]
mod tests;
