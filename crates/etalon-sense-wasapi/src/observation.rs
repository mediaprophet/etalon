//! Common typed metadata for one completed WASAPI batch.
//!
//! QPC/device positions and host dispatch time remain in their native,
//! declared forms. This adapter does not turn them into a sample `ClockStamp`;
//! use a separately qualified host time mapper for that transformation.

use etalon_sense_core::{
    adapter::{AdapterError, AuthorizedExtendedCaptureBuffer},
    evidence::EvidenceOrigin,
    observation::{HostPolicyDecision, MeasurementUncertainty, ObservationEnvelope, ObservationKind, QuantityUnitIds},
    policy::{PolicyOutcome, PolicyRequest},
    quality::{CalibrationRecord, CalibrationState, QualityRecord, SourceDescriptor, SourceKind},
    ContractError,
};

use crate::{CaptureMetadata, WasapiCaptureAdapter, WasapiFormat};
use etalon_sense_core::authority::TrustedCaptureClock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WasapiObservationMapping {
    pub source_kind: SourceKind,
    /// Exact host policy assertion associated with this batch, if supplied.
    pub policy: Option<(PolicyRequest, PolicyOutcome)>,
}

/// Raw metadata tied to the session that produced the copied bytes. QPC and
/// device positions stay raw; `host_clock_at_dispatch` is not re-labeled as a
/// sample timestamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WasapiBatchMetadata {
    pub session_id: u64,
    pub generation: u32,
    pub capture: CaptureMetadata,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WasapiObservation<'a> {
    pub envelope: ObservationEnvelope,
    pub metadata: WasapiBatchMetadata,
    pub payload: &'a [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationError {
    Capture(AdapterError),
    MetadataUnavailable,
    MetadataMismatch,
    Contract(ContractError),
}

/// Capture once and immediately pair the bytes with the adapter's just-written
/// metadata. This is the closest binding available from the current WASAPI API;
/// the side metadata itself has no hardware signature or portable batch ID.
pub fn capture_observation<'a, C: TrustedCaptureClock>(
    adapter: &mut WasapiCaptureAdapter<'_, C>,
    mut buffer: AuthorizedExtendedCaptureBuffer<'a>,
    mapping: WasapiObservationMapping,
    output: &mut Option<WasapiObservation<'a>>,
) -> Result<(), ObservationError> {
    *output = None;
    let session = buffer.session();
    let envelope = match make_envelope(session, mapping) {
        Ok(envelope) => envelope,
        Err(error) => {
            buffer.writable_bytes().fill(0);
            adapter.fail_closed_capture();
            return Err(error);
        }
    };
    let (samples, metadata) = adapter.capture_with_metadata(buffer).map_err(ObservationError::Capture)?;
    *output = Some(WasapiObservation {
        envelope,
        metadata: WasapiBatchMetadata {
            session_id: session.session_id(),
            generation: session.generation(),
            capture: metadata,
        },
        payload: samples.bytes(),
    });
    Ok(())
}

pub(crate) fn validate_produced_batch(
    layout: &etalon_sense_core::buffer::SampleLayout,
    sample_bytes: usize,
    metadata: CaptureMetadata,
    negotiated: WasapiFormat,
) -> Result<(), ObservationError> {
    let span = layout.required_byte_span().map_err(ObservationError::Contract)?;
    let bytes = usize::try_from(span)
        .map_err(|_| ObservationError::Contract(ContractError::ArithmeticOverflow))?;
    let packed_bytes = usize::try_from(metadata.frames)
        .ok()
        .and_then(|frames| frames.checked_mul(usize::from(metadata.format.block_align())))
        .ok_or(ObservationError::Contract(ContractError::ArithmeticOverflow))?;
    let expected_frames = layout.extents()[0];
    if layout.rank() != 2
        || sample_bytes != bytes
        || bytes != packed_bytes
        || metadata.format != negotiated
        || metadata.frames != expected_frames
        || layout.extents()[1] != u32::from(metadata.format.channels())
        || *layout.encoding() != metadata.format.encoding()
        || metadata.silent_frames > metadata.frames
    {
        return Err(ObservationError::MetadataMismatch);
    }
    Ok(())
}

fn make_envelope(
    session: etalon_sense_core::adapter::ExtendedCaptureSession,
    mapping: WasapiObservationMapping,
) -> Result<ObservationEnvelope, ObservationError> {
    let source = SourceDescriptor::new(session.source().source_id(), mapping.source_kind)
        .map_err(ObservationError::Contract)?;
    let capability = session.source().capability();
    let quantity_unit = QuantityUnitIds::new(Some(capability.quantity_id()), capability.unit_id())
        .map_err(ObservationError::Contract)?;
    let policy = mapping.policy.map_or(HostPolicyDecision::NotEvaluated, |(request, outcome)| {
        HostPolicyDecision::Evaluated { request, outcome }
    });
    let calibration = CalibrationRecord::new(CalibrationState::Missing)
        .map_err(ObservationError::Contract)?;
    ObservationEnvelope::new(
        ObservationKind::LiveFrame,
        source,
        Some(session.session_id()),
        Some(session.generation()),
        None,
        None,
        quantity_unit,
        MeasurementUncertainty::Unknown,
        QualityRecord::Unknown,
        calibration,
        EvidenceOrigin::Unknown,
        None,
        policy,
    ).map_err(ObservationError::Contract)
}

#[cfg(test)]
#[path = "observation_tests.rs"]
mod tests;
