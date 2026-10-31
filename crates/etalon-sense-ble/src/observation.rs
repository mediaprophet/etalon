//! Common metadata for one successful passive BLE RSSI batch.
//!
//! Bluetooth addresses, OS address types, and Windows timestamps remain raw
//! side metadata. No address becomes a subject/source identity, and no Windows time is converted
//! to a `ClockStamp` here. The borrow of the adapter's batch records prevents a
//! subsequent capture from replacing them while this observation is live.

use etalon_sense_core::{
    adapter::{AdapterError, AuthorizedExtendedCaptureBuffer, ExtendedSensorAdapter},
    evidence::EvidenceOrigin,
    observation::{HostPolicyDecision, MeasurementUncertainty, ObservationEnvelope, ObservationKind, QuantityUnitIds},
    policy::{PolicyOutcome, PolicyRequest},
    quality::{CalibrationRecord, CalibrationState, QualityRecord, SourceDescriptor, SourceKind},
    ContractError,
};

use etalon_sense_core::authority::TrustedCaptureClock;
use crate::{BleAdvertisementSample, BleRssiAdapter};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BleObservationMapping {
    pub source_kind: SourceKind,
    /// Exact host policy assertion associated with this batch, if supplied.
    pub policy: Option<(PolicyRequest, PolicyOutcome)>,
}

/// The cumulative drop-counter reads bracket queue draining; because some
/// callback drop paths do not take the queue mutex, their difference is not
/// an exact per-batch loss count and must not be used to infer sample gaps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BleDropCountInterval {
    cumulative_before_drain: u64,
    cumulative_after_drain: u64,
}

impl BleDropCountInterval {
    pub const fn cumulative_before_drain(self) -> u64 { self.cumulative_before_drain }
    pub const fn cumulative_after_drain(self) -> u64 { self.cumulative_after_drain }
}

/// Batch/session metadata plus each raw address, RSSI, and Windows UTC tick.
/// Each record exposes `radio_identifier_observation()` for typed peer metadata;
/// the batch envelope continues to describe the collecting source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BleBatchMetadata<'a> {
    pub session_id: u64,
    pub generation: u32,
    pub sample_count: usize,
    pub drop_count_interval: BleDropCountInterval,
    pub records: &'a [BleAdvertisementSample],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BleObservation<'bytes, 'records> {
    pub envelope: ObservationEnvelope,
    pub metadata: BleBatchMetadata<'records>,
    pub payload: &'bytes [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationError {
    Capture(AdapterError),
    MetadataMismatch,
    Contract(ContractError),
}

/// Capture and immediately bind the resulting RSSI bytes to the side records
/// retained by this adapter. It rejects a count/layout/RSSI mismatch. The
/// backend has no portable batch token, so this is a synchronous association,
/// not a cryptographic proof of origin or authenticity.
pub fn capture_observation<'bytes, 'records, C: TrustedCaptureClock>(
    adapter: &'records mut BleRssiAdapter<C>,
    buffer: AuthorizedExtendedCaptureBuffer<'bytes>,
    mapping: BleObservationMapping,
    output: &mut Option<BleObservation<'bytes, 'records>>,
) -> Result<(), ObservationError> {
    *output = None;
    let session = buffer.session();
    let layout = buffer.layout().clone();
    let samples = adapter.capture(buffer).map_err(ObservationError::Capture)?;
    let (dropped_before, dropped_after) = adapter
        .last_capture_drop_count_interval()
        .ok_or(ObservationError::MetadataMismatch)?;
    if dropped_after < dropped_before {
        return Err(ObservationError::MetadataMismatch);
    }
    let records = adapter.last_capture_records();
    validate_batch_binding(samples.bytes(), samples.layout(), &layout, records)?;

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
    let envelope = ObservationEnvelope::new(
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
    ).map_err(ObservationError::Contract)?;
    *output = Some(BleObservation {
        envelope,
        metadata: BleBatchMetadata {
            session_id: session.session_id(),
            generation: session.generation(),
            sample_count: records.len(),
            drop_count_interval: BleDropCountInterval {
                cumulative_before_drain: dropped_before,
                cumulative_after_drain: dropped_after,
            },
            records,
        },
        payload: samples.bytes(),
    });
    Ok(())
}

fn validate_batch_binding(
    payload: &[u8],
    captured_layout: &etalon_sense_core::buffer::SampleLayout,
    expected_layout: &etalon_sense_core::buffer::SampleLayout,
    records: &[BleAdvertisementSample],
) -> Result<(), ObservationError> {
    let count = usize::try_from(expected_layout.extents()[0])
        .map_err(|_| ObservationError::Contract(ContractError::ArithmeticOverflow))?;
    let expected_bytes = count.checked_mul(2).ok_or(ObservationError::Contract(ContractError::ArithmeticOverflow))?;
    if expected_layout.rank() != 1
        || *expected_layout.encoding() != etalon_sense_core::quantity::SampleEncoding::I16
        || expected_layout.strides_bytes()[0] != 2
        || captured_layout != expected_layout
        || payload.len() != expected_bytes
        || records.len() != count
    {
        return Err(ObservationError::MetadataMismatch);
    }
    for (record, value_bytes) in records.iter().zip(payload.chunks_exact(2)) {
        let expected_bytes = record.rssi_dbm.to_le_bytes();
        if value_bytes != expected_bytes.as_slice() {
            return Err(ObservationError::MetadataMismatch);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "observation_tests.rs"]
mod tests;
