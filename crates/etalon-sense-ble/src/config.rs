use etalon_sense_core::{
    adapter::{AdapterError, AdapterId, AuthorizedExtendedCapture, ExtendedDiscoveredSource},
    extended_capability::ExtendedCapability,
    quantity::SampleEncoding,
    time::ClockStamp,
    ContractError,
};

use crate::queue::QUEUE_CAPACITY;

pub const MAX_CAPTURE_SAMPLES: usize = QUEUE_CAPACITY;
pub const MAX_SESSION_OBSERVATIONS: usize = 4_096;
pub const MAX_SESSION_DURATION_MS: u32 = 60_000;

/// Exact adapter-side scope passed to a host-owned live authorization gate.
/// The gate implementation should retain the complete host policy request,
/// including subject, actor, context, and decision commitment, in its own
/// opaque shared state. The IDs and timestamps here are consistency inputs,
/// not authenticated evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BleCollectionScope {
    source: ExtendedDiscoveredSource,
    purpose_id: u64,
    authorized_at: ClockStamp,
    expires_at: ClockStamp,
}

impl BleCollectionScope {
    pub(super) const fn from_request(request: &AuthorizedExtendedCapture) -> Self {
        Self {
            source: request.source(),
            purpose_id: request.purpose_id(),
            authorized_at: request.authorized_at(),
            expires_at: request.expires_at(),
        }
    }

    pub const fn source(self) -> ExtendedDiscoveredSource {
        self.source
    }
    pub const fn purpose_id(self) -> u64 {
        self.purpose_id
    }
    pub const fn authorized_at(self) -> ClockStamp {
        self.authorized_at
    }
    pub const fn expires_at(self) -> ClockStamp {
        self.expires_at
    }
}

/// Host-owned live policy/revocation gate for scanning and every observation.
///
/// `with_permitted_collection` must use a trusted fresh clock and current
/// policy/revocation state, fail closed when unknown, and call `collect` exactly
/// once only while its policy snapshot remains valid. Hosts should serialize
/// revocation changes against that closure. Implementations must be bounded and
/// safe on a WinRT callback thread, must not unwind, and must not re-enter this
/// adapter. The adapter catches unwind-enabled panics and fails closed, but
/// hosts must not rely on panic catching when built with `panic=abort`. This
/// interface cannot make OS radio event delivery atomic with an external policy
/// update; without a real gate, no advertisement properties are read or retained.
pub trait BleCollectionGate: Send + Sync + 'static {
    fn with_permitted_collection(
        &self,
        scope: BleCollectionScope,
        collect: &mut dyn FnMut(),
    ) -> bool;
}

pub(super) fn run_collection_gate(
    gate: &dyn BleCollectionGate,
    scope: BleCollectionScope,
    collect: &mut dyn FnMut(),
) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gate.with_permitted_collection(scope, collect)
    }))
    .unwrap_or(false)
}

/// Host-declared stream IDs and bounded capture size for the BLE receiver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BleConfig {
    pub(super) adapter_id: AdapterId,
    pub(super) source_id: u64,
    pub(super) capability: ExtendedCapability,
    max_batch_samples: usize,
    max_session_observations: usize,
    max_session_duration_ms: u32,
}

impl BleConfig {
    /// Build a scalar I16 RSSI capability. The host is responsible for
    /// selecting registered modality, RSSI quantity, and dBm unit IDs.
    pub fn new(
        adapter_id: AdapterId,
        source_id: u64,
        modality_id: u64,
        rssi_quantity_id: u64,
        dbm_unit_id: u64,
        max_batch_samples: usize,
        max_session_observations: usize,
        max_session_duration_ms: u32,
    ) -> Result<Self, AdapterError> {
        if source_id == 0
            || dbm_unit_id == 0
            || max_batch_samples == 0
            || max_batch_samples > MAX_CAPTURE_SAMPLES
            || max_session_observations == 0
            || max_session_observations > MAX_SESSION_OBSERVATIONS
            || max_session_duration_ms == 0
            || max_session_duration_ms > MAX_SESSION_DURATION_MS
        {
            return Err(AdapterError::InvalidIdentifier);
        }
        let maximum = u32::try_from(max_batch_samples)
            .map_err(|_| AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
        let capability = ExtendedCapability::new(
            modality_id,
            rssi_quantity_id,
            Some(dbm_unit_id),
            SampleEncoding::I16,
            [maximum, 0, 0, 0],
            None,
        )?;
        Ok(Self {
            adapter_id,
            source_id,
            capability,
            max_batch_samples,
            max_session_observations,
            max_session_duration_ms,
        })
    }

    pub const fn capability(self) -> ExtendedCapability {
        self.capability
    }
    pub const fn source_id(self) -> u64 {
        self.source_id
    }
    pub const fn max_batch_samples(self) -> usize {
        self.max_batch_samples
    }
    pub const fn max_session_observations(self) -> usize {
        self.max_session_observations
    }
    pub const fn max_session_duration_ms(self) -> u32 {
        self.max_session_duration_ms
    }
}
