//! Bounded, caller-buffered alignment of typed observations in one target clock.
//!
//! This module never compares raw values from different clock domains. A
//! non-target timestamp requires exactly one supplied, valid
//! `ClockSynchronization` whose reference domain matches the target. This is
//! metadata alignment only; it does not fuse payloads or establish truth.

use core::num::NonZeroU64;

use crate::{
    observation::{ObservationEnvelope, ObservationKind, ObservationTimeAnchor},
    quality::SourceKind,
    synchronization::{ClockSynchronization, SynchronizationError},
    time::ClockStamp,
    ContractError,
};

/// Hard ceiling on expected observations and input rows per alignment call.
pub const MAX_ALIGNMENT_MEMBERS: usize = 64;
pub const MAX_ALIGNMENT_SYNCHRONIZATIONS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClockDomain {
    clock_id: u64,
    restart_epoch: u32,
}

impl ClockDomain {
    pub const fn new(clock_id: u64, restart_epoch: u32) -> Result<Self, ContractError> {
        if clock_id == 0 { return Err(ContractError::InvalidIdentifier); }
        Ok(Self { clock_id, restart_epoch })
    }

    pub const fn clock_id(self) -> u64 { self.clock_id }
    pub const fn restart_epoch(self) -> u32 { self.restart_epoch }

    pub const fn matches(self, stamp: ClockStamp) -> bool {
        self.clock_id == stamp.clock_id() && self.restart_epoch == stamp.restart_epoch()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingObservationPolicy {
    Reject,
    PreserveMissing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnknownTimestampPolicy {
    Reject,
    /// Keep absent timestamps or unknown event anchors in output without an
    /// aligned time. Such entries are excluded from skew and correlation.
    PreserveUnaligned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlignmentConfig {
    target: ClockDomain,
    max_skew_ns: u64,
    missing: MissingObservationPolicy,
    unknown_time: UnknownTimestampPolicy,
}

impl AlignmentConfig {
    pub const fn new(
        target: ClockDomain,
        max_skew_ns: u64,
        missing: MissingObservationPolicy,
        unknown_time: UnknownTimestampPolicy,
    ) -> Self {
        Self { target, max_skew_ns, missing, unknown_time }
    }

    pub const fn target(self) -> ClockDomain { self.target }
    pub const fn max_skew_ns(self) -> u64 { self.max_skew_ns }
    pub const fn missing_policy(self) -> MissingObservationPolicy { self.missing }
    pub const fn unknown_timestamp_policy(self) -> UnknownTimestampPolicy { self.unknown_time }
}

/// Stable matching key for one expected observation. Route and source kind are
/// included so a live, recorded, or synthetic frame cannot satisfy a key for
/// a different declared origin. It preserves unknown stream fields; callers
/// needing multiple observations with the same unknown key must supply a
/// stronger host-defined key outside this module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservationKey {
    observation_kind: ObservationKind,
    source_kind: SourceKind,
    source_id: u64,
    stream_id: Option<u64>,
    generation: Option<u32>,
    sequence: Option<u64>,
    quantity_id: Option<u64>,
    unit_id: Option<u64>,
}

impl ObservationKey {
    pub const fn new(
        observation_kind: ObservationKind,
        source_kind: SourceKind,
        source_id: u64,
        stream_id: Option<u64>,
        generation: Option<u32>,
        sequence: Option<u64>,
        quantity_id: Option<u64>,
        unit_id: Option<u64>,
    ) -> Result<Self, ContractError> {
        if source_id == 0 || matches!(stream_id, Some(0)) || matches!(generation, Some(0))
            || matches!(quantity_id, Some(0)) || matches!(unit_id, Some(0))
        {
            return Err(ContractError::InvalidIdentifier);
        }
        if stream_id.is_none() && (generation.is_some() || sequence.is_some()) {
            return Err(ContractError::InvalidSource);
        }
        Ok(Self {
            observation_kind, source_kind, source_id, stream_id, generation,
            sequence, quantity_id, unit_id,
        })
    }

    pub const fn from_observation(observation: ObservationEnvelope) -> Self {
        let quantity = observation.quantity_unit();
        Self {
            observation_kind: observation.kind(),
            source_kind: observation.source().kind(),
            source_id: observation.source().source_id(),
            stream_id: match observation.stream_id() { Some(value) => Some(value.get()), None => None },
            generation: match observation.generation() { Some(value) => Some(value.get()), None => None },
            sequence: observation.sequence(),
            quantity_id: match quantity.quantity_id() { Some(value) => Some(value.get()), None => None },
            unit_id: match quantity.unit_id() { Some(value) => Some(value.get()), None => None },
        }
    }

    pub const fn observation_kind(self) -> ObservationKind { self.observation_kind }
    pub const fn source_kind(self) -> SourceKind { self.source_kind }
}

/// Opaque host anchor reference. It has no event semantics by itself and is
/// ignored unless a `TimestampAnchorAuthority` explicitly resolves it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TimestampAnchorId(NonZeroU64);

impl TimestampAnchorId {
    pub const fn new(value: u64) -> Result<Self, ContractError> {
        match NonZeroU64::new(value) {
            Some(value) => Ok(Self(value)),
            None => Err(ContractError::InvalidIdentifier),
        }
    }

    pub const fn get(self) -> u64 { self.0.get() }
}

/// Envelope plus optional host anchor reference. The reference never overrides
/// the envelope's typed `ObservationTimeAnchor` without an explicit authority.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimedObservation {
    observation: ObservationEnvelope,
    host_anchor_id: Option<TimestampAnchorId>,
}

impl TimedObservation {
    pub const fn new(
        observation: ObservationEnvelope,
        host_anchor_id: Option<TimestampAnchorId>,
    ) -> Self {
        Self { observation, host_anchor_id }
    }

    pub const fn observation(self) -> ObservationEnvelope { self.observation }
    pub const fn host_anchor_id(self) -> Option<TimestampAnchorId> { self.host_anchor_id }
}

/// Canonical anchor used for equality after either typed envelope comparison
/// or explicit resolution by a host authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlignmentAnchor {
    Envelope(ObservationTimeAnchor),
    AuthorityMapped { authority_id: u64, anchor_id: TimestampAnchorId },
}

/// Explicit host authority for equivalence between an envelope's typed anchor
/// and optional host anchor IDs. `Unknown` has no equivalence by default; this
/// method must deliberately return a canonical ID to resolve it. The trait is
/// a trust boundary supplied by the caller, not a cryptographic verifier.
pub trait TimestampAnchorAuthority {
    /// Stable nonzero host registry identity for this mapping authority.
    fn authority_id(&self) -> u64;

    fn resolve(
        &self,
        envelope_anchor: ObservationTimeAnchor,
        host_anchor_id: Option<TimestampAnchorId>,
    ) -> Option<TimestampAnchorId>;
}

/// One matched output slot. The original envelope is retained unchanged;
/// aligned time and synchronization provenance are separate declarations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlignedObservation {
    key: ObservationKey,
    observation: Option<ObservationEnvelope>,
    target_timestamp: Option<ClockStamp>,
    alignment_anchor: Option<AlignmentAnchor>,
    synchronization_id: Option<u64>,
}

impl AlignedObservation {
    pub const fn key(self) -> ObservationKey { self.key }
    pub const fn observation(self) -> Option<ObservationEnvelope> { self.observation }
    pub const fn target_timestamp(self) -> Option<ClockStamp> { self.target_timestamp }
    pub const fn alignment_anchor(self) -> Option<AlignmentAnchor> { self.alignment_anchor }
    pub const fn synchronization_id(self) -> Option<u64> { self.synchronization_id }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlignmentReport {
    pub written: usize,
    pub present: usize,
    pub aligned: usize,
    pub unaligned: usize,
    pub missing: usize,
    /// Conservative width of the union of mapped timestamp uncertainty ranges.
    /// `None` means no timestamps were alignable under the selected policy.
    pub conservative_skew_ns: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlignmentError {
    TooManyMembers { maximum: usize },
    TooManySynchronizations { maximum: usize },
    OutputTooSmall { required: usize },
    DuplicateExpectedKey,
    DuplicateObservation,
    UnexpectedObservation,
    MissingObservation,
    UnknownTimestamp,
    InvalidTimestampAnchorAuthority,
    MissingClockSynchronization,
    AmbiguousClockSynchronization,
    TimestampAnchorMismatch,
    ClockSynchronization(SynchronizationError),
    SkewExceeded { measured_ns: u64, maximum_ns: u64 },
    ArithmeticOverflow,
}

/// Align one observation per expected key. `output` is cleared before work and
/// cleared again on every error, so duplicate, mapping, and skew failures are
/// atomic from the caller's perspective. All output slots follow `expected`
/// order; absent entries are retained only when configured.
pub fn align_observations(
    observations: &[TimedObservation],
    expected: &[ObservationKey],
    synchronizations: &[ClockSynchronization],
    config: AlignmentConfig,
    output: &mut [Option<AlignedObservation>],
) -> Result<AlignmentReport, AlignmentError> {
    align_with_authority(observations, expected, synchronizations, config, None, output)
}

/// Align using an explicit host authority for custom or cross-kind timestamp
/// anchor equivalence. This is the only path that can resolve an unknown
/// envelope anchor or declare two different typed anchors equivalent.
pub fn align_observations_with_anchor_authority(
    observations: &[TimedObservation],
    expected: &[ObservationKey],
    synchronizations: &[ClockSynchronization],
    config: AlignmentConfig,
    authority: &dyn TimestampAnchorAuthority,
    output: &mut [Option<AlignedObservation>],
) -> Result<AlignmentReport, AlignmentError> {
    align_with_authority(observations, expected, synchronizations, config, Some(authority), output)
}

fn align_with_authority(
    observations: &[TimedObservation],
    expected: &[ObservationKey],
    synchronizations: &[ClockSynchronization],
    config: AlignmentConfig,
    authority: Option<&dyn TimestampAnchorAuthority>,
    output: &mut [Option<AlignedObservation>],
) -> Result<AlignmentReport, AlignmentError> {
    output.fill(None);
    let result = align_inner(observations, expected, synchronizations, config, authority, output);
    if result.is_err() { output.fill(None); }
    result
}

fn align_inner(
    observations: &[TimedObservation],
    expected: &[ObservationKey],
    synchronizations: &[ClockSynchronization],
    config: AlignmentConfig,
    authority: Option<&dyn TimestampAnchorAuthority>,
    output: &mut [Option<AlignedObservation>],
) -> Result<AlignmentReport, AlignmentError> {
    if output.len() < expected.len() {
        return Err(AlignmentError::OutputTooSmall { required: expected.len() });
    }
    if expected.len() > MAX_ALIGNMENT_MEMBERS || observations.len() > MAX_ALIGNMENT_MEMBERS {
        return Err(AlignmentError::TooManyMembers { maximum: MAX_ALIGNMENT_MEMBERS });
    }
    if synchronizations.len() > MAX_ALIGNMENT_SYNCHRONIZATIONS {
        return Err(AlignmentError::TooManySynchronizations { maximum: MAX_ALIGNMENT_SYNCHRONIZATIONS });
    }
    for (index, key) in expected.iter().enumerate() {
        if expected[..index].contains(key) { return Err(AlignmentError::DuplicateExpectedKey); }
        output[index] = Some(AlignedObservation {
            key: *key, observation: None, target_timestamp: None,
            alignment_anchor: None, synchronization_id: None,
        });
    }

    let mut min_lower = None;
    let mut max_upper = None;
    let mut common_anchor: Option<AlignmentAnchor> = None;
    let mut aligned_count = 0usize;
    for timed in observations {
        let observation = timed.observation;
        let key = ObservationKey::from_observation(observation);
        let slot_index = expected.iter().position(|expected_key| *expected_key == key)
            .ok_or(AlignmentError::UnexpectedObservation)?;
        let slot = output[slot_index].as_mut().expect("expected slots initialized");
        if slot.observation.is_some() { return Err(AlignmentError::DuplicateObservation); }

        let anchor = match authority {
            Some(authority) => {
                let authority_id = authority.authority_id();
                if authority_id == 0 { return Err(AlignmentError::InvalidTimestampAnchorAuthority); }
                authority.resolve(observation.time_anchor(), timed.host_anchor_id)
                    .map(|anchor_id| AlignmentAnchor::AuthorityMapped { authority_id, anchor_id })
            }
            None => match observation.time_anchor() {
                ObservationTimeAnchor::Unknown | ObservationTimeAnchor::NativeEvent => None,
                anchor => Some(AlignmentAnchor::Envelope(anchor)),
            },
        };
        let (target_timestamp, synchronization_id) = match (observation.timestamp(), anchor) {
            (None, _) | (Some(_), None) => match config.unknown_time {
                UnknownTimestampPolicy::Reject => return Err(AlignmentError::UnknownTimestamp),
                UnknownTimestampPolicy::PreserveUnaligned => (None, None),
            },
            (Some(stamp), Some(anchor)) if config.target.matches(stamp) => {
                if common_anchor.is_some_and(|known| known != anchor) {
                    return Err(AlignmentError::TimestampAnchorMismatch);
                }
                common_anchor = Some(anchor);
                (Some(stamp), None)
            }
            (Some(stamp), Some(anchor)) => {
                if common_anchor.is_some_and(|known| known != anchor) {
                    return Err(AlignmentError::TimestampAnchorMismatch);
                }
                common_anchor = Some(anchor);
                let mapped = map_timestamp(stamp, config.target, synchronizations)?;
                (Some(mapped.0), Some(mapped.1))
            }
        };

        if let Some(stamp) = target_timestamp {
            aligned_count += 1;
            let interval = stamp.interval();
            min_lower = Some(min_lower.map_or(interval.lower_ns(), |current: u64| current.min(interval.lower_ns())));
            max_upper = Some(max_upper.map_or(interval.upper_ns(), |current: u64| current.max(interval.upper_ns())));
        }
        *slot = Some(AlignedObservation {
            key, observation: Some(observation), target_timestamp,
            alignment_anchor: anchor, synchronization_id,
        });
    }

    for slot in &output[..expected.len()] {
        if slot.is_some_and(|entry| entry.observation.is_none())
            && config.missing == MissingObservationPolicy::Reject
        {
            return Err(AlignmentError::MissingObservation);
        }
    }

    let conservative_skew_ns = match (min_lower, max_upper) {
        (Some(lower), Some(upper)) => Some(upper.checked_sub(lower).ok_or(AlignmentError::ArithmeticOverflow)?),
        _ => None,
    };
    if let Some(measured_ns) = conservative_skew_ns {
        if measured_ns > config.max_skew_ns {
            return Err(AlignmentError::SkewExceeded { measured_ns, maximum_ns: config.max_skew_ns });
        }
    }
    let present = observations.len();
    Ok(AlignmentReport {
        written: expected.len(),
        present,
        aligned: aligned_count,
        unaligned: present - aligned_count,
        missing: expected.len() - present,
        conservative_skew_ns,
    })
}

fn map_timestamp(
    stamp: ClockStamp,
    target: ClockDomain,
    synchronizations: &[ClockSynchronization],
) -> Result<(ClockStamp, u64), AlignmentError> {
    let mut selected = None;
    for synchronization in synchronizations {
        if synchronization.reference_domain() != (target.clock_id, target.restart_epoch)
            || synchronization.source_domain() != (stamp.clock_id(), stamp.restart_epoch())
        {
            continue;
        }
        if selected.is_some() { return Err(AlignmentError::AmbiguousClockSynchronization); }
        selected = Some(synchronization);
    }
    let synchronization = selected.ok_or(AlignmentError::MissingClockSynchronization)?;
    let mapped = synchronization.map(stamp)
        .map_err(AlignmentError::ClockSynchronization)?;
    if !target.matches(mapped) { return Err(AlignmentError::MissingClockSynchronization); }
    Ok((mapped, synchronization.synchronization_id()))
}

#[cfg(test)]
#[path = "alignment_tests.rs"]
mod tests;
