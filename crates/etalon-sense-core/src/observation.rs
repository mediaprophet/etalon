//! Fixed-size envelope joining sensor observations and cached query results.
//!
//! All identifiers and policy results are declarations supplied by a host.
//! This module assigns no unit mappings, source authority, identity model, or
//! privacy policy. `None` preserves unavailable metadata. Payload bytes remain
//! in caller-owned storage and are not accessed by this envelope.

use core::num::{NonZeroU32, NonZeroU64};

use crate::{
    evidence::{ArtifactDigest, EvidenceOrigin},
    policy::{PolicyOutcome, PolicyRequest},
    quality::{CalibrationRecord, QualityRecord, SourceDescriptor},
    time::ClockStamp,
    ContractError,
};

/// Declared observation route. It does not imply freshness or authenticity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationKind {
    /// A host reports that the data came from a live sensor path.
    LiveFrame,
    /// A frame read from a recording, as declared by the host.
    RecordedFrame,
    /// A frame generated synthetically, as declared by the host.
    SyntheticFrame,
    /// A host reports that the result came from a cache or prior query.
    CachedQuery,
    /// A host reports another derived or telemetry observation.
    HostDerived,
}

/// Opaque registry identifiers for a physical quantity and its unit.
///
/// No mapping is inferred between IDs, enum variants, or numeric values. The
/// pair may be partially unknown where a host only knows one identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantityUnitIds {
    quantity_id: Option<NonZeroU64>,
    unit_id: Option<NonZeroU64>,
}

impl QuantityUnitIds {
    pub const fn new(quantity_id: Option<u64>, unit_id: Option<u64>) -> Result<Self, ContractError> {
        let quantity_id = match quantity_id {
            Some(value) => match NonZeroU64::new(value) {
                Some(value) => Some(value),
                None => return Err(ContractError::InvalidIdentifier),
            },
            None => None,
        };
        let unit_id = match unit_id {
            Some(value) => match NonZeroU64::new(value) {
                Some(value) => Some(value),
                None => return Err(ContractError::InvalidIdentifier),
            },
            None => None,
        };
        Ok(Self { quantity_id, unit_id })
    }

    pub const fn quantity_id(self) -> Option<NonZeroU64> { self.quantity_id }
    pub const fn unit_id(self) -> Option<NonZeroU64> { self.unit_id }
}

/// Physical measurement uncertainty, kept separate from quality confidence.
///
/// Absolute and interval values use the declared unit ID when one is known.
/// Relative bounds are dimensionless fractions and are not probabilities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeasurementUncertainty {
    Unknown,
    AbsoluteBound { magnitude: f64, unit_id: Option<NonZeroU64> },
    RelativeBound { fraction: f64 },
    Interval { lower: f64, upper: f64, unit_id: Option<NonZeroU64> },
}

impl MeasurementUncertainty {
    pub fn absolute_bound(magnitude: f64, unit_id: Option<u64>) -> Result<Self, ContractError> {
        if !magnitude.is_finite() || magnitude < 0.0 {
            return Err(ContractError::InvalidQuality);
        }
        Ok(Self::AbsoluteBound { magnitude, unit_id: nonzero(unit_id)? })
    }

    pub fn relative_bound(fraction: f64) -> Result<Self, ContractError> {
        if !fraction.is_finite() || fraction < 0.0 {
            return Err(ContractError::InvalidQuality);
        }
        Ok(Self::RelativeBound { fraction })
    }

    pub fn interval(lower: f64, upper: f64, unit_id: Option<u64>) -> Result<Self, ContractError> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(ContractError::InvalidQuality);
        }
        Ok(Self::Interval { lower, upper, unit_id: nonzero(unit_id)? })
    }
}

fn nonzero(value: Option<u64>) -> Result<Option<NonZeroU64>, ContractError> {
    match value {
        Some(value) => NonZeroU64::new(value)
            .map(Some)
            .ok_or(ContractError::InvalidIdentifier),
        None => Ok(None),
    }
}

/// Optional, caller-declared host policy context and result.
///
/// `Permit` carries only the host's opaque decision commitment. Storing it does
/// not revalidate expiry, revocation, identity, or policy freshness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostPolicyDecision {
    NotEvaluated,
    Evaluated { request: PolicyRequest, outcome: PolicyOutcome },
}

/// What an observation's timestamp describes. This is independent of its
/// clock domain and does not imply synchronization with another source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationTimeAnchor {
    Unknown,
    FirstSample,
    LastSample,
    Midpoint,
    ReadCompletion,
    NativeEvent,
}

/// Bounded metadata for one stream frame, cached query result, or host result.
///
/// Source trust remains whatever the supplied `SourceDescriptor` declares
/// (currently unverified). `EvidenceOrigin` preserves unknown/asserted/presented
/// signature/verified distinctions from the evidence layer. A `Verified`
/// origin is only constructible through that layer's verifier boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObservationEnvelope {
    kind: ObservationKind,
    source: SourceDescriptor,
    stream_id: Option<NonZeroU64>,
    generation: Option<NonZeroU32>,
    sequence: Option<u64>,
    timestamp: Option<ClockStamp>,
    time_anchor: ObservationTimeAnchor,
    quantity_unit: QuantityUnitIds,
    uncertainty: MeasurementUncertainty,
    quality: QualityRecord,
    calibration: CalibrationRecord,
    evidence: EvidenceOrigin,
    payload_digest: Option<ArtifactDigest>,
    policy: HostPolicyDecision,
}

impl ObservationEnvelope {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: ObservationKind,
        source: SourceDescriptor,
        stream_id: Option<u64>,
        generation: Option<u32>,
        sequence: Option<u64>,
        timestamp: Option<ClockStamp>,
        quantity_unit: QuantityUnitIds,
        uncertainty: MeasurementUncertainty,
        quality: QualityRecord,
        calibration: CalibrationRecord,
        evidence: EvidenceOrigin,
        payload_digest: Option<ArtifactDigest>,
        policy: HostPolicyDecision,
    ) -> Result<Self, ContractError> {
        let stream_id = nonzero(stream_id)?;
        let generation = match generation {
            Some(value) => NonZeroU32::new(value).map(Some).ok_or(ContractError::InvalidIdentifier)?,
            None => None,
        };
        if stream_id.is_none() && (generation.is_some() || sequence.is_some()) {
            return Err(ContractError::InvalidSource);
        }
        let source_kind_matches = match kind {
            ObservationKind::LiveFrame => source.kind() == crate::quality::SourceKind::Live,
            ObservationKind::RecordedFrame => source.kind() == crate::quality::SourceKind::Recorded,
            ObservationKind::SyntheticFrame => source.kind() == crate::quality::SourceKind::Synthetic,
            ObservationKind::CachedQuery | ObservationKind::HostDerived => true,
        };
        if !source_kind_matches {
            return Err(ContractError::InvalidSource);
        }
        validate_uncertainty_unit(uncertainty, quantity_unit)?;
        if !evidence_source_matches(evidence, source) {
            return Err(ContractError::InvalidSource);
        }
        Ok(Self {
            kind, source, stream_id, generation, sequence, timestamp, time_anchor: ObservationTimeAnchor::Unknown, quantity_unit,
            uncertainty, quality, calibration, evidence, payload_digest, policy,
        })
    }

    pub const fn kind(self) -> ObservationKind { self.kind }
    pub const fn source(self) -> SourceDescriptor { self.source }
    pub const fn stream_id(self) -> Option<NonZeroU64> { self.stream_id }
    pub const fn generation(self) -> Option<NonZeroU32> { self.generation }
    pub const fn sequence(self) -> Option<u64> { self.sequence }
    pub const fn timestamp(self) -> Option<ClockStamp> { self.timestamp }
    pub const fn time_anchor(self) -> ObservationTimeAnchor { self.time_anchor }

    /// Attach declared timestamp meaning; a missing timestamp has no known
    /// temporal anchor. This does not authenticate the host's declaration.
    pub fn with_time_anchor(mut self, anchor: ObservationTimeAnchor) -> Result<Self, ContractError> {
        if self.timestamp.is_none() && anchor != ObservationTimeAnchor::Unknown {
            return Err(ContractError::InvalidTime);
        }
        self.time_anchor = anchor;
        Ok(self)
    }
    pub const fn quantity_unit(self) -> QuantityUnitIds { self.quantity_unit }
    pub const fn uncertainty(self) -> MeasurementUncertainty { self.uncertainty }
    pub const fn quality(self) -> QualityRecord { self.quality }
    pub const fn calibration(self) -> CalibrationRecord { self.calibration }
    pub const fn evidence(self) -> EvidenceOrigin { self.evidence }
    pub const fn payload_digest(self) -> Option<ArtifactDigest> { self.payload_digest }
    pub const fn policy(self) -> HostPolicyDecision { self.policy }
}

fn validate_uncertainty_unit(
    uncertainty: MeasurementUncertainty,
    quantity_unit: QuantityUnitIds,
) -> Result<(), ContractError> {
    match uncertainty {
        MeasurementUncertainty::AbsoluteBound { magnitude, .. }
            if !magnitude.is_finite() || magnitude < 0.0 => return Err(ContractError::InvalidQuality),
        MeasurementUncertainty::RelativeBound { fraction }
            if !fraction.is_finite() || fraction < 0.0 => return Err(ContractError::InvalidQuality),
        MeasurementUncertainty::Interval { lower, upper, .. }
            if !lower.is_finite() || !upper.is_finite() || lower > upper => return Err(ContractError::InvalidQuality),
        _ => {}
    }
    let uncertainty_unit = match uncertainty {
        MeasurementUncertainty::AbsoluteBound { unit_id, .. }
        | MeasurementUncertainty::Interval { unit_id, .. } => unit_id,
        MeasurementUncertainty::Unknown | MeasurementUncertainty::RelativeBound { .. } => return Ok(()),
    };
    match (uncertainty_unit, quantity_unit.unit_id()) {
        (Some(uncertainty_unit), Some(declared_unit)) if uncertainty_unit == declared_unit => Ok(()),
        _ => Err(ContractError::IncompatibleUnit),
    }
}

fn evidence_source_matches(evidence: EvidenceOrigin, source: SourceDescriptor) -> bool {
    match evidence {
        EvidenceOrigin::Unknown => true,
        EvidenceOrigin::Asserted { source: claimed, .. }
        | EvidenceOrigin::SignaturePresented { source: claimed, .. } => claimed == source,
        EvidenceOrigin::Verified(verified) => verified.source() == source,
    }
}

#[cfg(test)]
#[path = "observation_tests.rs"]
mod tests;
