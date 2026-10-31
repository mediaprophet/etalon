//! Caller-declared semantic envelope for one validated camera frame.
//!
//! The envelope binds metadata declarations to the camera declaration and
//! negotiated frame. It contains no pixels and proves neither byte provenance,
//! device identity, calibration, nor authorization.

use etalon_sense_core::{
    evidence::EvidenceOrigin,
    observation::{
        HostPolicyDecision, MeasurementUncertainty, ObservationEnvelope as CoreObservationEnvelope,
        ObservationKind, ObservationTimeAnchor, QuantityUnitIds,
    },
    quality::{CalibrationRecord, CalibrationState, QualityRecord, SourceDescriptor, SourceKind},
    quantity::{PhysicalQuantity, Unit},
    time::ClockStamp,
};

use crate::{
    format::{
        validate_negotiated_format, CameraModality, FrameDeclaration, FrameFormat, FramePlane,
        RadiometricStatus,
    },
    FrameMetadata, FrameTimestamp, NegotiatedFrame,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationError {
    InvalidIdentifier,
    SourceMismatch,
    ModalityMismatch,
    QuantityMismatch,
    UnitMismatch,
    FormatMismatch,
    InvalidTimestamp,
    InvalidPolicyClaim,
    GenerationOutOfRange,
}

/// Caller-provided semantic mapping for the capability's opaque quantity/unit
/// identifiers. Numeric IDs are checked for exact association only; this crate
/// does not resolve a registry or authenticate the mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservationQuantityMapping {
    quantity_id: u64,
    unit_id: Option<u64>,
    physical_quantity: Option<PhysicalQuantity>,
    unit: Option<Unit>,
}

impl ObservationQuantityMapping {
    pub fn new(
        quantity_id: u64,
        unit_id: Option<u64>,
        physical_quantity: Option<PhysicalQuantity>,
        unit: Option<Unit>,
    ) -> Result<Self, ObservationError> {
        if quantity_id == 0 || matches!(unit_id, Some(0)) {
            return Err(ObservationError::InvalidIdentifier);
        }
        if unit.is_some() && physical_quantity.is_none() {
            return Err(ObservationError::UnitMismatch);
        }
        if let (Some(quantity), Some(unit)) = (physical_quantity, unit) {
            if !quantity.accepts_unit(unit) {
                return Err(ObservationError::UnitMismatch);
            }
        }
        Ok(Self {
            quantity_id,
            unit_id,
            physical_quantity,
            unit,
        })
    }

    pub const fn quantity_id(self) -> u64 {
        self.quantity_id
    }
    pub const fn unit_id(self) -> Option<u64> {
        self.unit_id
    }
    pub const fn physical_quantity(self) -> Option<PhysicalQuantity> {
        self.physical_quantity
    }
    pub const fn unit(self) -> Option<Unit> {
        self.unit
    }
}

/// Native timestamp as supplied by the host/provider, without unit conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeTimestamp {
    ticks: i64,
    clock_domain_id: u64,
    unit_id: u64,
}

impl NativeTimestamp {
    pub fn new(ticks: i64, clock_domain_id: u64, unit_id: u64) -> Result<Self, ObservationError> {
        if clock_domain_id == 0 || unit_id == 0 {
            return Err(ObservationError::InvalidIdentifier);
        }
        Ok(Self {
            ticks,
            clock_domain_id,
            unit_id,
        })
    }

    pub const fn ticks(self) -> i64 {
        self.ticks
    }
    pub const fn clock_domain_id(self) -> u64 {
        self.clock_domain_id
    }
    pub const fn unit_id(self) -> u64 {
        self.unit_id
    }
}

/// Optional caller declaration about an external policy evaluation. It is
/// carried as metadata only; it is neither evaluated nor treated as a permit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostPolicyClaim {
    policy_id: u64,
    decision_id: u64,
    context_id: u64,
    outcome: HostPolicyOutcome,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostPolicyOutcome {
    Permit,
    Deny,
    Unknown,
}

impl HostPolicyClaim {
    pub fn new(
        policy_id: u64,
        decision_id: u64,
        context_id: u64,
        outcome: HostPolicyOutcome,
    ) -> Result<Self, ObservationError> {
        if policy_id == 0 || decision_id == 0 || context_id == 0 {
            return Err(ObservationError::InvalidPolicyClaim);
        }
        Ok(Self {
            policy_id,
            decision_id,
            context_id,
            outcome,
        })
    }

    pub const fn policy_id(self) -> u64 {
        self.policy_id
    }
    pub const fn decision_id(self) -> u64 {
        self.decision_id
    }
    pub const fn context_id(self) -> u64 {
        self.context_id
    }
    pub const fn outcome(self) -> HostPolicyOutcome {
        self.outcome
    }
}

/// Metadata-only association for one frame. `stream_id` and `generation` are
/// explicit host identities; `sequence` and gap are preserved from the frame
/// provider. A mapped clock is present only when supplied as a valid
/// `ClockStamp`; raw native ticks are retained independently and never
/// converted here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservationEnvelope {
    source_id: u64,
    stream_id: u64,
    generation: u64,
    modality: CameraModality,
    quantity: ObservationQuantityMapping,
    pixel_format: crate::FramePixelFormat,
    format_id: u64,
    width: u32,
    height: u32,
    frame_bytes: u32,
    planes: [FramePlane; crate::MAX_FRAME_PLANES],
    plane_count: u8,
    radiometric: RadiometricStatus,
    sequence: u64,
    gap_before: Option<u64>,
    raw_timestamp: Option<NativeTimestamp>,
    mapped_clock: Option<ClockStamp>,
    policy_claim: Option<HostPolicyClaim>,
}

impl ObservationEnvelope {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        declaration: FrameDeclaration,
        negotiated: NegotiatedFrame,
        metadata: FrameMetadata,
        stream_id: u64,
        generation: u64,
        quantity: ObservationQuantityMapping,
        raw_timestamp: Option<NativeTimestamp>,
        mapped_clock: Option<ClockStamp>,
        policy_claim: Option<HostPolicyClaim>,
    ) -> Result<Self, ObservationError> {
        if stream_id == 0 || generation == 0 {
            return Err(ObservationError::InvalidIdentifier);
        }
        if negotiated.source_id != declaration.source_id() {
            return Err(ObservationError::SourceMismatch);
        }
        if negotiated.modality != declaration.modality() {
            return Err(ObservationError::ModalityMismatch);
        }
        let capability = declaration.capability();
        if quantity.quantity_id != capability.quantity_id() {
            return Err(ObservationError::QuantityMismatch);
        }
        if quantity.unit_id != capability.unit_id() {
            return Err(ObservationError::UnitMismatch);
        }
        validate_negotiated_format(declaration, negotiated.format, negotiated.radiometric)
            .map_err(|_| ObservationError::FormatMismatch)?;
        if metadata.format_id == 0 || metadata.format_id != negotiated.format.format_id() {
            return Err(ObservationError::FormatMismatch);
        }

        let (raw_timestamp, mapped_clock) =
            validate_timestamp_pair(metadata.timestamp, raw_timestamp, mapped_clock)?;

        let format: FrameFormat = negotiated.format;
        Ok(Self {
            source_id: negotiated.source_id,
            stream_id,
            generation,
            modality: negotiated.modality,
            quantity,
            pixel_format: format.pixel_format(),
            format_id: format.format_id(),
            width: format.width(),
            height: format.height(),
            frame_bytes: format.frame_bytes(),
            planes: format.planes(),
            plane_count: format.plane_count(),
            radiometric: negotiated.radiometric,
            sequence: metadata.sequence,
            gap_before: metadata.gap_before,
            raw_timestamp,
            mapped_clock,
            policy_claim,
        })
    }

    pub const fn source_id(self) -> u64 {
        self.source_id
    }
    pub const fn stream_id(self) -> u64 {
        self.stream_id
    }
    pub const fn generation(self) -> u64 {
        self.generation
    }
    pub const fn modality(self) -> CameraModality {
        self.modality
    }
    pub const fn quantity(self) -> ObservationQuantityMapping {
        self.quantity
    }
    pub const fn pixel_format(self) -> crate::FramePixelFormat {
        self.pixel_format
    }
    pub const fn format_id(self) -> u64 {
        self.format_id
    }
    pub const fn width(self) -> u32 {
        self.width
    }
    pub const fn height(self) -> u32 {
        self.height
    }
    pub const fn frame_bytes(self) -> u32 {
        self.frame_bytes
    }
    pub const fn planes(self) -> [FramePlane; crate::MAX_FRAME_PLANES] {
        self.planes
    }
    pub const fn plane_count(self) -> u8 {
        self.plane_count
    }
    pub const fn radiometric(self) -> RadiometricStatus {
        self.radiometric
    }
    pub const fn sequence(self) -> u64 {
        self.sequence
    }
    pub const fn gap_before(self) -> Option<u64> {
        self.gap_before
    }
    pub const fn raw_timestamp(self) -> Option<NativeTimestamp> {
        self.raw_timestamp
    }
    pub const fn mapped_clock(self) -> Option<ClockStamp> {
        self.mapped_clock
    }
    pub const fn policy_claim(self) -> Option<HostPolicyClaim> {
        self.policy_claim
    }

    /// Convert to the shared Etalon observation contract. Source kind and time
    /// anchor are explicit host inputs; other quality, calibration, evidence,
    /// and policy fields remain unknown/unverified because this camera adapter
    /// cannot establish them. The richer raw timestamp and plane metadata stay
    /// available on this camera envelope and are not folded into the core type.
    pub fn to_core_observation(
        self,
        source_kind: SourceKind,
        time_anchor: ObservationTimeAnchor,
    ) -> Result<CoreObservationEnvelope, ObservationError> {
        let generation =
            u32::try_from(self.generation).map_err(|_| ObservationError::GenerationOutOfRange)?;
        let source = SourceDescriptor::new(self.source_id, source_kind)
            .map_err(|_| ObservationError::InvalidIdentifier)?;
        let quantity_unit =
            QuantityUnitIds::new(Some(self.quantity.quantity_id), self.quantity.unit_id)
                .map_err(|_| ObservationError::UnitMismatch)?;
        let calibration = CalibrationRecord::new(CalibrationState::Missing)
            .map_err(|_| ObservationError::InvalidPolicyClaim)?;
        let kind = match source_kind {
            SourceKind::Live => ObservationKind::LiveFrame,
            SourceKind::Synthetic => ObservationKind::SyntheticFrame,
            SourceKind::Recorded => ObservationKind::RecordedFrame,
        };
        CoreObservationEnvelope::new(
            kind,
            source,
            Some(self.stream_id),
            Some(generation),
            Some(self.sequence),
            self.mapped_clock,
            quantity_unit,
            MeasurementUncertainty::Unknown,
            QualityRecord::unknown(),
            calibration,
            EvidenceOrigin::Unknown,
            None,
            HostPolicyDecision::NotEvaluated,
        )
        .and_then(|observation| observation.with_time_anchor(time_anchor))
        .map_err(|_| ObservationError::InvalidTimestamp)
    }
}

fn validate_timestamp_pair(
    timestamp: FrameTimestamp,
    raw: Option<NativeTimestamp>,
    mapped: Option<ClockStamp>,
) -> Result<(Option<NativeTimestamp>, Option<ClockStamp>), ObservationError> {
    match timestamp {
        FrameTimestamp::Unknown => Ok((raw, mapped)),
        FrameTimestamp::Raw {
            ticks,
            clock_domain_id,
            unit_id,
        } => {
            let declared = NativeTimestamp::new(ticks, clock_domain_id, unit_id)?;
            if matches!(raw, Some(other) if other != declared) {
                return Err(ObservationError::InvalidTimestamp);
            }
            Ok((Some(declared), mapped))
        }
        FrameTimestamp::Mapped(stamp) => {
            if matches!(mapped, Some(other) if other != stamp) {
                return Err(ObservationError::InvalidTimestamp);
            }
            Ok((raw, Some(stamp)))
        }
    }
}

#[cfg(test)]
#[path = "observation_tests.rs"]
mod tests;
