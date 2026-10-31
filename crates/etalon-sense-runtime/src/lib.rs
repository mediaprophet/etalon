//! Composition of sensing metadata and caller-buffered signal execution.
//! No database, device discovery, persistence, or policy authority is provided.
#![no_std]
#![forbid(unsafe_code)]

use etalon_sense_core::{
    buffer::{SampleLayout, SampleRegion},
    capability::{Modality, SensorCapability},
    evidence::{ArtifactDigest, EvidenceOrigin, ProcessingEvidence},
    extended_capability::ExtendedCapability,
    ingest::{ExtendedIngestedFrame, IngestedFrame},
    observation::{HostPolicyDecision, MeasurementUncertainty, ObservationEnvelope, ObservationKind, QuantityUnitIds},
    quality::{CalibrationRecord, CalibrationState, QualityRecord, SourceDescriptor, SourceKind, TrustState},
    quantity::{PhysicalQuantity, SampleEncoding, SampleRate, Unit},
    stream_order::SequenceEvent,
    time::ClockStamp,
    ContractError,
};
use etalon_signal::{decode::ByteOrder, execution::{execute_i16_iq_power_into_with_admission, ExecutionError, ExecutionTarget, RegisteredSignalProvider}};
pub use etalon_signal::execution::{AllocationAdmission, ExecutionAllocationClass};

/// Metadata returned alongside caller-owned power samples. It carries the
/// input's declarations; it does not authenticate them or attest execution.
#[derive(Clone, Debug, PartialEq)]
pub struct PowerSummary {
    pub source_id: u64,
    /// Caller-declared stream identity and generation; neither is authenticated.
    pub stream_id: u64,
    pub generation: u32,
    /// Exact source sequence admitted by ingest, even when time is unknown.
    pub sequence: u64,
    /// Caller-declared acquisition class; this is not source authentication.
    pub source_kind: SourceKind,
    /// Always `Unverified` until a separate authority integration exists.
    pub source_trust: TrustState,
    /// Declared input meaning is retained without treating IDs or units as
    /// authenticated and without implying a conversion to physical power.
    pub input_semantics: InputSemantics,
    pub timestamp: Option<ClockStamp>,
    /// Actual declared sample rate; `None` remains unknown.
    pub sample_rate: Option<SampleRate>,
    /// Caller-declared buffer placement and shape, not proof of memory origin.
    pub region: SampleRegion,
    pub layout: SampleLayout,
    pub calibration: CalibrationRecord,
    pub quality: QualityRecord,
    pub sequence_event: SequenceEvent,
    pub target: ExecutionTarget,
    pub scale: f32,
    pub sample_count: usize,
}

/// The input metadata domain is explicit: built-in enums never share an
/// integer namespace with host-registered IDs, and `None` remains unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputSemantics {
    BuiltIn {
        modality: Modality,
        quantity: PhysicalQuantity,
        unit: Unit,
        encoding: SampleEncoding,
    },
    Registered {
        modality_id: u64,
        quantity_id: u64,
        unit_id: Option<u64>,
        encoding: SampleEncoding,
    },
}

impl From<SensorCapability> for InputSemantics {
    fn from(capability: SensorCapability) -> Self {
        Self::BuiltIn {
            modality: capability.modality(),
            quantity: capability.quantity(),
            unit: capability.unit(),
            encoding: capability.encoding(),
        }
    }
}

impl From<ExtendedCapability> for InputSemantics {
    fn from(capability: ExtendedCapability) -> Self {
        Self::Registered {
            modality_id: capability.modality_id(),
            quantity_id: capability.quantity_id(),
            unit_id: capability.unit_id(),
            encoding: capability.encoding(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineError {
    UnsupportedLayout,
    Execution(ExecutionError),
}

/// Calculate uncalibrated I/Q power for a validated built-in measurement.
/// Only contiguous one-dimensional complex-i16 samples are accepted; byte
/// order and amplitude scale are explicit, and no normalization is inferred.
/// Results use the caller's scale-squared convention; they are not calibrated
/// physical power and this call does not create provenance evidence.
/// This compatibility entry point imposes no allocation admission and does
/// not imply a zero-heap provider; use the `_with_admission` variant for that.
#[allow(clippy::too_many_arguments)]
pub fn power_for_frame(frame: &IngestedFrame<'_>, target: ExecutionTarget, provider: Option<&mut dyn RegisteredSignalProvider>, order: ByteOrder, scale: f32, workspace: &mut [f32], staging: &mut [f64], output: &mut [f64]) -> Result<PowerSummary, PipelineError> {
    power_for_frame_with_admission(
        frame, target, provider, order, scale, workspace, staging, output,
        AllocationAdmission::Unrestricted,
    )
}

/// Same operation with caller-selected allocation admission. In particular,
/// `RequireHotZeroHeap` checks the provider's declared class before calling it;
/// declarations are not runtime allocation measurements or proof.
#[allow(clippy::too_many_arguments)]
pub fn power_for_frame_with_admission(
    frame: &IngestedFrame<'_>,
    target: ExecutionTarget,
    provider: Option<&mut dyn RegisteredSignalProvider>,
    order: ByteOrder,
    scale: f32,
    workspace: &mut [f32],
    staging: &mut [f64],
    output: &mut [f64],
    admission: AllocationAdmission,
) -> Result<PowerSummary, PipelineError> {
    let measurement = frame.measurement();
    let count = execute_with_admission(frame.payload(), measurement.layout(), target, provider, order, scale, workspace, staging, output, admission)?;
    Ok(PowerSummary { source_id: measurement.source().source_id(), stream_id: frame.stream_id(), generation: frame.generation(), sequence: frame.sequence(), source_kind: measurement.source().kind(), source_trust: measurement.source().trust_state(), input_semantics: (*measurement.capability()).into(), timestamp: measurement.timestamp(), sample_rate: measurement.sample_rate(), region: *measurement.region(), layout: measurement.layout().clone(), calibration: measurement.calibration(), quality: measurement.quality(), sequence_event: frame.sequence_event(), target, scale, sample_count: count })
}

/// The same uncalibrated operation for a host-registered modality and quantity.
/// The host is responsible for interpreting registry identifiers and output
/// units; source assertions remain unverified.
#[allow(clippy::too_many_arguments)]
pub fn power_for_extended_frame(frame: &ExtendedIngestedFrame<'_>, target: ExecutionTarget, provider: Option<&mut dyn RegisteredSignalProvider>, order: ByteOrder, scale: f32, workspace: &mut [f32], staging: &mut [f64], output: &mut [f64]) -> Result<PowerSummary, PipelineError> {
    power_for_extended_frame_with_admission(
        frame, target, provider, order, scale, workspace, staging, output,
        AllocationAdmission::Unrestricted,
    )
}

/// Admission-aware variant for host-registered measurement declarations.
#[allow(clippy::too_many_arguments)]
pub fn power_for_extended_frame_with_admission(
    frame: &ExtendedIngestedFrame<'_>,
    target: ExecutionTarget,
    provider: Option<&mut dyn RegisteredSignalProvider>,
    order: ByteOrder,
    scale: f32,
    workspace: &mut [f32],
    staging: &mut [f64],
    output: &mut [f64],
    admission: AllocationAdmission,
) -> Result<PowerSummary, PipelineError> {
    let measurement = frame.measurement();
    let count = execute_with_admission(frame.payload(), measurement.layout(), target, provider, order, scale, workspace, staging, output, admission)?;
    Ok(PowerSummary { source_id: measurement.source().source_id(), stream_id: frame.stream_id(), generation: frame.generation(), sequence: frame.sequence(), source_kind: measurement.source().kind(), source_trust: measurement.source().trust_state(), input_semantics: (*measurement.capability()).into(), timestamp: measurement.timestamp(), sample_rate: measurement.sample_rate(), region: *measurement.region(), layout: measurement.layout().clone(), calibration: measurement.calibration(), quality: measurement.quality(), sequence_event: frame.sequence_event(), target, scale, sample_count: count })
}

/// Host-declared projection from a processed power summary to the existing
/// observation envelope. Processing attribution is carried separately because
/// the envelope has no processing field. Neither declaration proves execution,
/// calibration, physical units, output bytes, or source authenticity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PowerObservationProjection {
    observation: ObservationEnvelope,
    processing: ProcessingEvidence,
}

impl PowerObservationProjection {
    pub const fn observation(self) -> ObservationEnvelope { self.observation }
    pub const fn processing(self) -> ProcessingEvidence { self.processing }
}

/// Project one power summary using host-supplied output quantity/unit IDs and
/// operation/provider/configuration declarations. Unknown timestamps remain
/// unknown. Output uncertainty, quality, calibration, payload digest, evidence,
/// and policy remain unknown or missing because this helper receives neither
/// output values nor independent measurement/error/calibration attestations.
pub fn project_power_summary_to_observation(
    summary: &PowerSummary,
    output_quantity_unit: QuantityUnitIds,
    operation_id: u64,
    provider_id: Option<u64>,
    configuration_digest: Option<ArtifactDigest>,
) -> Result<PowerObservationProjection, ContractError> {
    let source = SourceDescriptor::new(summary.source_id, summary.source_kind)?;
    let processing = ProcessingEvidence::new(operation_id, configuration_digest, provider_id)?;
    let calibration = CalibrationRecord::new(CalibrationState::Missing)?;
    let observation = ObservationEnvelope::new(
        ObservationKind::HostDerived,
        source,
        Some(summary.stream_id),
        Some(summary.generation),
        Some(summary.sequence),
        summary.timestamp,
        output_quantity_unit,
        MeasurementUncertainty::Unknown,
        QualityRecord::Unknown,
        calibration,
        EvidenceOrigin::unknown(),
        None,
        HostPolicyDecision::NotEvaluated,
    )?;
    Ok(PowerObservationProjection { observation, processing })
}

#[allow(clippy::too_many_arguments)]
fn execute_with_admission(payload: &[u8], layout: &SampleLayout, target: ExecutionTarget, provider: Option<&mut dyn RegisteredSignalProvider>, order: ByteOrder, scale: f32, workspace: &mut [f32], staging: &mut [f64], output: &mut [f64], admission: AllocationAdmission) -> Result<usize, PipelineError> {
    if layout.rank() != 1 || *layout.encoding() != SampleEncoding::ComplexI16 || layout.strides_bytes()[0] != 4 {
        return Err(PipelineError::UnsupportedLayout);
    }
    execute_i16_iq_power_into_with_admission(target, provider, payload, order, scale, workspace, staging, output, admission).map_err(PipelineError::Execution)
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
