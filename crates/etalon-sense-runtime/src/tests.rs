use super::*;
use etalon_sense_core::{
    buffer::{BufferToken, SampleRegion},
    capability::{Modality, SensorCapability},
    extended_capability::ExtendedCapability,
    ingest::{ExtendedIngestStream, IngestStream},
    quality::{CalibrationState, SourceDescriptor, SourceKind},
    quantity::{PhysicalQuantity, SampleEncoding, SampleRate, Unit},
};

use etalon_signal::{
    decode::ByteOrder,
    execution::{AllocationAdmission, DeviceId, ExecutionError, ExecutionTarget, RegisteredSignalProvider},
    SignalError,
};

struct UnknownClassProvider { id: DeviceId, calls: usize }

impl RegisteredSignalProvider for UnknownClassProvider {
    fn device_id(&self) -> DeviceId { self.id }

    fn i16_iq_power_into(
        &mut self,
        _bytes: &[u8],
        _order: ByteOrder,
        _scale: f32,
        _workspace: &mut [f32],
        _staged_output: &mut [f64],
    ) -> Result<usize, SignalError> {
        self.calls += 1;
        Ok(0)
    }
}

fn builtin_capability() -> SensorCapability {
    SensorCapability::new(
        Modality::Wifi,
        PhysicalQuantity::RfChannelResponse,
        Unit::Unitless,
        SampleEncoding::ComplexI16,
        [4, 0, 0, 0],
        Some(SampleRate::new(100.0).unwrap()),
    )
    .unwrap()
}

fn extended_capability() -> ExtendedCapability {
    ExtendedCapability::new(
        801,
        802,
        None,
        SampleEncoding::ComplexI16,
        [4, 0, 0, 0],
        Some(SampleRate::new(20.0).unwrap()),
    )
    .unwrap()
}

fn layout(stride: u64) -> SampleLayout {
    SampleLayout::new(1, [2, 0, 0, 0], [stride, 0, 0, 0], SampleEncoding::ComplexI16).unwrap()
}

fn stamp(ticks: u64) -> ClockStamp { ClockStamp::new(11, 2, ticks, 1).unwrap() }

fn calibration() -> CalibrationRecord { CalibrationRecord::new(CalibrationState::Missing).unwrap() }

fn region(id: u64, span: u64) -> SampleRegion {
    SampleRegion::new(BufferToken::new(id, 1).unwrap(), 0, span, span, 2).unwrap()
}

fn source(id: u64) -> SourceDescriptor { SourceDescriptor::new(id, SourceKind::Live).unwrap() }

fn built_in_frame(payload: &[u8], stride: u64) -> IngestedFrame<'_> {
    let capability = builtin_capability();
    let layout = layout(stride);
    let span = layout.required_byte_span().unwrap();
    let mut stream = IngestStream::new(71, 72, 3).unwrap();
    stream
        .ingest(
            payload,
            layout,
            region(73, span),
            source(71),
            capability,
            stamp(1_000),
            None,
            calibration(),
            QualityRecord::Unknown,
            4,
        )
        .unwrap()
}

fn untimed_builtin_frame(payload: &[u8]) -> IngestedFrame<'_> {
    let capability = builtin_capability();
    let layout = layout(4);
    let span = layout.required_byte_span().unwrap();
    let mut stream = IngestStream::new(71, 72, 3).unwrap();
    stream.ingest_with_optional_timestamp(
        payload,
        layout,
        region(73, span),
        source(71),
        capability,
        None,
        None,
        calibration(),
        QualityRecord::Unknown,
        4,
    ).unwrap()
}

fn extended_frame(payload: &[u8]) -> ExtendedIngestedFrame<'_> {
    let capability = extended_capability();
    let layout = layout(4);
    let span = layout.required_byte_span().unwrap();
    let mut stream = ExtendedIngestStream::new(81, 82, 5, capability).unwrap();
    stream
        .ingest(
            payload,
            layout,
            region(83, span),
            source(81),
            stamp(2_000),
            None,
            calibration(),
            QualityRecord::Unknown,
            9,
        )
        .unwrap()
}

const IQ_BYTES: [u8; 8] = [2, 0, 4, 0, 0xFE, 0xFF, 4, 0];

#[test]
fn builtin_cpu_route_computes_known_power_and_preserves_unverified_metadata() {
    let frame = built_in_frame(&IQ_BYTES, 4);
    assert_eq!(frame.sequence_event(), SequenceEvent::First);

    let mut workspace = [0.0f32; 4];
    let mut staging = [0.0f64; 2];
    let mut output = [-1.0f64; 2];
    let summary = power_for_frame(
        &frame,
        ExecutionTarget::Cpu,
        None,
        ByteOrder::LittleEndian,
        0.5,
        &mut workspace,
        &mut staging,
        &mut output,
    )
    .unwrap();

    assert_eq!(output, [5.0, 5.0]);
    assert_eq!(workspace, [1.0, 2.0, -1.0, 2.0]);
    assert_eq!(summary.sample_count, 2);
    assert_eq!(summary.source_id, 71);
    assert_eq!(summary.stream_id, 72);
    assert_eq!(summary.generation, 3);
    assert_eq!(summary.sequence, 4);
    assert_eq!(summary.sample_rate, None);
    assert_eq!(summary.region.token(), BufferToken::new(73, 1).unwrap());
    assert_eq!(summary.layout, layout(4));
    assert_eq!(summary.source_kind, SourceKind::Live);
    assert_eq!(summary.source_trust, TrustState::Unverified);
    assert_eq!(summary.input_semantics, InputSemantics::BuiltIn {
        modality: Modality::Wifi,
        quantity: PhysicalQuantity::RfChannelResponse,
        unit: Unit::Unitless,
        encoding: SampleEncoding::ComplexI16,
    });
    assert_eq!(summary.quality, QualityRecord::Unknown);
    assert_eq!(summary.calibration.state(), CalibrationState::Missing);
    assert_eq!(summary.timestamp, Some(stamp(1_000)));
    assert_eq!(summary.sequence, 4);
    assert_eq!(summary.sequence_event, SequenceEvent::First);
    assert_eq!(summary.target, ExecutionTarget::Cpu);
    assert_eq!(summary.scale, 0.5);
}

#[test]
fn runtime_summary_preserves_an_unknown_timestamp() {
    let frame = untimed_builtin_frame(&IQ_BYTES);
    let mut workspace = [0.0f32; 4];
    let mut staging = [0.0f64; 2];
    let mut output = [0.0f64; 2];
    let summary = power_for_frame(
        &frame,
        ExecutionTarget::Cpu,
        None,
        ByteOrder::LittleEndian,
        1.0,
        &mut workspace,
        &mut staging,
        &mut output,
    ).unwrap();
    assert_eq!(summary.timestamp, None);
    assert_eq!(summary.sequence, 4);
}

#[test]
fn padded_layout_is_rejected_before_output_changes() {
    let padded_payload = [2, 0, 4, 0, 0, 0, 0, 0, 0xFE, 0xFF, 4, 0];
    let frame = built_in_frame(&padded_payload, 8);
    let mut workspace = [0.0f32; 6];
    let mut staging = [0.0f64; 2];
    let mut output = [77.0f64; 2];

    assert_eq!(
        power_for_frame(
            &frame,
            ExecutionTarget::Cpu,
            None,
            ByteOrder::LittleEndian,
            1.0,
            &mut workspace,
            &mut staging,
            &mut output,
        ),
        Err(PipelineError::UnsupportedLayout),
    );
    assert_eq!(output, [77.0; 2]);
}

#[test]
fn strict_runtime_admission_rejects_unknown_provider_before_dispatch_and_output_write() {
    let frame = built_in_frame(&IQ_BYTES, 4);
    let id = DeviceId::new(99).unwrap();
    let mut provider = UnknownClassProvider { id, calls: 0 };
    let mut workspace = [0.0f32; 4];
    let mut staging = [11.0f64; 2];
    let mut output = [77.0f64; 2];
    assert_eq!(
        power_for_frame_with_admission(
            &frame,
            ExecutionTarget::RegisteredDevice(id),
            Some(&mut provider),
            ByteOrder::LittleEndian,
            0.5,
            &mut workspace,
            &mut staging,
            &mut output,
            AllocationAdmission::RequireHotZeroHeap,
        ),
        Err(PipelineError::Execution(ExecutionError::AllocationClassRejected)),
    );
    assert_eq!(provider.calls, 0);
    assert_eq!(staging, [11.0; 2]);
    assert_eq!(output, [77.0; 2]);
}

#[test]
fn registered_extended_cpu_route_retains_unknown_unit_and_source_trust() {
    let frame = extended_frame(&IQ_BYTES);
    assert_eq!(frame.measurement().capability().unit_id(), None);
    assert_eq!(frame.measurement().source().trust_state(), TrustState::Unverified);
    let mut workspace = [0.0f32; 4];
    let mut staging = [0.0f64; 2];
    let mut output = [0.0f64; 2];
    let summary = power_for_extended_frame(
        &frame,
        ExecutionTarget::Cpu,
        None,
        ByteOrder::LittleEndian,
        0.5,
        &mut workspace,
        &mut staging,
        &mut output,
    )
    .unwrap();

    assert_eq!(output, [5.0, 5.0]);
    assert_eq!(summary.source_id, 81);
    assert_eq!(summary.stream_id, 82);
    assert_eq!(summary.generation, 5);
    assert_eq!(summary.sequence, 9);
    assert_eq!(summary.sample_rate, None);
    assert_eq!(summary.region.token(), BufferToken::new(83, 1).unwrap());
    assert_eq!(summary.layout, layout(4));
    assert_eq!(summary.source_kind, SourceKind::Live);
    assert_eq!(summary.source_trust, TrustState::Unverified);
    assert_eq!(summary.input_semantics, InputSemantics::Registered {
        modality_id: 801,
        quantity_id: 802,
        unit_id: None,
        encoding: SampleEncoding::ComplexI16,
    });
    assert_eq!(summary.quality, QualityRecord::Unknown);
    assert_eq!(summary.calibration.state(), CalibrationState::Missing);
    assert_eq!(summary.sequence_event, SequenceEvent::First);
}

#[test]
fn power_projection_preserves_sequence_and_keeps_host_declarations_unverified() {
    let frame = built_in_frame(&IQ_BYTES, 4);
    let mut workspace = [0.0f32; 4];
    let mut staging = [0.0f64; 2];
    let mut output = [0.0f64; 2];
    let summary = power_for_frame(
        &frame,
        ExecutionTarget::Cpu,
        None,
        ByteOrder::LittleEndian,
        0.5,
        &mut workspace,
        &mut staging,
        &mut output,
    ).unwrap();
    let quantity_unit = QuantityUnitIds::new(Some(900), Some(901)).unwrap();
    let digest = ArtifactDigest::new(
        etalon_sense_core::evidence::DigestAlgorithm::Sha256,
        [0xA5; 32],
    );
    let projection = project_power_summary_to_observation(
        &summary, quantity_unit, 902, Some(903), Some(digest),
    ).unwrap();
    let observation = projection.observation();
    assert_eq!(observation.kind(), ObservationKind::HostDerived);
    assert_eq!(observation.source().source_id(), 71);
    assert_eq!(observation.source().trust_state(), TrustState::Unverified);
    assert_eq!(observation.stream_id().map(|value| value.get()), Some(72));
    assert_eq!(observation.generation().map(|value| value.get()), Some(3));
    assert_eq!(observation.sequence(), Some(4));
    assert_eq!(observation.timestamp(), Some(stamp(1_000)));
    assert_eq!(observation.quantity_unit(), quantity_unit);
    assert_eq!(observation.uncertainty(), MeasurementUncertainty::Unknown);
    assert_eq!(observation.quality(), QualityRecord::Unknown);
    assert_eq!(observation.calibration().state(), CalibrationState::Missing);
    assert_eq!(observation.evidence(), EvidenceOrigin::Unknown);
    assert_eq!(observation.payload_digest(), None);
    assert_eq!(observation.policy(), HostPolicyDecision::NotEvaluated);
    assert_eq!(projection.processing().algorithm_id(), 902);
    assert_eq!(projection.processing().provider_id(), Some(903));
    assert_eq!(projection.processing().configuration_digest(), Some(digest));
}

#[test]
fn unknown_scale_and_workspace_errors_do_not_publish_output() {
    let frame = built_in_frame(&IQ_BYTES, 4);
    let mut workspace = [0.0f32; 4];
    let mut staging = [0.0f64; 2];
    let mut output = [91.0f64; 2];
    assert_eq!(
        power_for_frame(
            &frame,
            ExecutionTarget::Cpu,
            None,
            ByteOrder::LittleEndian,
            f32::NAN,
            &mut workspace,
            &mut staging,
            &mut output,
        ),
        Err(PipelineError::Execution(ExecutionError::Signal(
            etalon_signal::SignalError::InvalidParameter
        ))),
    );
    assert_eq!(output, [91.0; 2]);

    assert_eq!(
        power_for_frame(
            &frame,
            ExecutionTarget::Cpu,
            None,
            ByteOrder::LittleEndian,
            0.5,
            &mut workspace[..3],
            &mut staging,
            &mut output,
        ),
        Err(PipelineError::Execution(ExecutionError::Capacity)),
    );
    assert_eq!(output, [91.0; 2]);
}

