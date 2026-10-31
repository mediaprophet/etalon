use super::*;
use crate::*;
use core::sync::atomic::{AtomicBool, Ordering};
use etalon_sense_core::quality::TrustState;

struct Reader {
    id: u64,
    timestamp: Option<etalon_sense_core::time::ClockStamp>,
    overrun: OverrunReport,
    one_sample: [u8; 4],
    stream_position: (Option<u64>, Option<u32>, Option<u64>),
}

impl SdrReader for Reader {
    fn reader_id(&self) -> u64 { self.id }

    fn read_i16_iq(
        &mut self,
        request: SdrReadRequest,
        cancelled: &AtomicBool,
        output: &mut [u8],
    ) -> Result<SdrReadReport, ReaderError> {
        if cancelled.load(Ordering::Acquire) { return Err(ReaderError::Cancelled); }
        output.fill(0);
        if output.len() == self.one_sample.len() { output.copy_from_slice(&self.one_sample); }
        let tuning = request.tuning();
        SdrReadReport::new(
            output.len(),
            request.sample_count(),
            self.timestamp,
            if self.timestamp.is_some() { TimestampAnchor::FirstSample } else { TimestampAnchor::Unknown },
            AppliedTuning::Reported(tuning),
            self.overrun,
        ).and_then(|report| report.with_stream_position(
            self.stream_position.0, self.stream_position.1, self.stream_position.2,
        )).map_err(|_| ReaderError::Io)
    }
}

fn request(reader_id: u64, samples: usize) -> SdrReadRequest {
    let rate = SampleRateRatio::new(2_000_000, 1).unwrap();
    let tuning = RfTuning::new(915_000_000, rate, Some(1_500_000)).unwrap();
    SdrReadRequest::new(reader_id, tuning, samples, IqByteOrder::LittleEndianIq, 10_000).unwrap()
}

fn mapping(kind: ObservationKind, semantics: QuantityUnitIds) -> SdrObservationMapping {
    SdrObservationMapping {
        source_kind: SourceKind::Live,
        observation_kind: kind,
        quantity_unit: semantics,
    }
}

#[test]
fn iq_frame_preserves_reader_stream_tuning_and_unknowns() {
    let stamp = etalon_sense_core::time::ClockStamp::new(8, 2, 99, 1).unwrap();
    let mut reader = Reader {
        id: 17, timestamp: Some(stamp), overrun: OverrunReport::LostSamples(4),
        one_sample: [0; 4], stream_position: (Some(44), Some(3), Some(12)),
    };
    let mut bytes = [0u8; 8];
    let frame = acquire_into(&mut reader, request(17, 2), &AtomicBool::new(false), &mut bytes).unwrap();
    let mut output = None;
    let semantics = QuantityUnitIds::new(Some(301), Some(302)).unwrap();
    adapt_sdr_frame(frame, mapping(ObservationKind::LiveFrame, semantics), &mut output).unwrap();

    let observation = output.unwrap();
    assert_eq!(observation.envelope.source().source_id(), 17);
    assert_eq!(observation.envelope.source().trust_state(), TrustState::Unverified);
    assert_eq!(observation.envelope.stream_id().unwrap().get(), 44);
    assert_eq!(observation.envelope.generation().unwrap().get(), 3);
    assert_eq!(observation.envelope.sequence(), Some(12));
    assert_eq!(observation.envelope.timestamp(), Some(stamp));
    assert_eq!(observation.envelope.quality(), QualityRecord::Unknown);
    assert_eq!(observation.envelope.evidence(), EvidenceOrigin::Unknown);
    assert_eq!(observation.envelope.policy(), HostPolicyDecision::NotEvaluated);
    assert_eq!(observation.metadata.reader_id, 17);
    assert_eq!(observation.metadata.sample_count, 2);
    assert_eq!(observation.metadata.requested_tuning.center_frequency_hz(), 915_000_000);
    assert_eq!(observation.metadata.applied_tuning, AppliedTuning::Reported(request(17, 2).tuning()));
    assert_eq!(observation.metadata.overrun, OverrunReport::LostSamples(4));
    assert_eq!(observation.metadata.timestamp_anchor, TimestampAnchor::FirstSample);
    assert!(matches!(observation.payload, SdrPayload::InterleavedI16Iq(data) if data == &[0; 8]));
}

#[test]
fn power_frame_keeps_absent_time_and_does_not_infer_a_unit() {
    let mut reader = Reader {
        id: 19, timestamp: None, overrun: OverrunReport::Unknown,
        one_sample: [3, 0, 4, 0], stream_position: (None, None, None),
    };
    let mut bytes = [0u8; 4];
    let frame = acquire_into(&mut reader, request(19, 1), &AtomicBool::new(false), &mut bytes).unwrap();
    let mut workspace = [0.0f32; 2];
    let mut staged = [0.0f64; 1];
    let mut values = [0.0f64; 1];
    let powered = compute_iq_power_into(
        frame, ExecutionTarget::Cpu, None, 1.0,
        &mut workspace, &mut staged, &mut values,
    ).unwrap();
    let mut output = None;
    adapt_sdr_power_frame(
        powered,
        mapping(ObservationKind::HostDerived, QuantityUnitIds::new(Some(401), None).unwrap()),
        &mut output,
    ).unwrap();

    let observation = output.unwrap();
    assert_eq!(observation.envelope.timestamp(), None);
    assert_eq!(observation.envelope.quantity_unit().quantity_id().unwrap().get(), 401);
    assert_eq!(observation.envelope.quantity_unit().unit_id(), None);
    assert_eq!(observation.envelope.stream_id(), None);
    assert_eq!(observation.envelope.generation(), None);
    assert_eq!(observation.envelope.sequence(), None);
    assert_eq!(observation.metadata.sample_count, 1);
    assert_eq!(observation.metadata.timestamp_anchor, TimestampAnchor::Unknown);
    assert_eq!(observation.metadata.trust, CaptureTrust::HostReportedUnverified);
    assert!(matches!(observation.payload, SdrPayload::IqPowerF64(data) if data == &[25.0]));
}

#[test]
fn absent_stream_metadata_remains_unknown_in_envelope() {
    let mut reader = Reader {
        id: 3, timestamp: None, overrun: OverrunReport::Unknown,
        one_sample: [0; 4], stream_position: (None, None, None),
    };
    let mut bytes = [0u8; 4];
    let frame = acquire_into(&mut reader, request(3, 1), &AtomicBool::new(false), &mut bytes).unwrap();
    let mut output = None;
    adapt_sdr_frame(frame, mapping(
        ObservationKind::LiveFrame,
        QuantityUnitIds::new(None, None).unwrap(),
    ), &mut output).unwrap();
    let envelope = output.unwrap().envelope;
    assert_eq!(envelope.stream_id(), None);
    assert_eq!(envelope.generation(), None);
    assert_eq!(envelope.sequence(), None);
}
