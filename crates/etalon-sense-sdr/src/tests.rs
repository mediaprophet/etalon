use super::*;
use etalon_signal::{execution::DeviceId, SignalError};

struct UnknownClassProvider {
    id: DeviceId,
    calls: usize,
}

impl RegisteredSignalProvider for UnknownClassProvider {
    fn device_id(&self) -> DeviceId {
        self.id
    }

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

struct TestReader {
    id: u64,
    report: SdrReadReport,
    calls: usize,
}

impl SdrReader for TestReader {
    fn reader_id(&self) -> u64 {
        self.id
    }

    fn read_i16_iq(
        &mut self,
        _request: SdrReadRequest,
        cancelled: &core::sync::atomic::AtomicBool,
        output: &mut [u8],
    ) -> Result<SdrReadReport, ReaderError> {
        self.calls += 1;
        if cancelled.load(core::sync::atomic::Ordering::Acquire) {
            return Err(ReaderError::Cancelled);
        }
        for (index, byte) in output.iter_mut().enumerate() {
            *byte = index as u8;
        }
        Ok(self.report)
    }
}

fn request(reader_id: u64) -> SdrReadRequest {
    let rate = SampleRateRatio::new(2_000_000, 1).unwrap();
    let tune = RfTuning::new(915_000_000, rate, Some(1_500_000)).unwrap();
    SdrReadRequest::new(reader_id, tune, 2, IqByteOrder::LittleEndianIq, 100_000_000).unwrap()
}

fn report(samples: usize, bytes: usize, overrun: OverrunReport) -> SdrReadReport {
    let timestamp = ClockStamp::new(77, 3, 10_000, 25).unwrap();
    SdrReadReport::new(
        bytes,
        samples,
        Some(timestamp),
        TimestampAnchor::FirstSample,
        AppliedTuning::Reported(request(1).tuning()),
        overrun,
    )
    .unwrap()
}

#[test]
fn successful_read_borrows_exact_caller_buffer_and_stays_unverified() {
    let mut reader = TestReader {
        id: 1,
        report: report(2, 8, OverrunReport::NoneReported),
        calls: 0,
    };
    let mut output = [0u8; 8];
    let frame = acquire_into(
        &mut reader,
        request(1),
        &core::sync::atomic::AtomicBool::new(false),
        &mut output,
    )
    .unwrap();
    assert_eq!(frame.bytes(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(frame.sample_count(), 2);
    assert_eq!(
        frame.sample_layout().unwrap().encoding(),
        &SampleEncoding::ComplexI16
    );
    assert_eq!(frame.sample_layout().unwrap().extents(), &[2, 0, 0, 0]);
    assert_eq!(frame.requested_tuning().center_frequency_hz(), 915_000_000);
    assert_eq!(
        frame.applied_tuning(),
        AppliedTuning::Reported(request(1).tuning())
    );
    assert_eq!(frame.overrun(), OverrunReport::NoneReported);
    assert_eq!(frame.trust(), CaptureTrust::HostReportedUnverified);
    assert_eq!(reader.calls, 1);
}

#[test]
fn preflight_rejects_mismatched_reader_and_buffer_without_callback() {
    let mut reader = TestReader {
        id: 2,
        report: report(2, 8, OverrunReport::Unknown),
        calls: 0,
    };
    let mut output = [0u8; 8];
    let cancelled = core::sync::atomic::AtomicBool::new(false);
    assert_eq!(
        acquire_into(&mut reader, request(1), &cancelled, &mut output).err(),
        Some(SdrError::ReaderMismatch)
    );
    assert_eq!(
        acquire_into(&mut reader, request(2), &cancelled, &mut output[..4]).err(),
        Some(SdrError::BufferLength)
    );
    assert_eq!(reader.calls, 0);
}

#[test]
fn incomplete_callback_output_is_rejected_and_cannot_be_exposed_as_a_frame() {
    let mut reader = TestReader {
        id: 1,
        report: report(1, 4, OverrunReport::Unknown),
        calls: 0,
    };
    let mut output = [0u8; 8];
    assert_eq!(
        acquire_into(
            &mut reader,
            request(1),
            &core::sync::atomic::AtomicBool::new(false),
            &mut output
        )
        .err(),
        Some(SdrError::IncompleteRead)
    );
    assert_eq!(reader.calls, 1);
    // On error the caller owns the buffer and must discard its contents.
    assert_eq!(output, [0; 8]);
}

#[test]
fn reported_overrun_is_preserved_without_claiming_authenticity() {
    let mut reader = TestReader {
        id: 1,
        report: report(2, 8, OverrunReport::LostSamples(9)),
        calls: 0,
    };
    let mut output = [0u8; 8];
    let frame = acquire_into(
        &mut reader,
        request(1),
        &core::sync::atomic::AtomicBool::new(false),
        &mut output,
    )
    .unwrap();
    assert_eq!(frame.overrun(), OverrunReport::LostSamples(9));
    assert_eq!(frame.trust(), CaptureTrust::HostReportedUnverified);
}

#[test]
fn invalid_tuning_and_zero_overrun_count_are_rejected() {
    assert_eq!(SampleRateRatio::new(0, 1), Err(SdrError::InvalidTuning));
    let rate = SampleRateRatio::new(1, 1).unwrap();
    assert_eq!(RfTuning::new(0, rate, None), Err(SdrError::InvalidTuning));
    assert_eq!(
        SdrReadReport::new(
            8,
            2,
            Some(ClockStamp::new(1, 1, 1, 0).unwrap()),
            TimestampAnchor::Unknown,
            AppliedTuning::Unknown,
            OverrunReport::LostSamples(0),
        ),
        Err(SdrError::InvalidReport),
    );
}

#[test]
fn absent_timestamp_requires_unknown_anchor() {
    assert!(SdrReadReport::new(
        8,
        2,
        None,
        TimestampAnchor::Unknown,
        AppliedTuning::Unknown,
        OverrunReport::Unknown,
    )
    .is_ok());
    assert_eq!(
        SdrReadReport::new(
            8,
            2,
            None,
            TimestampAnchor::FirstSample,
            AppliedTuning::Unknown,
            OverrunReport::Unknown,
        ),
        Err(SdrError::InvalidReport),
    );
}

#[test]
fn cancelled_read_is_rejected_before_reader_call_and_clears_buffer() {
    let mut reader = TestReader {
        id: 1,
        report: report(2, 8, OverrunReport::Unknown),
        calls: 0,
    };
    let cancelled = core::sync::atomic::AtomicBool::new(true);
    let mut output = [0xFFu8; 8];
    assert_eq!(
        acquire_into(&mut reader, request(1), &cancelled, &mut output),
        Err(SdrError::Cancelled),
    );
    assert_eq!(reader.calls, 0);
    assert_eq!(output, [0; 8]);
}

#[test]
fn request_requires_a_nonzero_read_duration_budget() {
    let rate = SampleRateRatio::new(1, 1).unwrap();
    let tune = RfTuning::new(1, rate, None).unwrap();
    assert_eq!(
        SdrReadRequest::new(1, tune, 1, IqByteOrder::LittleEndianIq, 0),
        Err(SdrError::InvalidReadBudget),
    );
}

struct FailingReader;

impl SdrReader for FailingReader {
    fn reader_id(&self) -> u64 {
        1
    }

    fn read_i16_iq(
        &mut self,
        _request: SdrReadRequest,
        _cancelled: &core::sync::atomic::AtomicBool,
        output: &mut [u8],
    ) -> Result<SdrReadReport, ReaderError> {
        output.fill(0xA5);
        Err(ReaderError::Io)
    }
}

#[test]
fn reader_error_clears_partially_written_output() {
    let mut reader = FailingReader;
    let mut output = [0u8; 8];
    assert_eq!(
        acquire_into(
            &mut reader,
            request(1),
            &core::sync::atomic::AtomicBool::new(false),
            &mut output
        ),
        Err(SdrError::Reader(ReaderError::Io))
    );
    assert_eq!(output, [0; 8]);
}

#[test]
fn iq_power_pipeline_preserves_capture_declarations() {
    let payload = [3u8, 0, 4, 0];
    let tune = request(7).tuning();
    let frame = SdrFrame {
        reader_id: 7,
        requested_tuning: tune,
        applied_tuning: AppliedTuning::Reported(tune),
        sample_count: 1,
        byte_order: IqByteOrder::LittleEndianIq,
        timestamp: None,
        timestamp_anchor: TimestampAnchor::Unknown,
        overrun: OverrunReport::LostSamples(2),
        trust: CaptureTrust::HostReportedUnverified,
        bytes: &payload,
        stream_id: None,
        generation: None,
        sequence: None,
    };
    let mut workspace = [0.0f32; 2];
    let mut staged = [0.0f64; 1];
    let mut output = [0.0f64; 1];
    let powered = compute_iq_power_into(
        frame,
        ExecutionTarget::Cpu,
        None,
        1.0,
        &mut workspace,
        &mut staged,
        &mut output,
    )
    .unwrap();
    assert_eq!(powered.power_values(), &[25.0]);
    assert_eq!(powered.reader_id(), 7);
    assert_eq!(powered.requested_tuning(), tune);
    assert_eq!(powered.applied_tuning(), AppliedTuning::Reported(tune));
    assert_eq!(powered.timestamp(), None);
    assert_eq!(powered.timestamp_anchor(), TimestampAnchor::Unknown);
    assert_eq!(powered.overrun(), OverrunReport::LostSamples(2));
    assert_eq!(powered.trust(), CaptureTrust::HostReportedUnverified);
}

#[test]
fn strict_power_admission_rejects_unknown_provider_before_dispatch_and_output_write() {
    let payload = [3u8, 0, 4, 0];
    let tune = request(7).tuning();
    let frame = SdrFrame {
        reader_id: 7,
        requested_tuning: tune,
        applied_tuning: AppliedTuning::Reported(tune),
        sample_count: 1,
        byte_order: IqByteOrder::LittleEndianIq,
        timestamp: None,
        timestamp_anchor: TimestampAnchor::Unknown,
        overrun: OverrunReport::Unknown,
        trust: CaptureTrust::HostReportedUnverified,
        bytes: &payload,
        stream_id: None,
        generation: None,
        sequence: None,
    };
    let id = DeviceId::new(7).unwrap();
    let mut provider = UnknownClassProvider { id, calls: 0 };
    let mut workspace = [0.0f32; 2];
    let mut staged = [11.0f64; 1];
    let mut output = [77.0f64; 1];
    assert_eq!(
        compute_iq_power_into_with_admission(
            frame,
            ExecutionTarget::RegisteredDevice(id),
            Some(&mut provider),
            1.0,
            &mut workspace,
            &mut staged,
            &mut output,
            AllocationAdmission::RequireHotZeroHeap,
        ),
        Err(SdrError::SignalExecution(
            ExecutionError::AllocationClassRejected
        )),
    );
    assert_eq!(provider.calls, 0);
    assert_eq!(staged, [11.0]);
    assert_eq!(output, [77.0]);
}
