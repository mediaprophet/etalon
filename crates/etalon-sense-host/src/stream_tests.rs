use std::{cell::Cell, io, io::Read, rc::Rc};

use etalon_sense_core::{
    adapter::{
        AdapterError, AdapterId, AuthorizedExtendedCaptureBuffer, ExtendedAdapterDescriptor,
        ExtendedCaptureBuffer, ExtendedCaptureSession, ExtendedDiscoveredSource,
        ExtendedSensorAdapter,
    },
    authority::{
        AuthorizedCaptureRequest, CaptureAuthority, CaptureAuthorizationRequest,
        CapturePolicyDecision, CaptureSource, HostCapturePolicy, TrustedCaptureClock,
    },
    buffer::SampleLayout,
    extended_capability::ExtendedCapability,
    quantity::SampleEncoding,
    time::ClockStamp,
    ContractError,
};

use crate::stream::{StreamAdapter, StreamState};

#[derive(Clone)]
struct TestClock(Rc<Cell<u64>>);

impl TestClock {
    fn new(ticks: u64) -> Self {
        Self(Rc::new(Cell::new(ticks)))
    }
    fn stamp(&self) -> ClockStamp {
        ClockStamp::new(1, 1, self.0.get(), 0).unwrap()
    }
}

impl TrustedCaptureClock for TestClock {
    fn now(&mut self) -> Result<ClockStamp, AdapterError> {
        Ok(self.stamp())
    }
}

struct PermitPolicy {
    expiry: ClockStamp,
}

impl HostCapturePolicy for PermitPolicy {
    fn decide(
        &self,
        _request: &CaptureAuthorizationRequest,
        _now: ClockStamp,
    ) -> CapturePolicyDecision {
        CapturePolicyDecision::Permit {
            expires_at: self.expiry,
        }
    }
}

struct ChunkReader<'a> {
    data: &'a [u8],
    offset: usize,
    max_chunk: usize,
    interrupt_once: bool,
    change_clock: Option<(TestClock, u64)>,
    error_after: Option<usize>,
    oversized_once: bool,
}

impl Read for ChunkReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.interrupt_once {
            self.interrupt_once = false;
            return Err(io::ErrorKind::Interrupted.into());
        }
        if self.oversized_once {
            self.oversized_once = false;
            out.fill(0xCC);
            return Ok(out.len() + 1);
        }
        if let Some((clock, value)) = &self.change_clock {
            clock.0.set(*value);
        }
        if matches!(self.error_after, Some(limit) if self.offset >= limit) {
            out.fill(0xEE);
            return Err(io::Error::other("injected reader failure"));
        }
        if self.offset == self.data.len() {
            return Ok(0);
        }
        let count = out
            .len()
            .min(self.max_chunk)
            .min(self.data.len() - self.offset);
        out[..count].copy_from_slice(&self.data[self.offset..self.offset + count]);
        self.offset += count;
        Ok(count)
    }
}

fn fixture<'a>(
    data: &'a [u8],
    chunk: usize,
    clock: TestClock,
) -> (
    StreamAdapter<ChunkReader<'a>, TestClock>,
    ExtendedDiscoveredSource,
    SampleLayout,
) {
    let cap =
        ExtendedCapability::new(77, 51, None, SampleEncoding::U8, [4, 0, 0, 0], None).unwrap();
    let source = ExtendedDiscoveredSource::new(9, cap).unwrap();
    let layout = SampleLayout::new(1, [4, 0, 0, 0], [1, 0, 0, 0], SampleEncoding::U8).unwrap();
    let reader = ChunkReader {
        data,
        offset: 0,
        max_chunk: chunk,
        interrupt_once: false,
        change_clock: None,
        error_after: None,
        oversized_once: false,
    };
    let adapter = StreamAdapter::new(
        AdapterId::new(3).unwrap(),
        source,
        layout.clone(),
        4,
        reader,
        clock,
    )
    .unwrap();
    (adapter, source, layout)
}

fn issue(
    source: ExtendedDiscoveredSource,
    now: &TestClock,
    expiry: u64,
) -> etalon_sense_core::adapter::AuthorizedExtendedCapture {
    let request =
        CaptureAuthorizationRequest::new(CaptureSource::Extended(source), None, None, 17, 22)
            .unwrap();
    let policy = PermitPolicy {
        expiry: ClockStamp::new(1, 1, expiry, 0).unwrap(),
    };
    let issued = CaptureAuthority::issue(request, &policy, &mut now.clone()).unwrap();
    match issued.into_capture_request() {
        AuthorizedCaptureRequest::Extended(capture) => capture,
        AuthorizedCaptureRequest::BuiltIn(_) => unreachable!(),
    }
}

fn start<R: Read>(
    adapter: &mut StreamAdapter<R, TestClock>,
    source: ExtendedDiscoveredSource,
) -> ExtendedCaptureSession {
    adapter
        .start(issue(source, &adapter_clock(adapter), 1_000))
        .unwrap()
}

// Obtain the same deterministic test clock stamp without exposing adapter internals.
fn adapter_clock<R: Read>(_adapter: &StreamAdapter<R, TestClock>) -> TestClock {
    TestClock::new(100)
}

fn authorized_buffer<'a>(
    source: ExtendedDiscoveredSource,
    layout: SampleLayout,
    session: ExtendedCaptureSession,
    bytes: &'a mut [u8],
) -> AuthorizedExtendedCaptureBuffer<'a> {
    let adapter = ExtendedAdapterDescriptor::new(AdapterId::new(3).unwrap(), 77).unwrap();
    let raw = ExtendedCaptureBuffer::new(bytes, layout, source.capability()).unwrap();
    AuthorizedExtendedCaptureBuffer::new(
        adapter,
        session,
        issue(source, &TestClock::new(100), 1_000),
        raw,
    )
    .unwrap()
}

#[test]
fn assembles_partial_reads_into_one_exact_frame() {
    let clock = TestClock::new(100);
    let (mut adapter, source, layout) = fixture(&[1, 2, 3, 4], 2, clock);
    let session = start(&mut adapter, source);
    let mut out = [0; 4];
    let capture = adapter
        .capture(authorized_buffer(source, layout, session, &mut out))
        .unwrap();
    assert_eq!(capture.bytes(), &[1, 2, 3, 4]);
    assert_eq!(adapter.completed_frames(), 1);
}

#[test]
fn retries_interrupted_read_calls() {
    let clock = TestClock::new(100);
    let (mut adapter, source, layout) = fixture(&[4, 3, 2, 1], 4, clock);
    adapter.reader_mut().interrupt_once = true;
    let session = start(&mut adapter, source);
    let mut out = [0; 4];
    let capture = adapter
        .capture(authorized_buffer(source, layout, session, &mut out))
        .unwrap();
    assert_eq!(capture.bytes(), &[4, 3, 2, 1]);
}

#[test]
fn clean_eof_returns_no_frame_and_sticks() {
    let clock = TestClock::new(100);
    let (mut adapter, source, layout) = fixture(&[], 4, clock);
    let session = start(&mut adapter, source);
    let mut out = [0; 4];
    let result = adapter.capture(authorized_buffer(source, layout, session, &mut out));
    assert!(matches!(result, Err(AdapterError::Unavailable)));
    assert_eq!(adapter.completed_frames(), 0);
    assert_eq!(adapter.state(), StreamState::Eof);
}

#[test]
fn partial_eof_clears_buffer_and_never_returns_a_frame() {
    let clock = TestClock::new(100);
    let (mut adapter, source, layout) = fixture(&[8, 7], 2, clock);
    let session = start(&mut adapter, source);
    let mut out = [0xAA; 4];
    let result = adapter.capture(authorized_buffer(source, layout, session, &mut out));
    assert!(matches!(result, Err(AdapterError::BackendFailure)));
    assert_eq!(out, [0; 4]);
    assert_eq!(adapter.completed_frames(), 0);
    assert_eq!(adapter.state(), StreamState::Eof);
}

#[test]
fn reader_error_after_partial_data_faults_without_returning_a_frame() {
    let clock = TestClock::new(100);
    let (mut adapter, source, layout) = fixture(&[8, 7, 6, 5], 2, clock);
    adapter.reader_mut().error_after = Some(2);
    let session = start(&mut adapter, source);
    let mut out = [0xAA; 4];
    let result = adapter.capture(authorized_buffer(source, layout, session, &mut out));
    assert!(matches!(result, Err(AdapterError::BackendFailure)));
    assert_eq!(out, [0; 4]);
    assert_eq!(adapter.completed_frames(), 0);
    assert_eq!(adapter.state(), StreamState::Faulted);
}

#[test]
fn expiry_after_a_read_rejects_consumed_bytes_and_faults_stream() {
    let clock = TestClock::new(100);
    let (mut adapter, source, layout) = fixture(&[1, 2, 3, 4], 2, clock.clone());
    adapter.reader_mut().change_clock = Some((clock, 1_001));
    let session = start(&mut adapter, source);
    let mut out = [0xAA; 4];
    let result = adapter.capture(authorized_buffer(source, layout, session, &mut out));
    assert!(matches!(result, Err(AdapterError::AuthorizationExpired)));
    assert_eq!(out, [0; 4]);
    assert_eq!(adapter.completed_frames(), 0);
    assert_eq!(adapter.state(), StreamState::Faulted);
}

#[test]
fn oversized_reader_count_clears_the_entire_frame_buffer() {
    let clock = TestClock::new(100);
    let (mut adapter, source, layout) = fixture(&[1, 2, 3, 4], 4, clock);
    adapter.reader_mut().oversized_once = true;
    let session = start(&mut adapter, source);
    let mut out = [0xAA; 4];
    let result = adapter.capture(authorized_buffer(source, layout, session, &mut out));
    assert!(matches!(result, Err(AdapterError::BackendFailure)));
    assert_eq!(out, [0; 4]);
    assert_eq!(adapter.state(), StreamState::Faulted);
    assert_eq!(adapter.completed_frames(), 0);
}

#[test]
fn completion_counter_overflow_clears_the_completed_frame() {
    let clock = TestClock::new(100);
    let (mut adapter, source, layout) = fixture(&[1, 2, 3, 4], 4, clock);
    let session = start(&mut adapter, source);
    adapter.completed_frames = u64::MAX;
    let mut out = [0xAA; 4];
    let result = adapter.capture(authorized_buffer(source, layout, session, &mut out));
    assert!(matches!(result, Err(AdapterError::BackendFailure)));
    assert_eq!(out, [0; 4]);
    assert_eq!(adapter.state(), StreamState::Faulted);
}

#[test]
fn constructor_rejects_layout_exceeding_explicit_frame_budget() {
    let cap =
        ExtendedCapability::new(77, 51, None, SampleEncoding::U8, [4, 0, 0, 0], None).unwrap();
    let source = ExtendedDiscoveredSource::new(9, cap).unwrap();
    let layout = SampleLayout::new(1, [4, 0, 0, 0], [1, 0, 0, 0], SampleEncoding::U8).unwrap();
    let result = StreamAdapter::new(
        AdapterId::new(3).unwrap(),
        source,
        layout,
        3,
        io::empty(),
        TestClock::new(100),
    );
    assert!(matches!(
        result,
        Err(AdapterError::Buffer(ContractError::OutOfBounds))
    ));
}
