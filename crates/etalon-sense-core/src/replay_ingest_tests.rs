use super::*;
use crate::{
    adapter::{AdapterError, AdapterId, ExtendedCapturePermit, ExtendedSensorAdapter},
    buffer::{BufferToken, SampleLayout},
    evidence::EvidenceRecord,
    extended_capability::ExtendedCapability,
    ingest::ExtendedIngestStream,
    quality::{CalibrationRecord, CalibrationState, QualityRecord, TrustState},
    quantity::SampleEncoding,
    replay::{
        ReplayAuthorizationClock, ReplayFrameDescriptor, ReplayKind, ReplayMetadata,
        ReplayStreamBudget,
    },
    time::ClockStamp,
};

fn stamp(ticks: u64) -> ClockStamp {
    ClockStamp::new(19, 3, ticks, 1).unwrap()
}

fn capability() -> ExtendedCapability {
    ExtendedCapability::new(
        501,
        502,
        None,
        SampleEncoding::ComplexI16,
        [2, 0, 0, 0],
        None,
    )
    .unwrap()
}

fn layout() -> SampleLayout {
    SampleLayout::new(1, [2, 0, 0, 0], [4, 0, 0, 0], SampleEncoding::ComplexI16).unwrap()
}

fn region() -> SampleRegion {
    SampleRegion::new(BufferToken::new(88, 2).unwrap(), 0, 8, 8, 2).unwrap()
}

fn calibration() -> CalibrationRecord {
    CalibrationRecord::new(CalibrationState::Missing).unwrap()
}

struct FixedClock;
impl ReplayAuthorizationClock for FixedClock {
    fn trusted_now(&mut self) -> Result<ClockStamp, AdapterError> {
        Ok(stamp(5))
    }
}

fn setup<'b, 'f, 'c>(
    data: &'b [u8],
    frames: &'f [ReplayFrameDescriptor],
    clock: &'c mut FixedClock,
) -> ReplayStreamAdapter<'b, 'f, 'c> {
    ReplayStreamAdapter::new(
        AdapterId::new(41).unwrap(),
        31,
        capability(),
        51,
        2,
        data,
        frames,
        ReplayMetadata::new(
            ReplayKind::Prerecorded,
            EvidenceRecord::unknown(),
            Some(stamp(4)),
        ),
        ReplayStreamBudget::new(4, 32).unwrap(),
        clock,
    )
    .unwrap()
}

fn descriptors() -> [ReplayFrameDescriptor; 2] {
    [
        ReplayFrameDescriptor::new(
            0,
            layout(),
            stamp(10),
            7,
            None,
            calibration(),
            QualityRecord::Unknown,
        ),
        ReplayFrameDescriptor::new(
            8,
            layout(),
            stamp(12),
            9,
            None,
            calibration(),
            QualityRecord::Unknown,
        ),
    ]
}

fn request(source: crate::adapter::ExtendedDiscoveredSource) -> AuthorizedExtendedCapture {
    let permit = ExtendedCapturePermit::after_authorization(
        source.source_id(),
        9,
        source.capability(),
        stamp(100),
    )
    .unwrap();
    AuthorizedExtendedCapture::new(source, permit, stamp(3)).unwrap()
}

fn output<'a>(
    adapter: &ReplayStreamAdapter<'_, '_, '_>,
    bytes: &'a mut [u8; 8],
) -> ExtendedCaptureBuffer<'a> {
    ExtendedCaptureBuffer::new(bytes, layout(), adapter.source().capability()).unwrap()
}

#[test]
fn replay_ingest_uses_frame_metadata_and_preserves_unknown_provenance() {
    let data = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18];
    let frames = descriptors();
    let mut clock = FixedClock;
    let mut adapter = setup(&data, &frames, &mut clock);
    let source = adapter.source();
    let session = adapter.start(request(source)).unwrap();
    let mut stream = ExtendedIngestStream::new(31, 51, 2, capability()).unwrap();

    let mut first_bytes = [0; 8];
    let first_output = output(&adapter, &mut first_bytes);
    let first = capture_replay_and_ingest(
        &mut adapter,
        session,
        request(source),
        first_output,
        &mut stream,
        region(),
    )
    .unwrap();
    assert_eq!(first.ingested().payload(), &data[..8]);
    assert_eq!(first.ingested().measurement().timestamp(), Some(stamp(10)));
    assert_eq!(first.ingested().measurement().sample_rate(), None);
    assert_eq!(
        first.ingested().measurement().quality(),
        QualityRecord::Unknown
    );
    assert_eq!(
        first.ingested().measurement().source().trust_state(),
        TrustState::Unverified
    );
    assert_eq!(first.ingested().sequence_event(), SequenceEvent::First);
    assert_eq!(first.replay_info().sequence(), 7);
    assert_eq!(first.replay_metadata().kind(), ReplayKind::Prerecorded);

    let mut second_bytes = [0; 8];
    let second_output = output(&adapter, &mut second_bytes);
    let second = capture_replay_and_ingest(
        &mut adapter,
        session,
        request(source),
        second_output,
        &mut stream,
        region(),
    )
    .unwrap();
    assert_eq!(second.ingested().measurement().timestamp(), Some(stamp(12)));
    assert_eq!(
        second.ingested().sequence_event(),
        SequenceEvent::Gap { missing: 1 }
    );
    assert_eq!(
        second.replay_info().sequence_event(),
        SequenceEvent::Gap { missing: 1 }
    );
    assert_eq!(adapter.next_frame_index(), 2);
}

#[test]
fn stream_mismatch_and_bad_region_fail_before_replay_advances() {
    let data = [0; 16];
    let frames = descriptors();
    let mut clock = FixedClock;
    let mut adapter = setup(&data, &frames, &mut clock);
    let source = adapter.source();
    let session = adapter.start(request(source)).unwrap();
    let mut wrong_stream = ExtendedIngestStream::new(31, 52, 2, capability()).unwrap();
    let mut bytes = [0; 8];
    let wrong_output = output(&adapter, &mut bytes);
    let result = capture_replay_and_ingest(
        &mut adapter,
        session,
        request(source),
        wrong_output,
        &mut wrong_stream,
        region(),
    );
    assert!(matches!(
        result,
        Err(ReplayIngestFailure::BeforeCapture(
            ReplayIngestErrorKind::StreamMismatch
        ))
    ));
    assert_eq!(adapter.next_frame_index(), 0);

    let mut stream = ExtendedIngestStream::new(31, 51, 2, capability()).unwrap();
    let bad_region = SampleRegion::new(BufferToken::new(88, 2).unwrap(), 0, 4, 8, 2).unwrap();
    let mut bytes = [0; 8];
    let bad_output = output(&adapter, &mut bytes);
    let result = capture_replay_and_ingest(
        &mut adapter,
        session,
        request(source),
        bad_output,
        &mut stream,
        bad_region,
    );
    assert!(matches!(
        result,
        Err(ReplayIngestFailure::BeforeCapture(
            ReplayIngestErrorKind::Contract(ContractError::OutOfBounds)
        ))
    ));
    assert_eq!(adapter.next_frame_index(), 0);
}

#[test]
fn stream_order_failure_retains_captured_frame_without_advancing_ingest() {
    let data = [0; 16];
    let frames = descriptors();
    let mut clock = FixedClock;
    let mut adapter = setup(&data, &frames, &mut clock);
    let source = adapter.source();
    let session = adapter.start(request(source)).unwrap();
    let mut stream = ExtendedIngestStream::new(31, 51, 2, capability()).unwrap();
    let descriptor = SourceDescriptor::new(31, SourceKind::Recorded).unwrap();
    stream
        .ingest(
            &[0; 8],
            layout(),
            region(),
            descriptor,
            stamp(9),
            None,
            calibration(),
            QualityRecord::Unknown,
            8,
        )
        .unwrap();
    let before = stream.last_sequence();

    let mut bytes = [0; 8];
    let output = output(&adapter, &mut bytes);
    let result = capture_replay_and_ingest(
        &mut adapter,
        session,
        request(source),
        output,
        &mut stream,
        region(),
    );
    match result {
        Err(ReplayIngestFailure::Captured {
            error: ReplayIngestErrorKind::Contract(ContractError::InvalidSource),
            frame,
        }) => {
            assert_eq!(frame.info().sequence(), 7);
            assert_eq!(frame.samples().bytes(), &[0; 8]);
        }
        other => panic!("unexpected result: {other:?}"),
    }
    assert_eq!(stream.last_sequence(), before);
    assert_eq!(adapter.next_frame_index(), 1);
}

#[test]
fn replay_order_uses_sequence_after_an_untimed_prior_observation() {
    let data = [0; 16];
    let frames = descriptors();
    let mut clock = FixedClock;
    let mut adapter = setup(&data, &frames, &mut clock);
    let source = adapter.source();
    let session = adapter.start(request(source)).unwrap();
    let mut stream = ExtendedIngestStream::new(31, 51, 2, capability()).unwrap();
    let descriptor = SourceDescriptor::new(31, SourceKind::Recorded).unwrap();
    stream
        .ingest_with_optional_timestamp(
            &[0; 8],
            layout(),
            region(),
            descriptor,
            None,
            None,
            calibration(),
            QualityRecord::Unknown,
            6,
        )
        .unwrap();

    let mut bytes = [0; 8];
    let output = output(&adapter, &mut bytes);
    let result = capture_replay_and_ingest(
        &mut adapter,
        session,
        request(source),
        output,
        &mut stream,
        region(),
    );
    assert!(matches!(
        result,
        Err(ReplayIngestFailure::Captured {
            error: ReplayIngestErrorKind::ReplayOrderMismatch,
            ..
        })
    ));
    assert_eq!(stream.last_observation(), Some((6, None)));
    assert_eq!(adapter.next_frame_index(), 1);
}
