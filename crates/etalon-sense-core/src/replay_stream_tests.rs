use super::*;
use crate::{
    adapter::{
        AuthorizedExtendedCapture, AuthorizedExtendedCaptureBuffer, ExtendedCaptureBuffer,
        ExtendedCapturePermit,
    },
    evidence::EvidenceRecord,
    quality::CalibrationState,
    quantity::SampleEncoding,
    replay::ReplayKind,
};
use std::{cell::Cell, rc::Rc};

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

fn calibration() -> CalibrationRecord {
    CalibrationRecord::new(CalibrationState::Missing).unwrap()
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

struct FixedClock(ClockStamp);

impl ReplayAuthorizationClock for FixedClock {
    fn trusted_now(&mut self) -> Result<ClockStamp, AdapterError> {
        Ok(self.0)
    }
}

fn adapter<'b, 'f, 'c>(
    bytes: &'b [u8],
    frames: &'f [ReplayFrameDescriptor],
    clock: &'c mut FixedClock,
) -> ReplayStreamAdapter<'b, 'f, 'c> {
    ReplayStreamAdapter::new(
        AdapterId::new(41).unwrap(),
        31,
        capability(),
        51,
        2,
        bytes,
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

fn authorized(
    source: ExtendedDiscoveredSource,
    now: ClockStamp,
    expiry: ClockStamp,
) -> AuthorizedExtendedCapture {
    let permit = ExtendedCapturePermit::after_authorization(
        source.source_id(),
        9,
        source.capability(),
        expiry,
    )
    .unwrap();
    AuthorizedExtendedCapture::new(source, permit, now).unwrap()
}

fn output<'a>(
    adapter: &ReplayStreamAdapter<'_, '_, '_>,
    session: ExtendedCaptureSession,
    now: ClockStamp,
    expiry: ClockStamp,
    bytes: &'a mut [u8; 8],
) -> AuthorizedExtendedCaptureBuffer<'a> {
    let source = adapter.source();
    let buffer = ExtendedCaptureBuffer::new(bytes, layout(), source.capability()).unwrap();
    AuthorizedExtendedCaptureBuffer::new(
        adapter.descriptor(),
        session,
        authorized(source, now, expiry),
        buffer,
    )
    .unwrap()
}

#[test]
fn replay_stream_copies_authorized_frames_and_preserves_gaps_and_clock() {
    let data = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18];
    let frame_table = descriptors();
    let mut clock = FixedClock(stamp(5));
    let mut adapter = adapter(&data, &frame_table, &mut clock);
    let mut found = [adapter.source()];
    assert_eq!(adapter.discover(&mut found), Ok(1));
    let session = adapter
        .start(authorized(found[0], stamp(3), stamp(100)))
        .unwrap();

    let mut first_bytes = [0; 8];
    let first_output = output(&adapter, session, stamp(3), stamp(100), &mut first_bytes);
    let first = adapter.capture_frame(first_output).unwrap();
    assert_eq!(first.samples().bytes(), &data[..8]);
    assert_eq!(first.info().timestamp(), stamp(10));
    assert_eq!(first.info().sequence(), 7);
    assert_eq!(first.info().sequence_event(), SequenceEvent::First);
    assert_eq!(first.replay_metadata().kind(), ReplayKind::Prerecorded);

    let mut second_bytes = [0; 8];
    let second_output = output(&adapter, session, stamp(3), stamp(100), &mut second_bytes);
    let second = adapter.capture_frame(second_output).unwrap();
    assert_eq!(second.samples().bytes(), &data[8..]);
    assert_eq!(second.info().timestamp(), stamp(12));
    assert_eq!(second.info().sequence(), 9);
    assert_eq!(
        second.info().sequence_event(),
        SequenceEvent::Gap { missing: 1 }
    );
    assert_eq!(adapter.next_frame_index(), 2);
    assert!(adapter.is_exhausted());
}

#[test]
fn constructor_rejects_budget_bounds_rate_shape_and_clock_order() {
    let data = [0; 16];
    let frames = descriptors();
    let mut clock = FixedClock(stamp(5));
    assert_eq!(
        ReplayStreamAdapter::new(
            AdapterId::new(41).unwrap(),
            31,
            capability(),
            51,
            2,
            &data,
            &frames,
            ReplayMetadata::new(ReplayKind::Synthetic, EvidenceRecord::unknown(), None),
            ReplayStreamBudget::new(1, 32).unwrap(),
            &mut clock,
        )
        .err()
        .unwrap(),
        AdapterError::Buffer(ContractError::InvalidExtent)
    );
    assert_eq!(
        ReplayStreamAdapter::new(
            AdapterId::new(41).unwrap(),
            31,
            capability(),
            51,
            2,
            &data,
            &frames,
            ReplayMetadata::new(ReplayKind::Synthetic, EvidenceRecord::unknown(), None),
            ReplayStreamBudget::new(2, 8).unwrap(),
            &mut clock,
        )
        .err()
        .unwrap(),
        AdapterError::Buffer(ContractError::OutOfBounds)
    );

    let wrong_time = [
        frames[0].clone(),
        ReplayFrameDescriptor::new(
            8,
            layout(),
            ClockStamp::new(20, 3, 12, 1).unwrap(),
            9,
            None,
            calibration(),
            QualityRecord::Unknown,
        ),
    ];
    assert_eq!(
        ReplayStreamAdapter::new(
            AdapterId::new(41).unwrap(),
            31,
            capability(),
            51,
            2,
            &data,
            &wrong_time,
            ReplayMetadata::new(ReplayKind::Synthetic, EvidenceRecord::unknown(), None),
            ReplayStreamBudget::new(2, 32).unwrap(),
            &mut clock,
        )
        .err()
        .unwrap(),
        AdapterError::Buffer(ContractError::ClockDomainMismatch)
    );
}

#[test]
fn expired_or_mismatched_authorization_does_not_advance_replay() {
    let data = [0; 16];
    let frames = descriptors();
    struct SharedClock(Rc<Cell<ClockStamp>>);
    impl ReplayAuthorizationClock for SharedClock {
        fn trusted_now(&mut self) -> Result<ClockStamp, AdapterError> {
            Ok(self.0.get())
        }
    }
    let clock_state = Rc::new(Cell::new(stamp(5)));
    let mut clock = SharedClock(clock_state.clone());
    let mut adapter = ReplayStreamAdapter::new(
        AdapterId::new(41).unwrap(),
        31,
        capability(),
        51,
        2,
        &data,
        &frames,
        ReplayMetadata::new(
            ReplayKind::Prerecorded,
            EvidenceRecord::unknown(),
            Some(stamp(4)),
        ),
        ReplayStreamBudget::new(4, 32).unwrap(),
        &mut clock,
    )
    .unwrap();
    let source = adapter.source();
    let session = adapter
        .start(authorized(source, stamp(3), stamp(100)))
        .unwrap();

    let mut bytes = [0; 8];
    let valid = output(&adapter, session, stamp(3), stamp(100), &mut bytes);
    clock_state.set(stamp(101));
    assert_eq!(
        adapter.capture_frame(valid).unwrap_err(),
        AdapterError::AuthorizationExpired
    );
    assert_eq!(adapter.next_frame_index(), 0);

    let wrong = ExtendedDiscoveredSource::new(
        31,
        ExtendedCapability::new(
            501,
            503,
            None,
            SampleEncoding::ComplexI16,
            [2, 0, 0, 0],
            None,
        )
        .unwrap(),
    )
    .unwrap();
    let source_permit = ExtendedCapturePermit::after_authorization(
        source.source_id(),
        9,
        source.capability(),
        stamp(200),
    )
    .unwrap();
    assert_eq!(
        AuthorizedExtendedCapture::new(wrong, source_permit, stamp(5)),
        Err(AdapterError::AuthorizationDenied)
    );
}
