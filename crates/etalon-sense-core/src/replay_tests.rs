use super::{
    ExtendedReplayAdapter, ExtendedReplayBlock, ReplayAdapter, ReplayAuthorizationClock,
    ReplayBlock, ReplayKind, ReplayMetadata,
};
use crate::{
    adapter::{
        AdapterError, AdapterId, AuthorizedCapture, AuthorizedCaptureBuffer, CaptureBuffer,
        CapturePermit, DiscoveredSource, SensorAdapter, AuthorizedExtendedCapture,
        AuthorizedExtendedCaptureBuffer, ExtendedCaptureBuffer, ExtendedCapturePermit,
        ExtendedDiscoveredSource, ExtendedSensorAdapter,
    },
    buffer::SampleLayout,
    capability::{Modality, SensorCapability},
    evidence::EvidenceRecord,
    extended_capability::ExtendedCapability,
    quantity::{PhysicalQuantity, SampleEncoding, Unit},
    time::ClockStamp,
    ContractError,
};

fn stamp(ticks: u64) -> ClockStamp { ClockStamp::new(9, 1, ticks, 0).unwrap() }

fn capability() -> SensorCapability {
    SensorCapability::new(
        Modality::Wifi,
        PhysicalQuantity::RfChannelResponse,
        Unit::Unitless,
        SampleEncoding::ComplexI16,
        [2, 0, 0, 0],
        None,
    )
    .unwrap()
}

fn extended_capability() -> ExtendedCapability {
    ExtendedCapability::new(
        901,
        902,
        None,
        SampleEncoding::ComplexI16,
        [2, 0, 0, 0],
        None,
    )
    .unwrap()
}

fn extended_layout() -> SampleLayout {
    SampleLayout::new(1, [2, 0, 0, 0], [4, 0, 0, 0], SampleEncoding::ComplexI16).unwrap()
}

fn extended_block<'a>(bytes: &'a [u8]) -> ExtendedReplayBlock<'a> {
    ExtendedReplayBlock::new(
        27,
        extended_capability(),
        extended_layout(),
        bytes,
        ReplayMetadata::new(ReplayKind::Synthetic, EvidenceRecord::unknown(), None),
    )
    .unwrap()
}

fn extended_source() -> ExtendedDiscoveredSource {
    ExtendedDiscoveredSource::new(27, extended_capability()).unwrap()
}

fn extended_authorized(source: ExtendedDiscoveredSource, at: ClockStamp) -> AuthorizedExtendedCapture {
    let permit = ExtendedCapturePermit::after_authorization(
        27,
        23,
        source.capability(),
        stamp(100),
    )
    .unwrap();
    AuthorizedExtendedCapture::new(source, permit, at).unwrap()
}

fn layout() -> SampleLayout {
    SampleLayout::new(1, [2, 0, 0, 0], [4, 0, 0, 0], SampleEncoding::ComplexI16).unwrap()
}

fn block<'a>(bytes: &'a [u8]) -> ReplayBlock<'a> {
    ReplayBlock::new(
        17,
        capability(),
        layout(),
        bytes,
        ReplayMetadata::new(ReplayKind::Prerecorded, EvidenceRecord::unknown(), Some(stamp(3))),
    )
    .unwrap()
}

struct FixedClock(ClockStamp);

impl ReplayAuthorizationClock for FixedClock {
    fn trusted_now(&mut self) -> Result<ClockStamp, AdapterError> { Ok(self.0) }
}

fn authorized(source: DiscoveredSource, at: ClockStamp) -> AuthorizedCapture {
    let permit = CapturePermit::after_authorization(17, 23, source.capability(), stamp(100)).unwrap();
    AuthorizedCapture::new(source, permit, at).unwrap()
}

#[test]
fn block_requires_exact_layout_span_and_matching_capability() {
    let short = [0u8; 7];
    assert_eq!(
        ReplayBlock::new(
            17,
            capability(),
            layout(),
            &short,
            ReplayMetadata::new(ReplayKind::Synthetic, EvidenceRecord::unknown(), None),
        )
        .err()
        .unwrap(),
        AdapterError::Buffer(ContractError::OutOfBounds)
    );
}

#[test]
fn adapter_discovers_only_its_caller_supplied_source() {
    let samples = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut clock = FixedClock(stamp(10));
    let mut adapter = ReplayAdapter::new(AdapterId::new(4).unwrap(), block(&samples), &mut clock);
    let mut none = [];
    assert_eq!(adapter.discover(&mut none), Err(AdapterError::DiscoveryBufferTooSmall));
    let mut found = [DiscoveredSource::new(99, capability()).unwrap()];
    assert_eq!(adapter.discover(&mut found), Ok(1));
    assert_eq!(found[0].source_id(), 17);
    assert_eq!(adapter.metadata().kind(), ReplayKind::Prerecorded);
    assert_eq!(adapter.metadata().recorded_at(), Some(stamp(3)));
}

#[test]
fn capture_copies_once_to_validated_caller_buffer_and_needs_authorization() {
    let samples = [11, 12, 13, 14, 15, 16, 17, 18];
    let mut clock = FixedClock(stamp(10));
    let mut adapter = ReplayAdapter::new(AdapterId::new(4).unwrap(), block(&samples), &mut clock);
    let source = adapter.discover(&mut [DiscoveredSource::new(17, capability()).unwrap()]).unwrap();
    assert_eq!(source, 1);
    let source = DiscoveredSource::new(17, capability()).unwrap();
    let session = adapter.start(authorized(source, stamp(10))).unwrap();

    let mut destination = [0u8; 8];
    let buffer = CaptureBuffer::new(&mut destination, layout(), capability()).unwrap();
    let output = AuthorizedCaptureBuffer::new(
        adapter.descriptor(), session, authorized(source, stamp(10)), buffer,
    )
    .unwrap();
    let captured = adapter.capture(output).unwrap();
    assert_eq!(captured.bytes(), &samples);
    assert_eq!(captured.layout(), &layout());
    assert!(adapter.is_consumed());

    let mut second_destination = [0u8; 8];
    let second_buffer = CaptureBuffer::new(&mut second_destination, layout(), capability()).unwrap();
    let second_output = AuthorizedCaptureBuffer::new(
        adapter.descriptor(), session, authorized(source, stamp(10)), second_buffer,
    )
    .unwrap();
    assert_eq!(adapter.capture(second_output), Err(AdapterError::Unavailable));
}

#[test]
fn adapter_cannot_discover_or_authorize_another_source() {
    let samples = [0u8; 8];
    let mut clock = FixedClock(stamp(10));
    let mut adapter = ReplayAdapter::new(AdapterId::new(4).unwrap(), block(&samples), &mut clock);
    let other = DiscoveredSource::new(18, capability()).unwrap();
    assert_eq!(adapter.start(authorized(other, stamp(10))), Err(AdapterError::AuthorizationDenied));
}

#[test]
fn extended_block_requires_exact_layout_span() {
    let short = [0u8; 7];
    assert_eq!(
        ExtendedReplayBlock::new(
            27,
            extended_capability(),
            extended_layout(),
            &short,
            ReplayMetadata::new(ReplayKind::Synthetic, EvidenceRecord::unknown(), None),
        )
        .err()
        .unwrap(),
        AdapterError::Buffer(ContractError::OutOfBounds)
    );
}

#[test]
fn extended_replay_is_bound_to_exact_source_capability_and_buffer() {
    let samples = [21, 22, 23, 24, 25, 26, 27, 28];
    let mut clock = FixedClock(stamp(10));
    let mut adapter = ExtendedReplayAdapter::new(
        AdapterId::new(6).unwrap(),
        extended_block(&samples),
        &mut clock,
    )
    .unwrap();
    assert_eq!(adapter.metadata().kind(), ReplayKind::Synthetic);
    assert_eq!(adapter.source().source_id(), 27);
    assert_eq!(adapter.source().capability().unit_id(), None);

    let mut discovered = [extended_source()];
    assert_eq!(adapter.discover(&mut discovered), Ok(1));
    let source = discovered[0];
    let descriptor = adapter.descriptor();
    let session = adapter.start(extended_authorized(source, stamp(10))).unwrap();

    let mut destination = [0u8; 8];
    let buffer = ExtendedCaptureBuffer::new(&mut destination, extended_layout(), source.capability()).unwrap();
    let output = AuthorizedExtendedCaptureBuffer::new(
        descriptor,
        session,
        extended_authorized(source, stamp(10)),
        buffer,
    )
    .unwrap();
    let captured = adapter.capture(output).unwrap();
    assert_eq!(captured.bytes(), &samples);
    assert_eq!(captured.layout(), &extended_layout());
    assert!(adapter.is_consumed());
}

#[test]
fn extended_authorization_cannot_be_reused_for_changed_unit_or_source() {
    let source = extended_source();
    let permit = ExtendedCapturePermit::after_authorization(
        source.source_id(),
        23,
        source.capability(),
        stamp(100),
    )
    .unwrap();
    let wrong_capability = ExtendedCapability::new(
        901,
        902,
        Some(903),
        SampleEncoding::ComplexI16,
        [2, 0, 0, 0],
        None,
    )
    .unwrap();
    let changed_source = ExtendedDiscoveredSource::new(source.source_id(), wrong_capability).unwrap();
    assert_eq!(
        AuthorizedExtendedCapture::new(changed_source, permit, stamp(10)),
        Err(AdapterError::AuthorizationDenied)
    );
}
