use super::*;
use crate::buffer::BufferToken;
use crate::capability::Modality;
use crate::quality::{CalibrationState, SourceKind};
use crate::quantity::{PhysicalQuantity, SampleEncoding, Unit};

fn layout() -> SampleLayout {
    SampleLayout::new(1, [2, 0, 0, 0], [8, 0, 0, 0], SampleEncoding::ComplexF32).unwrap()
}

fn region() -> SampleRegion {
    SampleRegion::new(BufferToken::new(90, 1).unwrap(), 0, 16, 16, 4).unwrap()
}

fn source(source_id: u64) -> SourceDescriptor {
    SourceDescriptor::new(source_id, SourceKind::Recorded).unwrap()
}

fn capability() -> SensorCapability {
    SensorCapability::new(
        Modality::Wifi,
        PhysicalQuantity::RfChannelResponse,
        Unit::Unitless,
        SampleEncoding::ComplexF32,
        [4, 0, 0, 0],
        Some(SampleRate::new(100.0).unwrap()),
    )
    .unwrap()
}

fn calibration() -> CalibrationRecord {
    CalibrationRecord::new(CalibrationState::Missing).unwrap()
}

fn timestamp(sequence: u64) -> ClockStamp {
    ClockStamp::new(5, 2, sequence * 1_000, 2).unwrap()
}

fn stream() -> IngestStream {
    IngestStream::new(17, 25, 3).unwrap()
}

fn ingest<'a>(
    stream: &mut IngestStream,
    payload: &'a [u8],
    source_id: u64,
    sequence: u64,
) -> Result<IngestedFrame<'a>, ContractError> {
    stream.ingest(
        payload,
        layout(),
        region(),
        source(source_id),
        capability(),
        timestamp(sequence),
        None,
        calibration(),
        QualityRecord::Unknown,
        sequence,
    )
}

#[test]
fn admits_exact_payload_and_preserves_unknown_metadata() {
    let mut stream = stream();
    let payload = [0xA5; 16];
    let frame = ingest(&mut stream, &payload, 17, 4).unwrap();

    assert_eq!(frame.payload(), &payload);
    assert_eq!(frame.sequence_event(), SequenceEvent::First);
    assert_eq!(frame.sequence(), 4);
    assert_eq!(frame.stream_id(), 25);
    assert_eq!(frame.generation(), 3);
    assert_eq!(frame.measurement().source().source_id(), 17);
    assert_eq!(frame.measurement().sample_rate(), None);
    assert_eq!(frame.measurement().quality(), QualityRecord::Unknown);
    assert_eq!(frame.measurement().calibration(), calibration());
    assert_eq!(stream.last_sequence(), Some((4, timestamp(4))));
}

#[test]
fn unknown_timestamps_are_preserved_without_weakening_sequence_order() {
    let mut stream = stream();
    let payload = [0xA5; 16];
    let first = stream
        .ingest_with_optional_timestamp(
            &payload,
            layout(),
            region(),
            source(17),
            capability(),
            None,
            None,
            calibration(),
            QualityRecord::Unknown,
            4,
        )
        .unwrap();
    assert_eq!(first.measurement().timestamp(), None);
    assert_eq!(first.sequence_event(), SequenceEvent::First);
    assert_eq!(first.sequence(), 4);
    assert_eq!(stream.last_observation(), Some((4, None)));
    assert_eq!(stream.last_sequence(), None);
    assert_eq!(
        stream.ingest_with_optional_timestamp(
            &payload,
            layout(),
            region(),
            source(17),
            capability(),
            None,
            None,
            calibration(),
            QualityRecord::Unknown,
            4,
        ),
        Err(ContractError::InvalidSource)
    );

    let next = stream
        .ingest_with_optional_timestamp(
            &payload,
            layout(),
            region(),
            source(17),
            capability(),
            Some(timestamp(5)),
            None,
            calibration(),
            QualityRecord::Unknown,
            5,
        )
        .unwrap();
    assert_eq!(next.measurement().timestamp(), Some(timestamp(5)));
    assert_eq!(next.sequence(), 5);
    assert_eq!(stream.last_observation(), Some((5, Some(timestamp(5)))));
}

#[test]
fn extended_ingest_preserves_unknown_time_and_order_state() {
    let capability = crate::extended_capability::ExtendedCapability::new(
        701,
        702,
        None,
        SampleEncoding::ComplexF32,
        [4, 0, 0, 0],
        None,
    )
    .unwrap();
    let layout =
        SampleLayout::new(1, [2, 0, 0, 0], [8, 0, 0, 0], SampleEncoding::ComplexF32).unwrap();
    let region = SampleRegion::new(BufferToken::new(91, 1).unwrap(), 0, 16, 16, 4).unwrap();
    let mut stream = ExtendedIngestStream::new(17, 26, 3, capability).unwrap();
    let frame = stream
        .ingest_with_optional_timestamp(
            &[0; 16],
            layout,
            region,
            source(17),
            None,
            None,
            calibration(),
            QualityRecord::Unknown,
            4,
        )
        .unwrap();
    assert_eq!(frame.measurement().timestamp(), None);
    assert_eq!(frame.sequence_event(), SequenceEvent::First);
    assert_eq!(frame.sequence(), 4);
    assert_eq!(stream.last_observation(), Some((4, None)));
}

#[test]
fn sequence_gaps_are_returned_with_the_current_payload() {
    let mut stream = stream();
    let first = [1; 16];
    let after_gap = [2; 16];
    assert_eq!(
        ingest(&mut stream, &first, 17, 4).unwrap().sequence_event(),
        SequenceEvent::First
    );

    let frame = ingest(&mut stream, &after_gap, 17, 6).unwrap();
    assert_eq!(frame.sequence_event(), SequenceEvent::Gap { missing: 1 });
    assert_eq!(frame.sequence(), 6);
    assert_eq!(frame.payload(), &after_gap);
    assert_eq!(stream.last_sequence(), Some((6, timestamp(6))));
}

#[test]
fn wrong_source_or_payload_extent_does_not_advance_sequence() {
    let mut stream = stream();
    let payload = [0xA5; 16];
    assert_eq!(
        ingest(&mut stream, &payload, 18, 4),
        Err(ContractError::InvalidSource)
    );
    assert_eq!(
        ingest(&mut stream, &payload[..15], 17, 4),
        Err(ContractError::OutOfBounds)
    );
    let oversized = [0xA5; 17];
    assert_eq!(
        ingest(&mut stream, &oversized, 17, 4),
        Err(ContractError::OutOfBounds)
    );
    assert_eq!(stream.last_sequence(), None);
    assert_eq!(
        ingest(&mut stream, &payload, 17, 4)
            .unwrap()
            .sequence_event(),
        SequenceEvent::First
    );
}

#[test]
fn layout_encoding_and_rank_failures_leave_order_unchanged() {
    let mut stream = stream();
    let payload = [0u8; 16];
    let wrong_encoding =
        SampleLayout::new(1, [2, 0, 0, 0], [4, 0, 0, 0], SampleEncoding::F32).unwrap();
    assert_eq!(
        stream.ingest(
            &[0; 8],
            wrong_encoding,
            region(),
            source(17),
            capability(),
            timestamp(4),
            None,
            calibration(),
            QualityRecord::Unknown,
            4,
        ),
        Err(ContractError::IncompatibleEncoding)
    );
    let wrong_rank =
        SampleLayout::new(2, [2, 1, 0, 0], [8, 16, 0, 0], SampleEncoding::ComplexF32).unwrap();
    assert_eq!(
        stream.ingest(
            &payload,
            wrong_rank,
            region(),
            source(17),
            capability(),
            timestamp(4),
            None,
            calibration(),
            QualityRecord::Unknown,
            4,
        ),
        Err(ContractError::InvalidExtent)
    );
    assert_eq!(stream.last_sequence(), None);
    assert_eq!(
        ingest(&mut stream, &payload, 17, 4)
            .unwrap()
            .sequence_event(),
        SequenceEvent::First
    );
}

#[test]
fn duplicate_or_out_of_order_sequence_does_not_replace_last_frame() {
    let mut stream = stream();
    let payload = [0u8; 16];
    ingest(&mut stream, &payload, 17, 4).unwrap();
    assert_eq!(
        ingest(&mut stream, &payload, 17, 4),
        Err(ContractError::InvalidSource)
    );
    assert_eq!(stream.last_sequence(), Some((4, timestamp(4))));
}

fn extended_capability() -> ExtendedCapability {
    ExtendedCapability::new(
        801,
        802,
        None,
        SampleEncoding::I16,
        [4, 0, 0, 0],
        Some(SampleRate::new(20.0).unwrap()),
    )
    .unwrap()
}

fn extended_layout() -> SampleLayout {
    SampleLayout::new(1, [2, 0, 0, 0], [2, 0, 0, 0], SampleEncoding::I16).unwrap()
}

fn extended_region() -> SampleRegion {
    SampleRegion::new(BufferToken::new(91, 1).unwrap(), 0, 4, 4, 2).unwrap()
}

fn extended_stream() -> ExtendedIngestStream {
    ExtendedIngestStream::new(31, 41, 5, extended_capability()).unwrap()
}

#[test]
fn extended_ingest_preserves_unknown_unit_and_sequence_gap() {
    let mut stream = extended_stream();
    let payload = [0x10, 0x11, 0x12, 0x13];
    let first = stream
        .ingest(
            &payload,
            extended_layout(),
            extended_region(),
            source(31),
            timestamp(10),
            None,
            calibration(),
            QualityRecord::Unknown,
            10,
        )
        .unwrap();
    assert_eq!(first.sequence_event(), SequenceEvent::First);
    assert_eq!(first.payload(), &payload);
    assert_eq!(first.stream_id(), 41);
    assert_eq!(first.generation(), 5);
    assert_eq!(first.measurement().capability().modality_id(), 801);
    assert_eq!(first.measurement().capability().quantity_id(), 802);
    assert_eq!(first.measurement().capability().unit_id(), None);
    assert_eq!(first.measurement().sample_rate(), None);
    assert_eq!(first.measurement().quality(), QualityRecord::Unknown);

    let next = stream
        .ingest(
            &payload,
            extended_layout(),
            extended_region(),
            source(31),
            timestamp(12),
            None,
            calibration(),
            QualityRecord::Unknown,
            12,
        )
        .unwrap();
    assert_eq!(next.sequence_event(), SequenceEvent::Gap { missing: 1 });
    assert_eq!(stream.last_sequence(), Some((12, timestamp(12))));
}

#[test]
fn extended_ingest_rejects_bad_source_bytes_rate_and_order_transactionally() {
    let mut stream = extended_stream();
    let payload = [0x10, 0x11, 0x12, 0x13];
    assert_eq!(
        stream.ingest(
            &payload,
            extended_layout(),
            extended_region(),
            source(32),
            timestamp(10),
            None,
            calibration(),
            QualityRecord::Unknown,
            10
        ),
        Err(ContractError::InvalidSource)
    );
    assert_eq!(
        stream.ingest(
            &payload[..3],
            extended_layout(),
            extended_region(),
            source(31),
            timestamp(10),
            None,
            calibration(),
            QualityRecord::Unknown,
            10
        ),
        Err(ContractError::OutOfBounds)
    );
    assert_eq!(
        stream.ingest(
            &payload,
            extended_layout(),
            extended_region(),
            source(31),
            timestamp(10),
            Some(SampleRate::new(21.0).unwrap()),
            calibration(),
            QualityRecord::Unknown,
            10
        ),
        Err(ContractError::InvalidRate)
    );
    assert_eq!(stream.last_sequence(), None);
    assert_eq!(
        stream
            .ingest(
                &payload,
                extended_layout(),
                extended_region(),
                source(31),
                timestamp(10),
                None,
                calibration(),
                QualityRecord::Unknown,
                10
            )
            .unwrap()
            .sequence_event(),
        SequenceEvent::First
    );
    assert_eq!(stream.last_sequence(), Some((10, timestamp(10))));
}
