use super::*;
use crate::{
    buffer::BufferToken,
    capability::Modality,
    quality::{CalibrationState, SourceKind, TrustState},
    quantity::{PhysicalQuantity, SampleEncoding, Unit},
};

fn parts() -> (
    SensorCapability,
    SampleLayout,
    SampleRegion,
    SourceDescriptor,
    ClockStamp,
    CalibrationRecord,
) {
    let capability = SensorCapability::new(
        Modality::Wifi,
        PhysicalQuantity::RfChannelResponse,
        Unit::Unitless,
        SampleEncoding::ComplexF32,
        [64, 0, 0, 0],
        Some(SampleRate::new(100.0).unwrap()),
    )
    .unwrap();
    let layout =
        SampleLayout::new(1, [32, 0, 0, 0], [8, 0, 0, 0], SampleEncoding::ComplexF32).unwrap();
    let region = SampleRegion::new(BufferToken::new(1, 1).unwrap(), 0, 256, 256, 4).unwrap();
    let source = SourceDescriptor::new(1, SourceKind::Recorded).unwrap();
    let time = ClockStamp::new(1, 0, 1000, 10).unwrap();
    let calibration = CalibrationRecord::new(CalibrationState::Missing).unwrap();
    (capability, layout, region, source, time, calibration)
}

#[test]
fn preserves_unknowns_and_does_not_claim_trust() {
    let (c, l, r, s, t, cal) = parts();
    let sample =
        Measurement::new(c, l.clone(), r, s, t, None, cal, QualityRecord::Unknown).unwrap();
    assert_eq!(sample.sample_rate(), None);
    assert_eq!(sample.quality(), QualityRecord::Unknown);
    assert_eq!(sample.source().trust_state(), TrustState::Unverified);
    assert_eq!(sample.calibration().state(), CalibrationState::Missing);
    let untimed = Measurement::new_with_optional_timestamp(
        c,
        l,
        r,
        s,
        None,
        None,
        cal,
        QualityRecord::Unknown,
    )
    .unwrap();
    assert_eq!(untimed.timestamp(), None);
}

#[test]
fn rejects_rate_extent_encoding_and_region_mismatch() {
    let (c, l, r, s, t, cal) = parts();
    assert_eq!(
        Measurement::new(
            c,
            l.clone(),
            r,
            s,
            t,
            Some(SampleRate::new(101.0).unwrap()),
            cal,
            QualityRecord::Unknown
        ),
        Err(ContractError::InvalidRate)
    );
    let large =
        SampleLayout::new(1, [65, 0, 0, 0], [8, 0, 0, 0], SampleEncoding::ComplexF32).unwrap();
    assert_eq!(
        Measurement::new(c, large, r, s, t, None, cal, QualityRecord::Unknown),
        Err(ContractError::InvalidExtent)
    );
    let wrong = SampleLayout::new(1, [32, 0, 0, 0], [4, 0, 0, 0], SampleEncoding::F32).unwrap();
    assert_eq!(
        Measurement::new(c, wrong, r, s, t, None, cal, QualityRecord::Unknown),
        Err(ContractError::IncompatibleEncoding)
    );
    let small = SampleRegion::new(BufferToken::new(2, 1).unwrap(), 0, 128, 128, 4).unwrap();
    assert_eq!(
        Measurement::new(c, l, small, s, t, None, cal, QualityRecord::Unknown),
        Err(ContractError::OutOfBounds)
    );
}

#[test]
fn extended_measurement_preserves_registry_ids_unknown_unit_and_unknown_metadata() {
    let capability = ExtendedCapability::new(
        701,
        702,
        None,
        SampleEncoding::ComplexF32,
        [8, 0, 0, 0],
        Some(SampleRate::new(200.0).unwrap()),
    )
    .unwrap();
    let layout =
        SampleLayout::new(1, [4, 0, 0, 0], [8, 0, 0, 0], SampleEncoding::ComplexF32).unwrap();
    let region = SampleRegion::new(BufferToken::new(70, 2).unwrap(), 0, 32, 32, 4).unwrap();
    let source = SourceDescriptor::new(71, SourceKind::Recorded).unwrap();
    let timestamp = ClockStamp::new(8, 1, 900, 20).unwrap();
    let calibration = CalibrationRecord::new(CalibrationState::Missing).unwrap();

    let measurement = ExtendedMeasurement::new(
        capability,
        layout.clone(),
        region,
        source,
        timestamp,
        None,
        calibration,
        QualityRecord::Unknown,
    )
    .unwrap();

    assert_eq!(measurement.capability().modality_id(), 701);
    assert_eq!(measurement.capability().quantity_id(), 702);
    assert_eq!(measurement.capability().unit_id(), None);
    assert_eq!(measurement.source().source_id(), 71);
    assert_eq!(measurement.timestamp(), Some(timestamp));
    assert_eq!(measurement.sample_rate(), None);
    assert_eq!(measurement.calibration(), calibration);
    assert_eq!(measurement.quality(), QualityRecord::Unknown);
    let untimed = ExtendedMeasurement::new_with_optional_timestamp(
        capability,
        layout,
        region,
        source,
        None,
        None,
        calibration,
        QualityRecord::Unknown,
    )
    .unwrap();
    assert_eq!(untimed.timestamp(), None);
}

#[test]
fn extended_measurement_checks_encoding_rate_extents_and_region() {
    let capability = ExtendedCapability::new(
        701,
        702,
        Some(703),
        SampleEncoding::I16,
        [8, 0, 0, 0],
        Some(SampleRate::new(100.0).unwrap()),
    )
    .unwrap();
    let region = SampleRegion::new(BufferToken::new(70, 2).unwrap(), 0, 16, 16, 2).unwrap();
    let source = SourceDescriptor::new(71, SourceKind::Recorded).unwrap();
    let timestamp = ClockStamp::new(8, 1, 900, 20).unwrap();
    let calibration = CalibrationRecord::new(CalibrationState::Missing).unwrap();
    let layout = SampleLayout::new(1, [8, 0, 0, 0], [2, 0, 0, 0], SampleEncoding::I16).unwrap();

    assert_eq!(
        ExtendedMeasurement::new(
            capability,
            layout.clone(),
            region,
            source,
            timestamp,
            Some(SampleRate::new(101.0).unwrap()),
            calibration,
            QualityRecord::Unknown
        ),
        Err(ContractError::InvalidRate)
    );
    let wrong_encoding =
        SampleLayout::new(1, [8, 0, 0, 0], [4, 0, 0, 0], SampleEncoding::F32).unwrap();
    assert_eq!(
        ExtendedMeasurement::new(
            capability,
            wrong_encoding,
            region,
            source,
            timestamp,
            None,
            calibration,
            QualityRecord::Unknown
        ),
        Err(ContractError::IncompatibleEncoding)
    );
    let too_large = SampleLayout::new(1, [9, 0, 0, 0], [2, 0, 0, 0], SampleEncoding::I16).unwrap();
    assert_eq!(
        ExtendedMeasurement::new(
            capability,
            too_large,
            region,
            source,
            timestamp,
            None,
            calibration,
            QualityRecord::Unknown
        ),
        Err(ContractError::InvalidExtent)
    );
    let short_region = SampleRegion::new(BufferToken::new(71, 1).unwrap(), 0, 8, 8, 2).unwrap();
    assert_eq!(
        ExtendedMeasurement::new(
            capability,
            layout,
            short_region,
            source,
            timestamp,
            None,
            calibration,
            QualityRecord::Unknown
        ),
        Err(ContractError::OutOfBounds)
    );
}
