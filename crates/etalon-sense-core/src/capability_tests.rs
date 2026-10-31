use super::*;

#[test]
fn wifi_rssi_declaration_does_not_claim_channel_response() {
    let rssi = SensorCapability::new(
        Modality::Wifi,
        PhysicalQuantity::Power,
        Unit::DecibelMilliwatt,
        SampleEncoding::F32,
        [1, 0, 0, 0],
        Some(SampleRate::new(10.0).unwrap()),
    )
    .unwrap();

    assert_eq!(rssi.modality(), Modality::Wifi);
    assert_eq!(rssi.quantity(), PhysicalQuantity::Power);
    assert_ne!(rssi.quantity(), PhysicalQuantity::RfChannelResponse);
    assert_eq!(rssi.encoding(), SampleEncoding::F32);
    assert_eq!(rssi.max_rank(), 1);
}

#[test]
fn channel_response_must_preserve_complex_samples() {
    assert_eq!(
        SensorCapability::new(
            Modality::Wifi,
            PhysicalQuantity::RfChannelResponse,
            Unit::Unitless,
            SampleEncoding::F32,
            [64, 0, 0, 0],
            None,
        ),
        Err(ContractError::IncompatibleEncoding)
    );

    let csi = SensorCapability::new(
        Modality::Wifi,
        PhysicalQuantity::RfChannelResponse,
        Unit::Unitless,
        SampleEncoding::ComplexF32,
        [64, 0, 0, 0],
        Some(SampleRate::new(1_000.0).unwrap()),
    )
    .unwrap();
    assert_eq!(csi.max_payload_bytes(), 64 * 8);
    assert_eq!(csi.max_rank(), 1);
}

#[test]
fn capability_rejects_unit_and_extent_errors() {
    assert_eq!(
        SensorCapability::new(
            Modality::Microphone,
            PhysicalQuantity::AcousticPressure,
            Unit::Meter,
            SampleEncoding::F32,
            [256, 0, 0, 0],
            None,
        ),
        Err(ContractError::IncompatibleUnit)
    );
    assert_eq!(
        SensorCapability::new(
            Modality::Radar,
            PhysicalQuantity::Distance,
            Unit::Meter,
            SampleEncoding::F32,
            [32, 0, 1, 0],
            None,
        ),
        Err(ContractError::InvalidExtent)
    );
}

#[test]
fn capability_computes_bounded_payload_size() {
    let image = SensorCapability::new(
        Modality::VisibleCamera,
        PhysicalQuantity::Dimensionless,
        Unit::Unitless,
        SampleEncoding::U8,
        [1920, 1080, 3, 0],
        None,
    )
    .unwrap();
    assert_eq!(image.max_payload_bytes(), 1920 * 1080 * 3);
    assert_eq!(image.max_rank(), 3);

    let radar_cube = SensorCapability::new(
        Modality::Radar,
        PhysicalQuantity::Distance,
        Unit::Meter,
        SampleEncoding::F32,
        [64, 32, 4, 16],
        None,
    )
    .unwrap();
    assert_eq!(radar_cube.max_rank(), 4);

    assert_eq!(
        SensorCapability::new(
            Modality::Sdr,
            PhysicalQuantity::RfChannelResponse,
            Unit::Unitless,
            SampleEncoding::ComplexF32,
            [u32::MAX, u32::MAX, u32::MAX, u32::MAX],
            None,
        ),
        Err(ContractError::ArithmeticOverflow)
    );
}
