use super::*;

#[test]
fn encoding_sizes_and_alignment_match_scalar_layouts() {
    assert_eq!(SampleEncoding::U8.element_bytes(), 1);
    assert_eq!(SampleEncoding::U16.element_bytes(), 2);
    assert_eq!(SampleEncoding::I16.element_bytes(), 2);
    assert_eq!(SampleEncoding::I32.element_bytes(), 4);
    assert_eq!(SampleEncoding::F32.element_bytes(), 4);
    assert_eq!(SampleEncoding::F64.element_bytes(), 8);
    assert_eq!(SampleEncoding::ComplexI16.element_bytes(), 4);
    assert_eq!(SampleEncoding::ComplexF32.element_bytes(), 8);
    assert_eq!(SampleEncoding::ComplexI16.alignment_bytes(), 2);
    assert_eq!(SampleEncoding::U16.alignment_bytes(), 2);
    assert_eq!(SampleEncoding::ComplexF32.alignment_bytes(), 4);
    assert!(SampleEncoding::ComplexF32.is_complex());
    assert!(!SampleEncoding::F32.is_complex());
    assert!(!SampleEncoding::U16.is_complex());
}

#[test]
fn unit_mapping_rejects_incompatible_dimensions() {
    assert!(PhysicalQuantity::ElectricField.accepts_unit(Unit::VoltPerMeter));
    assert!(PhysicalQuantity::Power.accepts_unit(Unit::DecibelMilliwatt));
    assert!(PhysicalQuantity::Temperature.accepts_unit(Unit::Celsius));
    assert!(!PhysicalQuantity::Power.accepts_unit(Unit::Pascal));
    assert!(!PhysicalQuantity::Distance.accepts_unit(Unit::MeterPerSecond));
    assert!(!PhysicalQuantity::Frequency.accepts_unit(Unit::Watt));
}

#[test]
fn channel_response_requires_complex_encoding() {
    assert!(PhysicalQuantity::RfChannelResponse.accepts_encoding(SampleEncoding::ComplexF32));
    assert!(!PhysicalQuantity::RfChannelResponse.accepts_encoding(SampleEncoding::F32));
    assert!(!PhysicalQuantity::Power.accepts_encoding(SampleEncoding::ComplexI16));
}

#[test]
fn spectral_bands_enforce_finite_order_and_axis_sign() {
    assert!(SpectralBand::new(SpectralAxisKind::BasebandHertz, -20.0, 20.0).is_ok());
    assert!(SpectralBand::new(SpectralAxisKind::CarrierHertz, 2.4e9, 2.5e9).is_ok());
    assert!(SpectralBand::new(SpectralAxisKind::WavelengthMeters, 400e-9, 700e-9).is_ok());
    assert_eq!(
        SpectralBand::new(SpectralAxisKind::CarrierHertz, -1.0, 1.0),
        Err(ContractError::InvalidBand)
    );
    assert_eq!(
        SpectralBand::new(SpectralAxisKind::WavelengthMeters, 0.0, 1.0),
        Err(ContractError::InvalidBand)
    );
    assert_eq!(
        SpectralBand::new(SpectralAxisKind::BasebandHertz, 1.0, 1.0),
        Err(ContractError::InvalidBand)
    );
    assert!(SpectralBand::new(SpectralAxisKind::ModulationHertz, f64::NAN, 1.0).is_err());
}

#[test]
fn sample_rate_must_be_finite_and_positive() {
    assert_eq!(SampleRate::new(48_000.0).unwrap().hz(), 48_000.0);
    assert_eq!(SampleRate::new(0.0), Err(ContractError::InvalidRate));
    assert_eq!(SampleRate::new(-1.0), Err(ContractError::InvalidRate));
    assert_eq!(SampleRate::new(f64::INFINITY), Err(ContractError::InvalidRate));
}
