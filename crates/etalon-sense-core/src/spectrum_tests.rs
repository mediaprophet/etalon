use super::{AcousticBand, SpectralCoverage};
use crate::{
    capability::{Modality, SensorCapability},
    quantity::{PhysicalQuantity, SampleEncoding, SpectralAxisKind, SpectralBand, Unit},
    ContractError,
};

fn band(kind: SpectralAxisKind, lower: f64, upper: f64) -> SpectralBand {
    SpectralBand::new(kind, lower, upper).unwrap()
}

fn unknown(modality: Modality) -> SpectralCoverage {
    SpectralCoverage::new(modality, None, None, None, None, None).unwrap()
}

#[test]
fn unknown_coverage_is_preserved_without_inference() {
    let coverage = unknown(Modality::Wifi);
    assert_eq!(coverage.carrier(), None);
    assert_eq!(coverage.baseband(), None);
    assert_eq!(coverage.modulation(), None);
    assert_eq!(coverage.wavelength(), None);
    assert_eq!(coverage.acoustic(), None);

    let microphone = unknown(Modality::Microphone);
    assert_eq!(microphone.carrier(), None);
    assert_eq!(microphone.acoustic(), None);
}

#[test]
fn electromagnetic_axes_are_distinct_and_preserved() {
    let carrier = band(SpectralAxisKind::CarrierHertz, 2.4e9, 2.5e9);
    let baseband = band(SpectralAxisKind::BasebandHertz, -20e6, 20e6);
    let modulation = band(SpectralAxisKind::ModulationHertz, 0.0, 10e6);
    let wavelength = band(SpectralAxisKind::WavelengthMeters, 0.11, 0.13);
    let coverage = SpectralCoverage::new(
        Modality::Sdr,
        Some(carrier),
        Some(baseband),
        Some(modulation),
        Some(wavelength),
        None,
    )
    .unwrap();
    assert_eq!(coverage.carrier(), Some(carrier));
    assert_eq!(coverage.baseband(), Some(baseband));
    assert_eq!(coverage.modulation(), Some(modulation));
    assert_eq!(coverage.wavelength(), Some(wavelength));
    assert_eq!(coverage.acoustic(), None);
}

#[test]
fn acoustic_spectrum_has_its_own_axis_and_cannot_be_an_emf_band() {
    let acoustic = AcousticBand::new(20.0, 20_000.0).unwrap();
    let coverage =
        SpectralCoverage::new(Modality::Microphone, None, None, None, None, Some(acoustic))
            .unwrap();
    assert_eq!(coverage.acoustic(), Some(acoustic));
    assert_eq!(coverage.carrier(), None);
    assert_eq!(coverage.baseband(), None);
    assert_eq!(coverage.modulation(), None);

    assert_eq!(
        SpectralCoverage::new(Modality::Wifi, None, None, None, None, Some(acoustic),),
        Err(ContractError::InvalidBand)
    );
    assert_eq!(
        SpectralCoverage::new(
            Modality::Microphone,
            Some(band(SpectralAxisKind::CarrierHertz, 1.0, 2.0)),
            None,
            None,
            None,
            None,
        ),
        Err(ContractError::InvalidBand)
    );
}

#[test]
fn axes_must_match_their_declared_dimensions() {
    assert_eq!(
        SpectralCoverage::new(
            Modality::Wifi,
            Some(band(SpectralAxisKind::BasebandHertz, -1.0, 1.0)),
            None,
            None,
            None,
            None,
        ),
        Err(ContractError::InvalidBand)
    );
    assert_eq!(
        SpectralCoverage::new(Modality::VisibleCamera, None, None, None, None, None,),
        Ok(unknown(Modality::VisibleCamera))
    );
    assert_eq!(
        SpectralCoverage::new(
            Modality::Microphone,
            None,
            None,
            None,
            Some(band(SpectralAxisKind::WavelengthMeters, 400e-9, 700e-9)),
            None,
        ),
        Err(ContractError::InvalidBand)
    );
}

#[test]
fn invalid_acoustic_bounds_are_rejected() {
    assert_eq!(
        AcousticBand::new(-1.0, 1.0),
        Err(ContractError::InvalidBand)
    );
    assert_eq!(AcousticBand::new(2.0, 2.0), Err(ContractError::InvalidBand));
    assert!(AcousticBand::new(0.0, 1.0).is_ok());
}

#[test]
fn wavelength_can_be_reported_for_em_or_optical_but_is_not_derived() {
    let wavelength = band(SpectralAxisKind::WavelengthMeters, 400e-9, 700e-9);
    let visible = SpectralCoverage::new(
        Modality::VisibleCamera,
        None,
        None,
        None,
        Some(wavelength),
        None,
    )
    .unwrap();
    assert_eq!(visible.wavelength(), Some(wavelength));
    assert_eq!(visible.carrier(), None);

    let rf = SpectralCoverage::new(
        Modality::Wifi,
        None,
        None,
        None,
        Some(band(SpectralAxisKind::WavelengthMeters, 0.1, 0.2)),
        None,
    )
    .unwrap();
    assert_eq!(rf.wavelength().unwrap().lower(), 0.1);
    assert_eq!(rf.carrier(), None);
}

#[test]
fn coverage_must_match_the_stream_modality_without_claiming_device_support() {
    let coverage = unknown(Modality::Microphone);
    let microphone = SensorCapability::new(
        Modality::Microphone,
        PhysicalQuantity::AcousticPressure,
        Unit::Pascal,
        SampleEncoding::F32,
        [1024, 0, 0, 0],
        None,
    )
    .unwrap();
    assert_eq!(coverage.validate_for(microphone), Ok(()));

    let wifi = SensorCapability::new(
        Modality::Wifi,
        PhysicalQuantity::RfChannelResponse,
        Unit::Unitless,
        SampleEncoding::ComplexI16,
        [64, 0, 0, 0],
        None,
    )
    .unwrap();
    assert_eq!(
        coverage.validate_for(wifi),
        Err(ContractError::UnsupportedOperation)
    );
}
