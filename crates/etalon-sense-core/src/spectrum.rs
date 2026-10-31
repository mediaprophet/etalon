use crate::{
    capability::{Modality, SensorCapability},
    quantity::{SpectralAxisKind, SpectralBand},
    ContractError,
};

/// Acoustic-frequency range in hertz. It is separate from electromagnetic
/// carrier/baseband/modulation axes so microphone spectra cannot be labeled as
/// EMF coverage by this contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcousticBand {
    lower_hz: f64,
    upper_hz: f64,
}

impl AcousticBand {
    pub fn new(lower_hz: f64, upper_hz: f64) -> Result<Self, ContractError> {
        if !lower_hz.is_finite() || !upper_hz.is_finite() || lower_hz < 0.0 || lower_hz >= upper_hz
        {
            return Err(ContractError::InvalidBand);
        }
        Ok(Self { lower_hz, upper_hz })
    }

    pub const fn lower_hz(self) -> f64 {
        self.lower_hz
    }

    pub const fn upper_hz(self) -> f64 {
        self.upper_hz
    }
}

/// Declared physical spectral coverage for a modality. `None` means the range
/// is unknown or was not supplied; this type does not infer coverage from a
/// device, sample rate, modulation, or another axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralCoverage {
    modality: Modality,
    carrier: Option<SpectralBand>,
    baseband: Option<SpectralBand>,
    modulation: Option<SpectralBand>,
    wavelength: Option<SpectralBand>,
    acoustic: Option<AcousticBand>,
}

impl SpectralCoverage {
    pub fn new(
        modality: Modality,
        carrier: Option<SpectralBand>,
        baseband: Option<SpectralBand>,
        modulation: Option<SpectralBand>,
        wavelength: Option<SpectralBand>,
        acoustic: Option<AcousticBand>,
    ) -> Result<Self, ContractError> {
        validate_band_kind(carrier, SpectralAxisKind::CarrierHertz)?;
        validate_band_kind(baseband, SpectralAxisKind::BasebandHertz)?;
        validate_band_kind(modulation, SpectralAxisKind::ModulationHertz)?;
        validate_band_kind(wavelength, SpectralAxisKind::WavelengthMeters)?;

        let electromagnetic_modality = matches!(
            modality,
            Modality::Wifi
                | Modality::Bluetooth
                | Modality::Cellular
                | Modality::Sdr
                | Modality::Radar
                | Modality::FieldInstrument
                | Modality::VisibleCamera
                | Modality::NearInfraredCamera
                | Modality::ThermalCamera
        );

        if (carrier.is_some() || baseband.is_some() || modulation.is_some())
            && !electromagnetic_modality
        {
            return Err(ContractError::InvalidBand);
        }
        if wavelength.is_some() && !electromagnetic_modality {
            return Err(ContractError::InvalidBand);
        }
        if acoustic.is_some() && modality != Modality::Microphone {
            return Err(ContractError::InvalidBand);
        }
        Ok(Self {
            modality,
            carrier,
            baseband,
            modulation,
            wavelength,
            acoustic,
        })
    }

    /// Confirm that this declaration belongs to the supplied stream modality.
    /// This does not assert that hardware supports the declared ranges.
    pub fn validate_for(&self, capability: SensorCapability) -> Result<(), ContractError> {
        if self.modality != capability.modality() {
            return Err(ContractError::UnsupportedOperation);
        }
        Ok(())
    }

    pub const fn modality(self) -> Modality {
        self.modality
    }

    pub const fn carrier(self) -> Option<SpectralBand> {
        self.carrier
    }

    pub const fn baseband(self) -> Option<SpectralBand> {
        self.baseband
    }

    pub const fn modulation(self) -> Option<SpectralBand> {
        self.modulation
    }

    pub const fn wavelength(self) -> Option<SpectralBand> {
        self.wavelength
    }

    pub const fn acoustic(self) -> Option<AcousticBand> {
        self.acoustic
    }
}

fn validate_band_kind(
    band: Option<SpectralBand>,
    expected: SpectralAxisKind,
) -> Result<(), ContractError> {
    if let Some(band) = band {
        if band.kind() != expected {
            return Err(ContractError::InvalidBand);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "spectrum_tests.rs"]
mod tests;
