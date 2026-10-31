//! Validated physical quantities, units, sample encodings, and spectral axes.

use crate::ContractError;

/// Scalar or complex representation of each captured sample element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleEncoding {
    U8,
    /// Unsigned 16-bit integer sample. Byte order is defined by the enclosing
    /// buffer/format contract; this enum specifies width and scalar semantics.
    U16,
    I16,
    I32,
    F32,
    F64,
    ComplexI16,
    ComplexF32,
}

impl SampleEncoding {
    /// Number of bytes occupied by one scalar or complex sample element.
    pub const fn element_bytes(self) -> u32 {
        match self {
            Self::U8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::ComplexI16 => 4,
            Self::I32 | Self::F32 => 4,
            Self::F64 | Self::ComplexF32 => 8,
        }
    }

    /// Required alignment for one element in a sample buffer.
    pub const fn alignment_bytes(self) -> u32 {
        match self {
            Self::U8 => 1,
            Self::U16 | Self::I16 | Self::ComplexI16 => 2,
            Self::I32 | Self::F32 | Self::ComplexF32 => 4,
            Self::F64 => 8,
        }
    }

    /// Whether each element contains an in-phase and quadrature pair.
    pub const fn is_complex(self) -> bool {
        matches!(self, Self::ComplexI16 | Self::ComplexF32)
    }
}

/// Physical dimension described by a sensor stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicalQuantity {
    Dimensionless,
    RfChannelResponse,
    RfInPhase,
    RfQuadrature,
    ElectricField,
    MagneticField,
    Power,
    PowerSpectralDensity,
    OpticalRadiance,
    OpticalIntensity,
    Temperature,
    AcousticPressure,
    Distance,
    Velocity,
    Frequency,
}

impl PhysicalQuantity {
    /// Whether `unit` is dimensionally compatible with this quantity.
    pub const fn accepts_unit(self, unit: Unit) -> bool {
        match self {
            Self::Dimensionless | Self::RfChannelResponse => matches!(unit, Unit::Unitless),
            Self::RfInPhase | Self::RfQuadrature => matches!(unit, Unit::Volt | Unit::Millivolt),
            Self::ElectricField => matches!(unit, Unit::VoltPerMeter | Unit::MillivoltPerMeter),
            Self::MagneticField => matches!(
                unit,
                Unit::Tesla | Unit::Microtesla | Unit::Nanotesla | Unit::AmperePerMeter
            ),
            Self::Power => matches!(unit, Unit::Watt | Unit::Milliwatt | Unit::DecibelMilliwatt),
            Self::PowerSpectralDensity => matches!(
                unit,
                Unit::WattPerHertz | Unit::DecibelMilliwattPerHertz
            ),
            Self::OpticalRadiance => matches!(unit, Unit::WattPerSquareMeterSteradian),
            Self::OpticalIntensity => {
                matches!(unit, Unit::WattPerSquareMeter | Unit::WattPerSteradian)
            }
            Self::Temperature => matches!(unit, Unit::Kelvin | Unit::Celsius),
            Self::AcousticPressure => matches!(unit, Unit::Pascal),
            Self::Distance => matches!(unit, Unit::Meter | Unit::Centimeter | Unit::Millimeter),
            Self::Velocity => matches!(unit, Unit::MeterPerSecond | Unit::KilometerPerHour),
            Self::Frequency => matches!(
                unit,
                Unit::Hertz | Unit::Kilohertz | Unit::Megahertz | Unit::Gigahertz
            ),
        }
    }

    /// Whether this quantity can be represented by `encoding` without
    /// silently dropping a complex channel response.
    pub const fn accepts_encoding(self, encoding: SampleEncoding) -> bool {
        match self {
            Self::RfChannelResponse => encoding.is_complex(),
            _ => !encoding.is_complex(),
        }
    }
}

/// Unit attached to a sensor value. Conversion is explicit and is not
/// performed by the capability declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Unitless,
    Volt,
    Millivolt,
    VoltPerMeter,
    MillivoltPerMeter,
    Tesla,
    Microtesla,
    Nanotesla,
    AmperePerMeter,
    Watt,
    Milliwatt,
    DecibelMilliwatt,
    WattPerHertz,
    DecibelMilliwattPerHertz,
    WattPerSquareMeter,
    WattPerSteradian,
    WattPerSquareMeterSteradian,
    Kelvin,
    Celsius,
    Pascal,
    Meter,
    Centimeter,
    Millimeter,
    MeterPerSecond,
    KilometerPerHour,
    Hertz,
    Kilohertz,
    Megahertz,
    Gigahertz,
}

/// Kind of physical axis used to label a spectral measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectralAxisKind {
    CarrierHertz,
    BasebandHertz,
    ModulationHertz,
    WavelengthMeters,
}

/// Finite, ordered bounds on one physical spectral axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralBand {
    kind: SpectralAxisKind,
    lower: f64,
    upper: f64,
}

impl SpectralBand {
    /// Construct an ordered band. Baseband bounds may be signed; carrier and
    /// modulation frequencies must be nonnegative; wavelengths must be positive.
    pub fn new(kind: SpectralAxisKind, lower: f64, upper: f64) -> Result<Self, ContractError> {
        if !lower.is_finite() || !upper.is_finite() || lower >= upper {
            return Err(ContractError::InvalidBand);
        }
        let bounds_valid = match kind {
            SpectralAxisKind::CarrierHertz | SpectralAxisKind::ModulationHertz => lower >= 0.0,
            SpectralAxisKind::BasebandHertz => true,
            SpectralAxisKind::WavelengthMeters => lower > 0.0,
        };
        if !bounds_valid {
            return Err(ContractError::InvalidBand);
        }
        Ok(Self { kind, lower, upper })
    }

    pub const fn kind(self) -> SpectralAxisKind {
        self.kind
    }

    pub const fn lower(self) -> f64 {
        self.lower
    }

    pub const fn upper(self) -> f64 {
        self.upper
    }
}

/// Positive sample frequency in hertz.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleRate(f64);

impl SampleRate {
    pub fn new(hz: f64) -> Result<Self, ContractError> {
        if !hz.is_finite() || hz <= 0.0 {
            return Err(ContractError::InvalidRate);
        }
        Ok(Self(hz))
    }

    pub const fn hz(self) -> f64 {
        self.0
    }
}

#[cfg(test)]
#[path = "quantity_tests.rs"]
mod tests;
