//! Explicit unit conversion for compatible linear units.
//!
//! Transformations needing a physical model or logarithm are unsupported.
//! The result says nothing about the accuracy or calibration of the input.
use crate::{quantity::{PhysicalQuantity, Unit}, ContractError};

/// Convert a finite value between explicitly declared units of one quantity.
/// Magnetic field strength and flux density are intentionally not converted
/// without a material model. Logarithmic power conversion is not inferred.
pub fn convert(value: f64, quantity: PhysicalQuantity, from: Unit, to: Unit) -> Result<f64, ContractError> {
    if !value.is_finite() { return Err(ContractError::InvalidValue); }
    if !quantity.accepts_unit(from) || !quantity.accepts_unit(to) {
        return Err(ContractError::IncompatibleUnit);
    }
    if from == to { return Ok(value); }
    if quantity == PhysicalQuantity::Temperature {
        let converted = match (from, to) {
            (Unit::Celsius, Unit::Kelvin) => value + 273.15,
            (Unit::Kelvin, Unit::Celsius) => value - 273.15,
            _ => return Err(ContractError::UnsupportedOperation),
        };
        return if converted.is_finite() { Ok(converted) } else { Err(ContractError::ArithmeticOverflow) };
    }
    let (from_group, from_scale) = linear_scale(from).ok_or(ContractError::UnsupportedOperation)?;
    let (to_group, to_scale) = linear_scale(to).ok_or(ContractError::UnsupportedOperation)?;
    if from_group != to_group { return Err(ContractError::UnsupportedOperation); }
    let converted = value * (from_scale / to_scale);
    if !converted.is_finite() || (value != 0.0 && converted == 0.0) {
        return Err(ContractError::ArithmeticOverflow);
    }
    Ok(converted)
}

// Group IDs are local to this module, not ontology or serialized identifiers.
fn linear_scale(unit: Unit) -> Option<(u8, f64)> {
    match unit {
        Unit::Volt => Some((1, 1.0)), Unit::Millivolt => Some((1, 0.001)),
        Unit::VoltPerMeter => Some((2, 1.0)), Unit::MillivoltPerMeter => Some((2, 0.001)),
        Unit::Tesla => Some((3, 1.0)), Unit::Microtesla => Some((3, 1e-6)), Unit::Nanotesla => Some((3, 1e-9)),
        Unit::Watt => Some((4, 1.0)), Unit::Milliwatt => Some((4, 0.001)),
        Unit::Meter => Some((5, 1.0)), Unit::Centimeter => Some((5, 0.01)), Unit::Millimeter => Some((5, 0.001)),
        Unit::MeterPerSecond => Some((6, 1.0)), Unit::KilometerPerHour => Some((6, 1.0 / 3.6)),
        Unit::Hertz => Some((7, 1.0)), Unit::Kilohertz => Some((7, 1e3)), Unit::Megahertz => Some((7, 1e6)), Unit::Gigahertz => Some((7, 1e9)),
        _ => None,
    }
}

#[cfg(test)]
#[path = "conversion_tests.rs"]
mod tests;
