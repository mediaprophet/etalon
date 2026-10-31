use super::{
    AffineCalibrationProfile, CalibrationCondition, CalibrationDatum, CalibrationError,
    CalibrationInterval,
};
use crate::{
    authority::TrustedCaptureClock,
    observation::{MeasurementUncertainty, QuantityUnitIds},
    processing::OperationId,
    quality::CalibrationArtifactId,
    time::ClockStamp,
};
use core::num::NonZeroU64;

/// Applied arithmetic with the original envelope intact. This does not upgrade
/// source trust, authorization, raw quality or calibration declarations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationApplication {
    pub original: CalibrationDatum,
    pub value: f64,
    pub quantity_unit: QuantityUnitIds,
    pub uncertainty: MeasurementUncertainty,
    pub artifact_id: CalibrationArtifactId,
    pub method: OperationId,
    pub environment_id: NonZeroU64,
    pub applied_at: ClockStamp,
}

/// Apply within declared device/environment/condition/value/time bounds. The
/// caller composes this numeric operation with its existing host policy gate.
/// Output is cleared on every failure. Clocks require an explicit common epoch.
pub fn apply_affine_calibration(
    profile: &AffineCalibrationProfile<'_, '_>,
    input: CalibrationDatum,
    environment: Option<NonZeroU64>,
    conditions: &[CalibrationCondition],
    clock: &mut impl TrustedCaptureClock,
    output: &mut Option<CalibrationApplication>,
) -> Result<(), CalibrationError> {
    *output = None;
    profile.check_context(input, environment, conditions)?;
    let specification = profile.specification();
    let raw = input.value().ok_or(CalibrationError::MissingValue)?;
    if !specification.input_range.contains(raw) {
        return Err(CalibrationError::OutsideRange);
    }
    let sample_time = input
        .envelope()
        .timestamp()
        .ok_or(CalibrationError::UnknownTime)?;
    specification.validity.check(sample_time)?;
    let now = clock
        .now()
        .map_err(|_| CalibrationError::ClockUnavailable)?;
    specification.validity.check(now)?;
    if sample_time.interval().lower_ns() > now.interval().upper_ns() {
        return Err(CalibrationError::OutsideValidity);
    }
    let range = input_interval(raw, input.envelope().uncertainty(), specification.input)?;
    if range.is_some_and(|range| !specification.input_range.encloses(range)) {
        return Err(CalibrationError::OutsideRange);
    }
    let value = specification.coefficients.map(raw)?;
    let uncertainty = match (range, specification.model_error_bound) {
        (Some(range), Some(error)) => {
            let gain = specification.coefficients.gain();
            let offset = specification.coefficients.offset();
            let (low, high) = if gain >= 0.0 {
                (range.lower(), range.upper())
            } else {
                (range.upper(), range.lower())
            };
            let lower = directed(
                directed(directed(low * gain, false)? + offset, false)? - error,
                false,
            )?;
            let upper = directed(
                directed(directed(high * gain, true)? + offset, true)? + error,
                true,
            )?;
            MeasurementUncertainty::Interval {
                lower,
                upper,
                unit_id: specification.output.unit_id(),
            }
        }
        _ => MeasurementUncertainty::Unknown,
    };
    *output = Some(CalibrationApplication {
        original: input,
        value,
        quantity_unit: specification.output,
        uncertainty,
        artifact_id: specification.artifact_id,
        method: specification.method,
        environment_id: specification.environment_id,
        applied_at: now,
    });
    Ok(())
}

fn input_interval(
    value: f64,
    uncertainty: MeasurementUncertainty,
    semantics: QuantityUnitIds,
) -> Result<Option<CalibrationInterval>, CalibrationError> {
    let (lower, upper) = match uncertainty {
        MeasurementUncertainty::Unknown => return Ok(None),
        MeasurementUncertainty::AbsoluteBound { magnitude, unit_id } => {
            if unit_id != semantics.unit_id() || !magnitude.is_finite() || magnitude < 0.0 {
                return Err(CalibrationError::InvalidValue);
            }
            if magnitude == 0.0 {
                (value, value)
            } else {
                (
                    directed(value - magnitude, false)?,
                    directed(value + magnitude, true)?,
                )
            }
        }
        MeasurementUncertainty::RelativeBound { fraction } => {
            if !fraction.is_finite() || fraction < 0.0 {
                return Err(CalibrationError::InvalidValue);
            }
            if fraction == 0.0 || value == 0.0 {
                (value, value)
            } else {
                let error = directed(value.abs() * fraction, true)?;
                (
                    directed(value - error, false)?,
                    directed(value + error, true)?,
                )
            }
        }
        MeasurementUncertainty::Interval {
            lower,
            upper,
            unit_id,
        } => {
            if unit_id != semantics.unit_id() {
                return Err(CalibrationError::SemanticsMismatch);
            }
            (lower, upper)
        }
    };
    let interval = CalibrationInterval::new(lower, upper)?;
    if !interval.contains(value) {
        return Err(CalibrationError::InvalidRange);
    }
    Ok(Some(interval))
}

// One representable step outward includes rounding in each preceding basic
// arithmetic operation. Overflow of the outward endpoint fails closed.
fn directed(value: f64, upward: bool) -> Result<f64, CalibrationError> {
    if !value.is_finite() {
        return Err(CalibrationError::NumericalRange);
    }
    let adjusted = if value == 0.0 {
        if upward {
            f64::from_bits(1)
        } else {
            -f64::from_bits(1)
        }
    } else {
        let bits = value.to_bits();
        f64::from_bits(if (value > 0.0) == upward {
            bits + 1
        } else {
            bits - 1
        })
    };
    if !adjusted.is_finite() {
        return Err(CalibrationError::NumericalRange);
    }
    Ok(adjusted)
}
