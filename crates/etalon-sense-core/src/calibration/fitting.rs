use core::num::NonZeroU64;
use crate::{observation::QuantityUnitIds, quality::SourceDescriptor};
use super::{CalibrationError, CalibrationInterval, CalibrationObservation};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineCoefficients { gain: f64, offset: f64 }

impl AffineCoefficients {
    pub fn new(gain: f64, offset: f64) -> Result<Self, CalibrationError> {
        if !gain.is_finite() || !offset.is_finite() { return Err(CalibrationError::InvalidValue); }
        Ok(Self { gain, offset })
    }
    pub const fn gain(self) -> f64 { self.gain }
    pub const fn offset(self) -> f64 { self.offset }
    pub fn map(self, value: f64) -> Result<f64, CalibrationError> {
        let product = self.gain * value;
        let mapped = product + self.offset;
        if !value.is_finite() || !product.is_finite() || !mapped.is_finite()
            || (self.gain != 0.0 && value != 0.0 && product == 0.0)
        { return Err(CalibrationError::NumericalRange); }
        Ok(mapped)
    }
}

/// An arithmetic fit, not a qualified calibration or uncertainty estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoPointAffineFit {
    pub coefficients: AffineCoefficients,
    pub input_range: CalibrationInterval,
    pub input: QuantityUnitIds,
    pub output: QuantityUnitIds,
    pub source: SourceDescriptor,
    pub environment_id: NonZeroU64,
    pub observation_ids: [NonZeroU64; 2],
}

/// Caller chooses a strictly positive minimum input separation in input units.
/// No residual/error bound is inferred from fitting exactly two points.
pub fn fit_affine_two_point(
    first: CalibrationObservation<'_>, second: CalibrationObservation<'_>, minimum_input_span: f64,
) -> Result<TwoPointAffineFit, CalibrationError> {
    if !minimum_input_span.is_finite() || minimum_input_span <= 0.0 {
        return Err(CalibrationError::InvalidRange);
    }
    if first.id() == second.id() { return Err(CalibrationError::DuplicateObservation); }
    let environment_id = first.environment_id().ok_or(CalibrationError::EnvironmentMismatch)?;
    if second.environment_id() != Some(environment_id) { return Err(CalibrationError::EnvironmentMismatch); }
    let source = first.measured().envelope().source();
    if second.measured().envelope().source() != source { return Err(CalibrationError::SourceMismatch); }
    let first_ref = first.reference().ok_or(CalibrationError::MissingReference)?;
    let second_ref = second.reference().ok_or(CalibrationError::MissingReference)?;
    let input = first.measured().envelope().quantity_unit();
    let output = first_ref.envelope().quantity_unit();
    require_semantics(input, output)?;
    if second.measured().envelope().quantity_unit() != input || second_ref.envelope().quantity_unit() != output {
        return Err(CalibrationError::SemanticsMismatch);
    }
    let x1 = first.measured().value().ok_or(CalibrationError::MissingValue)?;
    let x2 = second.measured().value().ok_or(CalibrationError::MissingValue)?;
    let y1 = first_ref.value().ok_or(CalibrationError::MissingValue)?;
    let y2 = second_ref.value().ok_or(CalibrationError::MissingValue)?;
    let dx = x2 - x1;
    let dy = y2 - y1;
    if !dx.is_finite() || !dy.is_finite() { return Err(CalibrationError::NumericalRange); }
    if dx.abs() < minimum_input_span { return Err(CalibrationError::DegenerateFit); }
    let gain = dy / dx;
    let product = gain * x1;
    let offset = y1 - product;
    if !gain.is_finite() || !offset.is_finite() || (dy != 0.0 && gain == 0.0)
        || (gain != 0.0 && x1 != 0.0 && product == 0.0)
    {
        return Err(CalibrationError::NumericalRange);
    }
    Ok(TwoPointAffineFit {
        coefficients: AffineCoefficients::new(gain, offset)?, input_range: CalibrationInterval::new(x1.min(x2), x1.max(x2))?,
        input, output, source, environment_id, observation_ids: [first.id(), second.id()],
    })
}

pub(super) fn require_semantics(input: QuantityUnitIds, output: QuantityUnitIds) -> Result<(), CalibrationError> {
    if input.quantity_id().is_none() || input.unit_id().is_none() || output.quantity_id().is_none() || output.unit_id().is_none() {
        return Err(CalibrationError::UnknownSemantics);
    }
    if input.quantity_id() != output.quantity_id() { return Err(CalibrationError::SemanticsMismatch); }
    Ok(())
}
