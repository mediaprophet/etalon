use core::num::NonZeroU64;
use crate::{
    observation::QuantityUnitIds, processing::OperationId,
    quality::{CalibrationArtifactId, SourceDescriptor}, time::{ClockStamp, TimeInterval},
};
use super::{
    fitting::require_semantics, observations::validate_conditions,
    AffineCoefficients, CalibrationCondition, CalibrationDatum, CalibrationError,
    CalibrationInterval, CalibrationObservation, MAX_CALIBRATION_CONDITIONS, MAX_CALIBRATION_OBSERVATIONS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CalibrationValidity {
    pub clock_id: NonZeroU64,
    pub restart_epoch: u32,
    pub interval: TimeInterval,
}

impl CalibrationValidity {
    pub fn check(self, stamp: ClockStamp) -> Result<(), CalibrationError> {
        if stamp.clock_id() != self.clock_id.get() || stamp.restart_epoch() != self.restart_epoch {
            return Err(CalibrationError::ClockMismatch);
        }
        if stamp.interval().lower_ns() < self.interval.lower_ns() || stamp.interval().upper_ns() > self.interval.upper_ns() {
            return Err(CalibrationError::OutsideValidity);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationConditionRequirement {
    pub id: NonZeroU64,
    pub unit_id: NonZeroU64,
    pub allowed: CalibrationInterval,
}

/// Coefficients and applicability/error claims supplied by the application.
/// `model_error_bound` covers reference, fit, coefficient and unmodeled errors
/// in output units; None preserves lack of a qualified total error bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineCalibrationSpecification {
    pub artifact_id: CalibrationArtifactId,
    pub method: OperationId,
    pub source: SourceDescriptor,
    pub input: QuantityUnitIds,
    pub output: QuantityUnitIds,
    pub environment_id: NonZeroU64,
    pub coefficients: AffineCoefficients,
    pub input_range: CalibrationInterval,
    pub validity: CalibrationValidity,
    pub model_error_bound: Option<f64>,
}

pub struct AffineCalibrationProfile<'a, 'conditions> {
    specification: AffineCalibrationSpecification,
    requirements: &'a [CalibrationConditionRequirement],
    observations: &'a [CalibrationObservation<'conditions>],
}

impl<'a, 'conditions> AffineCalibrationProfile<'a, 'conditions> {
    pub fn new(
        specification: AffineCalibrationSpecification,
        requirements: &'a [CalibrationConditionRequirement],
        observations: &'a [CalibrationObservation<'conditions>],
    ) -> Result<Self, CalibrationError> {
        if requirements.len() > MAX_CALIBRATION_CONDITIONS || observations.len() > MAX_CALIBRATION_OBSERVATIONS {
            return Err(CalibrationError::CapacityExceeded);
        }
        if observations.len() < 2 { return Err(CalibrationError::MissingReference); }
        require_semantics(specification.input, specification.output)?;
        if specification.model_error_bound.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(CalibrationError::InvalidValue);
        }
        for (index, requirement) in requirements.iter().enumerate() {
            if requirements[..index].iter().any(|other| other.id == requirement.id) {
                return Err(CalibrationError::DuplicateCondition);
            }
        }
        let profile = Self { specification, requirements, observations };
        let mut lower = f64::INFINITY;
        let mut upper = f64::NEG_INFINITY;
        for (index, observation) in observations.iter().enumerate() {
            if observations[..index].iter().any(|other| other.id() == observation.id()) {
                return Err(CalibrationError::DuplicateObservation);
            }
            profile.check_context(observation.measured(), observation.environment_id(), observation.conditions())?;
            let reference = observation.reference().ok_or(CalibrationError::MissingReference)?;
            if reference.envelope().quantity_unit() != specification.output { return Err(CalibrationError::SemanticsMismatch); }
            reference.value().ok_or(CalibrationError::MissingValue)?;
            let value = observation.measured().value().ok_or(CalibrationError::MissingValue)?;
            lower = lower.min(value);
            upper = upper.max(value);
        }
        if lower >= upper { return Err(CalibrationError::DegenerateFit); }
        if !CalibrationInterval::new(lower, upper)?.encloses(specification.input_range) {
            return Err(CalibrationError::OutsideRange);
        }
        Ok(profile)
    }

    pub const fn specification(&self) -> AffineCalibrationSpecification { self.specification }
    pub const fn requirements(&self) -> &'a [CalibrationConditionRequirement] { self.requirements }
    pub const fn observations(&self) -> &'a [CalibrationObservation<'conditions>] { self.observations }

    pub(super) fn check_context(
        &self, datum: CalibrationDatum, environment: Option<NonZeroU64>, conditions: &[CalibrationCondition],
    ) -> Result<(), CalibrationError> {
        validate_conditions(conditions)?;
        if datum.envelope().source() != self.specification.source { return Err(CalibrationError::SourceMismatch); }
        if datum.envelope().quantity_unit() != self.specification.input { return Err(CalibrationError::SemanticsMismatch); }
        if environment != Some(self.specification.environment_id) { return Err(CalibrationError::EnvironmentMismatch); }
        for requirement in self.requirements {
            let actual = conditions.iter().find(|condition| condition.id == requirement.id)
                .ok_or(CalibrationError::MissingCondition)?;
            if actual.unit_id != Some(requirement.unit_id) { return Err(CalibrationError::ConditionMismatch); }
            let range = actual.value.ok_or(CalibrationError::MissingCondition)?;
            if !requirement.allowed.encloses(range) { return Err(CalibrationError::ConditionMismatch); }
        }
        Ok(())
    }
}
