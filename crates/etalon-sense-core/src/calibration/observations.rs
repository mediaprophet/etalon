use super::{CalibrationError, MAX_CALIBRATION_CONDITIONS, MAX_CALIBRATION_OBSERVATIONS};
use crate::observation::ObservationEnvelope;
use core::num::NonZeroU64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationInterval {
    lower: f64,
    upper: f64,
}

impl CalibrationInterval {
    pub fn new(lower: f64, upper: f64) -> Result<Self, CalibrationError> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(CalibrationError::InvalidRange);
        }
        Ok(Self { lower, upper })
    }
    pub const fn lower(self) -> f64 {
        self.lower
    }
    pub const fn upper(self) -> f64 {
        self.upper
    }
    pub fn contains(self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }
    pub fn encloses(self, other: Self) -> bool {
        self.contains(other.lower) && self.contains(other.upper)
    }
}

/// A host-named environmental or setup condition. An interval represents the
/// caller's entire declared error range; absence is not a zero-valued condition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationCondition {
    pub id: NonZeroU64,
    pub unit_id: Option<NonZeroU64>,
    pub value: Option<CalibrationInterval>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationDatum {
    envelope: ObservationEnvelope,
    value: Option<f64>,
}

impl CalibrationDatum {
    pub fn new(
        envelope: ObservationEnvelope,
        value: Option<f64>,
    ) -> Result<Self, CalibrationError> {
        if value.is_some_and(|value| !value.is_finite()) {
            return Err(CalibrationError::InvalidValue);
        }
        Ok(Self { envelope, value })
    }
    pub const fn envelope(self) -> ObservationEnvelope {
        self.envelope
    }
    pub const fn value(self) -> Option<f64> {
        self.value
    }
}

/// Pair existing capture/observation envelopes without erasing their source,
/// clock, quality, uncertainty, evidence or host-policy declarations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationObservation<'a> {
    id: NonZeroU64,
    measured: CalibrationDatum,
    reference: Option<CalibrationDatum>,
    environment_id: Option<NonZeroU64>,
    conditions: &'a [CalibrationCondition],
}

impl<'a> CalibrationObservation<'a> {
    pub fn new(
        id: NonZeroU64,
        measured: CalibrationDatum,
        reference: Option<CalibrationDatum>,
        environment_id: Option<NonZeroU64>,
        conditions: &'a [CalibrationCondition],
    ) -> Result<Self, CalibrationError> {
        validate_conditions(conditions)?;
        Ok(Self {
            id,
            measured,
            reference,
            environment_id,
            conditions,
        })
    }
    pub const fn id(self) -> NonZeroU64 {
        self.id
    }
    pub const fn measured(self) -> CalibrationDatum {
        self.measured
    }
    pub const fn reference(self) -> Option<CalibrationDatum> {
        self.reference
    }
    pub const fn environment_id(self) -> Option<NonZeroU64> {
        self.environment_id
    }
    pub const fn conditions(self) -> &'a [CalibrationCondition] {
        self.conditions
    }
}

pub(super) fn validate_conditions(
    conditions: &[CalibrationCondition],
) -> Result<(), CalibrationError> {
    if conditions.len() > MAX_CALIBRATION_CONDITIONS {
        return Err(CalibrationError::CapacityExceeded);
    }
    for (index, condition) in conditions.iter().enumerate() {
        if conditions[..index]
            .iter()
            .any(|other| other.id == condition.id)
        {
            return Err(CalibrationError::DuplicateCondition);
        }
    }
    Ok(())
}

/// Record a bounded batch into caller storage; failure leaves output intact.
/// Acquisition itself uses the existing sensor adapters and host policy gates.
pub fn record_calibration_observations<'a>(
    observations: &[CalibrationObservation<'a>],
    output: &mut [Option<CalibrationObservation<'a>>],
) -> Result<usize, CalibrationError> {
    if observations.len() > MAX_CALIBRATION_OBSERVATIONS {
        return Err(CalibrationError::CapacityExceeded);
    }
    if output.len() < observations.len() {
        return Err(CalibrationError::OutputTooSmall);
    }
    for (index, observation) in observations.iter().enumerate() {
        if observations[..index]
            .iter()
            .any(|other| other.id == observation.id)
        {
            return Err(CalibrationError::DuplicateObservation);
        }
    }
    for (slot, observation) in output.iter_mut().zip(observations) {
        *slot = Some(*observation);
    }
    Ok(observations.len())
}
