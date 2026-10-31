//! Bounded sensor-calibration observations, affine fitting and application.
//!
//! Guided procedures, physical references, qualification and authorization are
//! supplied by the application. Arithmetic and metadata checks do not establish
//! traceability, environmental coverage, accuracy or source authenticity.

mod application;
mod fitting;
mod observations;
mod profile;

pub use application::*;
pub use fitting::*;
pub use observations::*;
pub use profile::*;

pub const MAX_CALIBRATION_OBSERVATIONS: usize = 64;
pub const MAX_CALIBRATION_CONDITIONS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationError {
    CapacityExceeded,
    OutputTooSmall,
    DuplicateObservation,
    DuplicateCondition,
    InvalidValue,
    InvalidRange,
    MissingReference,
    MissingValue,
    UnknownSemantics,
    SourceMismatch,
    EnvironmentMismatch,
    SemanticsMismatch,
    MissingCondition,
    ConditionMismatch,
    OutsideRange,
    UnknownTime,
    ClockMismatch,
    OutsideValidity,
    ClockUnavailable,
    DegenerateFit,
    NumericalRange,
}

#[cfg(test)]
mod tests;
