//! Migration evidence consistency checks. These APIs do not authenticate evidence,
//! observe hardware execution, or authorize a production cutover.
//! In schema v1, mode is a workload label. Controller identity, instrumentation
//! and intervention cadence are not verified; mode behavior cannot be qualified.
#![no_std]
#![forbid(unsafe_code)]

mod receipt;
mod validate;

pub use receipt::*;
pub use validate::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    pub absolute: f64,
    pub relative: f64,
}

impl Tolerance {
    pub fn is_valid(self) -> bool {
        self.absolute.is_finite()
            && self.relative.is_finite()
            && self.absolute >= 0.0
            && self.relative >= 0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumericalComparison {
    pub compared: usize,
    pub mismatches: usize,
    pub max_absolute_error: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparisonError {
    EmptyInput,
    LengthMismatch,
    InvalidTolerance,
    NaN { index: usize },
    Infinity { index: usize },
    NonFiniteBound { index: usize },
}

/// Compares against a caller-provided independent reference; it does not generate
/// the reference. Non-finite values and overflowing error bounds are rejected.
pub fn compare_f32(
    reference: &[f32],
    candidate: &[f32],
    tolerance: Tolerance,
) -> Result<NumericalComparison, ComparisonError> {
    if !tolerance.is_valid() {
        return Err(ComparisonError::InvalidTolerance);
    }
    if reference.len() != candidate.len() {
        return Err(ComparisonError::LengthMismatch);
    }
    if reference.is_empty() {
        return Err(ComparisonError::EmptyInput);
    }
    let mut result = NumericalComparison {
        compared: reference.len(),
        mismatches: 0,
        max_absolute_error: 0.0,
    };
    for (index, (&expected, &actual)) in reference.iter().zip(candidate).enumerate() {
        if expected.is_nan() || actual.is_nan() {
            return Err(ComparisonError::NaN { index });
        }
        if !expected.is_finite() || !actual.is_finite() {
            return Err(ComparisonError::Infinity { index });
        }
        let difference = (f64::from(expected) - f64::from(actual)).abs();
        let bound = tolerance.absolute + tolerance.relative * f64::from(expected).abs();
        if !bound.is_finite() {
            return Err(ComparisonError::NonFiniteBound { index });
        }
        result.max_absolute_error = result.max_absolute_error.max(difference);
        if difference > bound {
            result.mismatches += 1;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
