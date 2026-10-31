//! Caller-windowed samples with no generated coefficients or hidden normalization.

use crate::SignalError;

/// Multiply each sample by its caller-supplied finite coefficient.
///
/// Coefficients have no assumed shape or range: the caller chooses the window
/// (including rectangular, Hann, or another function) and owns its spectral
/// interpretation. Inputs and every result are validated before `out` changes.
/// Extra output capacity is left untouched.
pub fn apply_window_into(
    samples: &[f32],
    coefficients: &[f32],
    out: &mut [f32],
) -> Result<usize, SignalError> {
    if samples.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if coefficients.len() != samples.len() || out.len() < samples.len() {
        return Err(SignalError::LengthMismatch);
    }
    if samples.iter().any(|sample| !sample.is_finite())
        || coefficients.iter().any(|coefficient| !coefficient.is_finite())
    {
        return Err(SignalError::NonFiniteInput);
    }

    // Preflight in wider precision so a numeric-range error is transactional.
    for (&sample, &coefficient) in samples.iter().zip(coefficients) {
        let result = f64::from(sample) * f64::from(coefficient);
        if !result.is_finite() || !(result as f32).is_finite() {
            return Err(SignalError::NumericRange);
        }
    }
    for ((&sample, &coefficient), output) in
        samples.iter().zip(coefficients).zip(out.iter_mut())
    {
        *output = (f64::from(sample) * f64::from(coefficient)) as f32;
    }
    Ok(samples.len())
}

#[cfg(test)]
#[path = "window_tests.rs"]
mod tests;
