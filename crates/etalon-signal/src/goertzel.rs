//! Single-bin Goertzel power for a caller-supplied recurrence coefficient.
//!
//! Supplying `2*cos(omega)` avoids a libm dependency. Frequency-to-coefficient
//! conversion and sample-rate calibration remain the caller's responsibility.

use crate::SignalError;

/// Compute unnormalized power for one Goertzel bin.
///
/// `coefficient` must be finite and in `[-2, 2]`. The result is the classic
/// recurrence power `s1² + s2² - coefficient*s1*s2`; it is not a calibrated
/// amplitude or a hardware-qualified frequency estimate.
pub fn power_at_coefficient(samples: &[f32], coefficient: f64) -> Result<f64, SignalError> {
    if samples.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if !coefficient.is_finite() || !(-2.0..=2.0).contains(&coefficient) {
        return Err(SignalError::InvalidParameter);
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }

    let mut previous = 0.0f64;
    let mut previous_2 = 0.0f64;
    for &sample in samples {
        let current = f64::from(sample) + coefficient * previous - previous_2;
        if !current.is_finite() {
            return Err(SignalError::NumericRange);
        }
        previous_2 = previous;
        previous = current;
    }

    let power = previous * previous + previous_2 * previous_2 - coefficient * previous * previous_2;
    if !power.is_finite() {
        return Err(SignalError::NumericRange);
    }
    if power < 0.0 {
        // Roundoff may produce a tiny negative residue at a mathematical zero.
        // Only clamp values within a scale-aware floating point tolerance.
        let scale = previous * previous + previous_2 * previous_2;
        let tolerance = f64::EPSILON * scale * 8.0;
        if -power <= tolerance {
            return Ok(0.0);
        }
        return Err(SignalError::NumericRange);
    }
    Ok(power)
}

#[cfg(test)]
#[path = "goertzel_tests.rs"]
mod tests;
