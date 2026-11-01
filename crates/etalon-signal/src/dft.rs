//! Bounded reference complex transform with caller-supplied coefficients.
//!
//! This is a CPU reference operation, not an FFT, acceleration backend,
//! calibrated spectrum, or a guarantee that coefficients are DFT twiddles.
use crate::SignalError;

/// Upper bound on the quadratic work and coefficient matrix size.
pub const MAX_DFT_SAMPLES: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DftScale {
    Raw,
    DivideBySampleCount,
}

/// Transform interleaved complex samples using row-major interleaved complex
/// coefficients: `output[k] = sum(input[n] * coefficient[k,n])`. The caller
/// selects sign, coefficient generation, frequency labeling, and windowing.
///
/// All inputs and results are checked before modifying output. Coefficient
/// components must lie in [-1, 1], but unit magnitude is not established.
/// `out` must contain exactly two values per input complex sample.
pub fn transform_into(
    input: &[f32],
    coefficients: &[f32],
    out: &mut [f64],
    scale: DftScale,
) -> Result<usize, SignalError> {
    if input.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if !input.len().is_multiple_of(2) {
        return Err(SignalError::LengthMismatch);
    }
    let count = input.len() / 2;
    if count > MAX_DFT_SAMPLES {
        return Err(SignalError::InvalidParameter);
    }
    let expected = count
        .checked_mul(count)
        .and_then(|v| v.checked_mul(2))
        .ok_or(SignalError::NumericRange)?;
    if coefficients.len() != expected || out.len() != input.len() {
        return Err(SignalError::LengthMismatch);
    }
    if input.iter().any(|v| !v.is_finite()) || coefficients.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    if coefficients.iter().any(|v| *v < -1.0 || *v > 1.0) {
        return Err(SignalError::InvalidParameter);
    }
    for bin in 0..count {
        let (real, imag) = calculate_bin(input, coefficients, bin, count, scale);
        if !real.is_finite() || !imag.is_finite() {
            return Err(SignalError::NumericRange);
        }
    }
    for bin in 0..count {
        let (real, imag) = calculate_bin(input, coefficients, bin, count, scale);
        out[bin * 2] = real;
        out[bin * 2 + 1] = imag;
    }
    Ok(count)
}

fn calculate_bin(
    input: &[f32],
    coefficients: &[f32],
    bin: usize,
    count: usize,
    scale: DftScale,
) -> (f64, f64) {
    let mut real = 0.0;
    let mut imag = 0.0;
    for index in 0..count {
        let a = f64::from(input[index * 2]);
        let b = f64::from(input[index * 2 + 1]);
        let c = f64::from(coefficients[(bin * count + index) * 2]);
        let d = f64::from(coefficients[(bin * count + index) * 2 + 1]);
        real += a * c - b * d;
        imag += a * d + b * c;
    }
    if scale == DftScale::DivideBySampleCount {
        (real / count as f64, imag / count as f64)
    } else {
        (real, imag)
    }
}

#[cfg(test)]
#[path = "dft_tests.rs"]
mod tests;
