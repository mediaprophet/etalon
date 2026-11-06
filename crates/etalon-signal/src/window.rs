//! Caller-windowed samples and canonical periodic (DFT-even) window generators.

use crate::{filter::sin_cos_f64, SignalError};

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
        || coefficients
            .iter()
            .any(|coefficient| !coefficient.is_finite())
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
    for ((&sample, &coefficient), output) in samples.iter().zip(coefficients).zip(out.iter_mut()) {
        *output = (f64::from(sample) * f64::from(coefficient)) as f32;
    }
    Ok(samples.len())
}

/// Apply a precomputed window in-place: `samples[k] *= window[k]`.
#[inline]
pub fn apply_window_in_place(samples: &mut [f32], window: &[f32]) -> Result<(), SignalError> {
    if samples.len() != window.len() {
        return Err(SignalError::LengthMismatch);
    }
    for (s, &w) in samples.iter_mut().zip(window.iter()) {
        *s *= w;
    }
    Ok(())
}

/// Generate a periodic (DFT-even) Hann window: $w[k] = 0.5 (1 - \cos(2\pi k / N))$.
pub fn hann_window(out: &mut [f32]) -> Result<(), SignalError> {
    let n_len = out.len();
    if n_len == 0 {
        return Err(SignalError::EmptyInput);
    }
    if n_len == 1 {
        out[0] = 1.0;
        return Ok(());
    }
    let n = n_len as f64;
    for (k, val) in out.iter_mut().enumerate() {
        let (_, c) = sin_cos_f64(core::f64::consts::TAU * (k as f64) / n);
        *val = (0.5 * (1.0 - c)) as f32;
    }
    Ok(())
}

/// Generate a periodic (DFT-even) Hamming window: $w[k] = 0.54 - 0.46 \cos(2\pi k / N)$.
pub fn hamming_window(out: &mut [f32]) -> Result<(), SignalError> {
    let n_len = out.len();
    if n_len == 0 {
        return Err(SignalError::EmptyInput);
    }
    if n_len == 1 {
        out[0] = 1.0;
        return Ok(());
    }
    let n = n_len as f64;
    for (k, val) in out.iter_mut().enumerate() {
        let (_, c) = sin_cos_f64(core::f64::consts::TAU * (k as f64) / n);
        *val = (0.54 - 0.46 * c) as f32;
    }
    Ok(())
}

/// Generate a periodic (DFT-even) 3-term Blackman window:
/// $w[k] = 0.42 - 0.5 \cos(2\pi k / N) + 0.08 \cos(4\pi k / N)$.
pub fn blackman_window(out: &mut [f32]) -> Result<(), SignalError> {
    let n_len = out.len();
    if n_len == 0 {
        return Err(SignalError::EmptyInput);
    }
    if n_len == 1 {
        out[0] = 1.0;
        return Ok(());
    }
    let n = n_len as f64;
    for (k, val) in out.iter_mut().enumerate() {
        let phase = core::f64::consts::TAU * (k as f64) / n;
        let (_, c1) = sin_cos_f64(phase);
        let (_, c2) = sin_cos_f64(2.0 * phase);
        *val = (0.42 - 0.5 * c1 + 0.08 * c2) as f32;
    }
    Ok(())
}

/// Generate a periodic (DFT-even) 4-term Blackman-Harris window (-92 dB sidelobe floor).
pub fn blackman_harris_window(out: &mut [f32]) -> Result<(), SignalError> {
    let n_len = out.len();
    if n_len == 0 {
        return Err(SignalError::EmptyInput);
    }
    if n_len == 1 {
        out[0] = 1.0;
        return Ok(());
    }
    const A0: f64 = 0.35875;
    const A1: f64 = 0.48829;
    const A2: f64 = 0.14128;
    const A3: f64 = 0.01168;
    let n = n_len as f64;
    for (k, val) in out.iter_mut().enumerate() {
        let phase = core::f64::consts::TAU * (k as f64) / n;
        let (_, c1) = sin_cos_f64(phase);
        let (_, c2) = sin_cos_f64(2.0 * phase);
        let (_, c3) = sin_cos_f64(3.0 * phase);
        *val = (A0 - A1 * c1 + A2 * c2 - A3 * c3) as f32;
    }
    Ok(())
}

#[cfg(test)]
#[path = "window_tests.rs"]
mod tests;
