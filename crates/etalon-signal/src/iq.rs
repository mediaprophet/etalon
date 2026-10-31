//! Complex sample power without implicit calibration or unit conversion.
use crate::SignalError;

/// Write I² + Q² for interleaved I/Q pairs. The output is uncalibrated
/// squared input units; the caller determines calibration and physical units.
/// Input and output are fully validated before output is modified.
pub fn power_into(interleaved_iq: &[f32], output: &mut [f64]) -> Result<usize, SignalError> {
    if interleaved_iq.len() % 2 != 0 || output.len() < interleaved_iq.len() / 2 {
        return Err(SignalError::LengthMismatch);
    }
    if interleaved_iq.iter().any(|sample| !sample.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    let count = interleaved_iq.len() / 2;
    for (pair, result) in interleaved_iq.chunks_exact(2).zip(output.iter_mut()) {
        let i = f64::from(pair[0]);
        let q = f64::from(pair[1]);
        *result = i * i + q * q;
    }
    Ok(count)
}

#[cfg(test)]
#[path = "iq_tests.rs"]
mod tests;
