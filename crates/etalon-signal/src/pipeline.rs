//! Small reference signal pipelines over caller-owned memory.
use crate::{decode::{decode_i16_into, ByteOrder}, iq::power_into, SignalError};

/// Decode interleaved signed i16 I/Q samples and compute uncalibrated power.
/// `scratch` holds the decoded scalars and `output` holds one power per pair.
/// Unit interpretation, source authentication and calibrated scaling belong
/// to the caller. All dimensions and numeric ranges are checked before either
/// output buffer is modified.
pub fn i16_iq_power_into(bytes: &[u8], order: ByteOrder, scale: f32, scratch: &mut [f32], output: &mut [f64]) -> Result<usize, SignalError> {
    if bytes.len() % 4 != 0 || scratch.len() != bytes.len() / 2 || output.len() != bytes.len() / 4 {
        return Err(SignalError::LengthMismatch);
    }
    if !scale.is_finite() { return Err(SignalError::InvalidParameter); }
    // A finite scale may still overflow f32 after multiplication by a sample.
    for pair in bytes.chunks_exact(2) {
        let value = match order {
            ByteOrder::LittleEndian => i16::from_le_bytes([pair[0], pair[1]]),
            ByteOrder::BigEndian => i16::from_be_bytes([pair[0], pair[1]]),
        };
        if !((f64::from(value) * f64::from(scale)) as f32).is_finite() {
            return Err(SignalError::NumericRange);
        }
    }
    decode_i16_into(bytes, scratch, order, scale)?;
    power_into(scratch, output)
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
