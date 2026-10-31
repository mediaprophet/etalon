//! Explicit scalar sample decoding; channel and physical meaning stay external.
use crate::SignalError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteOrder { LittleEndian, BigEndian }

/// Decode signed 16-bit sample scalars to f32, applying an explicit scale.
/// Interleaved I/Q remains interleaved; this function does not pair channels,
/// infer units, subtract an offset, or establish calibration. The complete
/// input and resulting range are checked before the output is modified.
pub fn decode_i16_into(input: &[u8], out: &mut [f32], order: ByteOrder, scale: f32) -> Result<usize, SignalError> {
    if input.len() % 2 != 0 || out.len() != input.len() / 2 { return Err(SignalError::LengthMismatch); }
    if !scale.is_finite() { return Err(SignalError::InvalidParameter); }
    for pair in input.chunks_exact(2) {
        let result = f64::from(read_i16(pair, order)) * f64::from(scale);
        if !result.is_finite() || !(result as f32).is_finite() { return Err(SignalError::NumericRange); }
    }
    for (pair, result) in input.chunks_exact(2).zip(out.iter_mut()) {
        *result = (f64::from(read_i16(pair, order)) * f64::from(scale)) as f32;
    }
    Ok(out.len())
}

/// Decode unsigned 8-bit scalar samples with an explicit digital zero code
/// and scale. I/Q ordering is retained as supplied by the caller; neither
/// offset calibration nor any physical unit is inferred. All results are
/// checked before output is modified.
pub fn decode_u8_centered_into(input: &[u8], out: &mut [f32], zero_code: u8, scale: f32) -> Result<usize, SignalError> {
    if out.len() != input.len() { return Err(SignalError::LengthMismatch); }
    if !scale.is_finite() { return Err(SignalError::InvalidParameter); }
    for &sample in input {
        let centered = i16::from(sample) - i16::from(zero_code);
        if !((f64::from(centered) * f64::from(scale)) as f32).is_finite() {
            return Err(SignalError::NumericRange);
        }
    }
    for (&sample, value) in input.iter().zip(out.iter_mut()) {
        let centered = i16::from(sample) - i16::from(zero_code);
        *value = (f64::from(centered) * f64::from(scale)) as f32;
    }
    Ok(input.len())
}

fn read_i16(pair: &[u8], order: ByteOrder) -> i16 {
    let bytes = [pair[0], pair[1]];
    match order { ByteOrder::LittleEndian => i16::from_le_bytes(bytes), ByteOrder::BigEndian => i16::from_be_bytes(bytes) }
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
