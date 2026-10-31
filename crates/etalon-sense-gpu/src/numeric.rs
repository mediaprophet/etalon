use super::{GpuError, BYTES_PER_INPUT_PAIR, BYTES_PER_OUTPUT_PAIR, PARAM_BYTES};
use etalon_signal::decode::ByteOrder;

pub(super) fn checked_bytes(count: usize, per_item: usize) -> Result<usize, GpuError> {
    count.checked_mul(per_item).ok_or(GpuError::CapacityExceeded)
}

pub(super) fn persistent_buffer_bytes(max_pairs: usize) -> Result<u64, GpuError> {
    let sample_buffers = checked_bytes(max_pairs, 3 * BYTES_PER_OUTPUT_PAIR)?;
    let sample_buffers = u64::try_from(sample_buffers).map_err(|_| GpuError::CapacityExceeded)?;
    sample_buffers.checked_add(PARAM_BYTES).ok_or(GpuError::CapacityExceeded)
}

pub(super) fn validate_request(
    bytes: &[u8],
    scale: f32,
    workspace_len: usize,
    output_len: usize,
    max_pairs: usize,
) -> Result<usize, GpuError> {
    if !scale.is_finite() { return Err(GpuError::InvalidScale); }
    if bytes.is_empty() { return Err(GpuError::EmptyInput); }
    if bytes.len() % BYTES_PER_INPUT_PAIR != 0 { return Err(GpuError::LengthMismatch); }
    let count = bytes.len() / BYTES_PER_INPUT_PAIR;
    if count > max_pairs { return Err(GpuError::CapacityExceeded); }
    if workspace_len != count * 2 || output_len != count { return Err(GpuError::LengthMismatch); }
    Ok(count)
}

pub(super) fn validate_power_values(values: &[f32], upper_bound: f64) -> Result<(), GpuError> {
    if values
        .iter()
        .any(|power| !power.is_finite() || *power < 0.0 || f64::from(*power) > upper_bound)
    {
        return Err(GpuError::NumericRange);
    }
    Ok(())
}

pub(super) fn stage_power_values(
    values: &[f32],
    output: &mut [f64],
    upper_bound: f64,
) -> Result<usize, GpuError> {
    if output.len() != values.len() {
        return Err(GpuError::LengthMismatch);
    }
    validate_power_values(values, upper_bound)?;
    for (target, &value) in output.iter_mut().zip(values) {
        *target = f64::from(value);
    }
    Ok(values.len())
}

pub(super) fn make_params(scale: f32, order: ByteOrder, count: usize) -> [u8; PARAM_BYTES as usize] {
    let mut bytes = [0u8; PARAM_BYTES as usize];
    bytes[0..4].copy_from_slice(&scale.to_le_bytes());
    let big_endian = u32::from(matches!(order, ByteOrder::BigEndian));
    bytes[4..8].copy_from_slice(&big_endian.to_le_bytes());
    bytes[8..12].copy_from_slice(&(count as u32).to_le_bytes());
    bytes
}
