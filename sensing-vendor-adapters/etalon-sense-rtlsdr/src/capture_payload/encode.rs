use super::{total_bytes, PayloadError, HEADER_BYTES, MAGIC, SCHEMA_VERSION};
use crate::{GainMode, ReadReport};

/// Retain the native receipt and raw bytes in a single custody payload. Size
/// pairing alone does not authenticate that the receipt describes these bytes.
/// The caller's authority must establish that binding before signing/storing.
/// Output must have the exact required size; every error clears all output.
pub fn encode_capture_payload(
    report: &ReadReport,
    raw_iq: &[u8],
    byte_budget: u64,
    output: &mut [u8],
) -> Result<usize, PayloadError> {
    output.fill(0);
    let bytes = total_bytes(report.pair_count())?;
    if bytes as u64 > byte_budget {
        return Err(PayloadError::BudgetExceeded);
    }
    if output.len() != bytes || raw_iq.len() != bytes - HEADER_BYTES {
        return Err(PayloadError::InvalidLength);
    }
    let identity = report.usb_identity();
    let strings = [
        identity.manufacturer(),
        identity.product(),
        identity.serial(),
    ];
    if report.target_id() == 0
        || report.generation() == 0
        || strings
            .iter()
            .any(|value| value.len() > 255 || value.contains(&0))
    {
        return Err(PayloadError::InvalidMetadata);
    }
    // All validation precedes writes; the fixed offsets are within HEADER_BYTES.
    output[..8].copy_from_slice(MAGIC);
    output[8..10].copy_from_slice(&SCHEMA_VERSION.to_le_bytes());
    output[10..12].copy_from_slice(&(HEADER_BYTES as u16).to_le_bytes());
    put64(output, 16, report.target_id());
    put32(output, 24, report.generation());
    put32(output, 28, report.device_index());
    put32(output, 32, report.pair_count() as u32);
    put64(output, 40, report.sequence());
    let requested = report.requested_config();
    put32(output, 48, requested.center_frequency_hz());
    put32(output, 52, requested.sample_rate_hz());
    put32(output, 56, requested.frequency_correction_ppm() as u32);
    if let GainMode::ManualTenthsDb(gain) = requested.tuner_gain() {
        put32(output, 60, 1);
        put32(output, 64, gain as u32);
    }
    put32(output, 68, u32::from(requested.digital_agc()));
    let applied = report.applied_config();
    put32(output, 72, applied.center_frequency_hz());
    put32(output, 76, applied.sample_rate_hz());
    put32(output, 80, applied.frequency_correction_ppm() as u32);
    if let Some(gain) = applied.tuner_gain_tenths_db() {
        put32(output, 84, 1);
        put32(output, 88, gain as u32);
    }
    put64(output, 96, raw_iq.len() as u64);
    for (index, value) in strings.iter().enumerate() {
        let length_offset = 104 + index * 2;
        output[length_offset..length_offset + 2]
            .copy_from_slice(&(value.len() as u16).to_le_bytes());
        let offset = 128 + index * 256;
        output[offset..offset + value.len()].copy_from_slice(value);
    }
    output[HEADER_BYTES..].copy_from_slice(raw_iq);
    Ok(bytes)
}

fn put32(out: &mut [u8], offset: usize, value: u32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put64(out: &mut [u8], offset: usize, value: u64) {
    out[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
