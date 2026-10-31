use crate::{Config, GainMode};
use super::{total_bytes, HistoricalAppliedConfig, PayloadError, UsbStringDeclarations,
    UntrustedCapturePayload, HEADER_BYTES, MAGIC, SCHEMA_VERSION};

/// Canonical structural decode with an explicit total-byte budget. Borrowed
/// output is untrusted historical data and must never be used as a live grant.
pub fn decode_capture_payload(bytes: &[u8], byte_budget: u64)
    -> Result<UntrustedCapturePayload<'_>, PayloadError> {
    if bytes.len() as u64 > byte_budget { return Err(PayloadError::BudgetExceeded); }
    if bytes.len() < HEADER_BYTES { return Err(PayloadError::InvalidLength); }
    if &bytes[..8] != MAGIC || u16_at(bytes, 8) != SCHEMA_VERSION
        || usize::from(u16_at(bytes, 10)) != HEADER_BYTES {
        return Err(PayloadError::NonCanonical);
    }
    for range in [12..16, 36..40, 92..96, 110..128, 896..HEADER_BYTES] {
        if bytes[range].iter().any(|byte| *byte != 0) { return Err(PayloadError::NonCanonical); }
    }
    let target_id = u64_at(bytes, 16);
    let generation = u32_at(bytes, 24);
    let pair_count = u32_at(bytes, 32);
    let total = total_bytes(pair_count as usize)?;
    if total != bytes.len() || u64_at(bytes, 96) != (total - HEADER_BYTES) as u64 {
        return Err(PayloadError::InvalidLength);
    }
    if target_id == 0 || generation == 0 { return Err(PayloadError::InvalidMetadata); }
    let gain = match u32_at(bytes, 60) {
        0 if u32_at(bytes, 64) == 0 => GainMode::Automatic,
        1 => GainMode::ManualTenthsDb(u32_at(bytes, 64) as i32),
        _ => return Err(PayloadError::NonCanonical),
    };
    let agc = match u32_at(bytes, 68) {
        0 => false, 1 => true, _ => return Err(PayloadError::NonCanonical),
    };
    let requested = Config::new(u32_at(bytes, 48), u32_at(bytes, 52),
        u32_at(bytes, 56) as i32, gain, agc).map_err(|_| PayloadError::InvalidMetadata)?;
    let applied_gain = match u32_at(bytes, 84) {
        0 if u32_at(bytes, 88) == 0 => None,
        1 => Some(u32_at(bytes, 88) as i32),
        _ => return Err(PayloadError::NonCanonical),
    };
    let applied = HistoricalAppliedConfig { center_frequency_hz: u32_at(bytes, 72),
        sample_rate_hz: u32_at(bytes, 76), frequency_correction_ppm: u32_at(bytes, 80) as i32,
        tuner_gain_tenths_db: applied_gain };
    if applied.center_frequency_hz != requested.center_frequency_hz()
        || applied.frequency_correction_ppm != requested.frequency_correction_ppm()
        || applied.sample_rate_hz == 0
        || match gain { GainMode::Automatic => applied_gain.is_some(),
            GainMode::ManualTenthsDb(gain) => applied_gain != Some(gain) } {
        return Err(PayloadError::InvalidMetadata);
    }
    let manufacturer = usb_string(bytes, 0)?;
    let product = usb_string(bytes, 1)?;
    let serial = usb_string(bytes, 2)?;
    Ok(UntrustedCapturePayload { target_id, generation, device_index: u32_at(bytes, 28),
        pair_count, host_read_sequence: u64_at(bytes, 40), requested, applied,
        usb_strings: UsbStringDeclarations { manufacturer, product, serial },
        raw_iq: &bytes[HEADER_BYTES..] })
}

fn usb_string(bytes: &[u8], index: usize) -> Result<&[u8], PayloadError> {
    let length = usize::from(u16_at(bytes, 104 + index * 2));
    if length > 255 { return Err(PayloadError::NonCanonical); }
    let field = &bytes[128 + index * 256..128 + (index + 1) * 256];
    if field[..length].contains(&0) || field[length..].iter().any(|byte| *byte != 0) {
        return Err(PayloadError::NonCanonical);
    }
    Ok(&field[..length])
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}
fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3],
        bytes[offset + 4], bytes[offset + 5], bytes[offset + 6], bytes[offset + 7]])
}
