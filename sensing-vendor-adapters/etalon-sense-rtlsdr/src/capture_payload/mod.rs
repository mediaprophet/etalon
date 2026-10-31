//! Opt-in canonical receipt plus raw I/Q storage, without authentication.

mod encode;
mod decode;
pub use encode::encode_capture_payload;
pub use decode::decode_capture_payload;

use crate::Config;

pub const HEADER_BYTES: usize = 1024;
pub const SCHEMA_VERSION: u16 = 1;
pub const MAGIC: &[u8; 8] = b"ETRTLSD1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PayloadError { InvalidLength, BudgetExceeded, InvalidMetadata, NonCanonical }

/// Native getter declarations, not independently measured tuning or gain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoricalAppliedConfig {
    pub center_frequency_hz: u32,
    pub sample_rate_hz: u32,
    pub frequency_correction_ppm: i32,
    pub tuner_gain_tenths_db: Option<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UsbStringDeclarations<'a> {
    pub manufacturer: &'a [u8],
    pub product: &'a [u8],
    pub serial: &'a [u8],
}

/// Historical unsigned data. Decoding creates no current grant, identity,
/// calibration, clock, continuity, provenance or custody-verification token.
pub struct UntrustedCapturePayload<'a> {
    pub target_id: u64,
    pub generation: u32,
    pub device_index: u32,
    pub pair_count: u32,
    pub host_read_sequence: u64,
    pub requested: Config,
    pub applied: HistoricalAppliedConfig,
    pub usb_strings: UsbStringDeclarations<'a>,
    pub raw_iq: &'a [u8],
}

fn total_bytes(pairs: usize) -> Result<usize, PayloadError> {
    if pairs == 0 || pairs > crate::MAX_PAIRS { return Err(PayloadError::InvalidLength); }
    pairs.checked_mul(2).and_then(|raw| raw.checked_add(HEADER_BYTES))
        .ok_or(PayloadError::InvalidLength)
}

#[cfg(test)]
mod tests;
