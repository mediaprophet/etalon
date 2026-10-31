//! Zero-copy view of a bounded little-endian PCM16 RIFF/WAVE recording.
//! This parser does not acquire a microphone or establish recording provenance.
use crate::SignalError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pcm16Wave<'a> {
    samples: &'a [u8],
    channels: u16,
    sample_rate_hz: u32,
    frames: usize,
}

impl<'a> Pcm16Wave<'a> {
    /// Parse one complete RIFF container, rejecting truncation, ambiguous format
    /// or data chunks, inconsistent PCM layout, and unaligned sample frames.
    /// Unknown chunks are skipped using their RIFF padding byte.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, SignalError> {
        if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err(SignalError::InvalidParameter);
        }
        let riff_size = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let end = riff_size.checked_add(8).ok_or(SignalError::NumericRange)?;
        if end != bytes.len() || end < 12 { return Err(SignalError::LengthMismatch); }
        let mut cursor = 12usize;
        let mut format: Option<(u16, u32, u16)> = None;
        let mut data: Option<&[u8]> = None;
        while cursor < end {
            let header_end = cursor.checked_add(8).ok_or(SignalError::NumericRange)?;
            if header_end > end { return Err(SignalError::LengthMismatch); }
            let size = u32::from_le_bytes([bytes[cursor + 4], bytes[cursor + 5], bytes[cursor + 6], bytes[cursor + 7]]) as usize;
            let chunk_end = header_end.checked_add(size).ok_or(SignalError::NumericRange)?;
            let next = chunk_end.checked_add(size & 1).ok_or(SignalError::NumericRange)?;
            if next > end { return Err(SignalError::LengthMismatch); }
            match &bytes[cursor..cursor + 4] {
                b"fmt " => {
                    if format.is_some() || size < 16 { return Err(SignalError::InvalidParameter); }
                    let fmt = &bytes[header_end..chunk_end];
                    let codec = u16::from_le_bytes([fmt[0], fmt[1]]);
                    let channels = u16::from_le_bytes([fmt[2], fmt[3]]);
                    let rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
                    let byte_rate = u32::from_le_bytes([fmt[8], fmt[9], fmt[10], fmt[11]]);
                    let align = u16::from_le_bytes([fmt[12], fmt[13]]);
                    let bits = u16::from_le_bytes([fmt[14], fmt[15]]);
                    // Accept the 16-byte PCM WAVEFORMAT payload, or the
                    // 18-byte WAVEFORMATEX form only when its cbSize is zero.
                    // Partial or uninterpreted extensions make the format
                    // declaration ambiguous for this PCM16-only view.
                    let extension_is_valid = match fmt.len() {
                        16 => true,
                        18 => u16::from_le_bytes([fmt[16], fmt[17]]) == 0,
                        _ => false,
                    };
                    if codec != 1 || channels == 0 || rate == 0 || bits != 16
                        || !extension_is_valid
                        || channels.checked_mul(2) != Some(align)
                        || rate.checked_mul(u32::from(align)) != Some(byte_rate) {
                        return Err(SignalError::InvalidParameter);
                    }
                    format = Some((channels, rate, align));
                }
                b"data" => {
                    if data.is_some() { return Err(SignalError::InvalidParameter); }
                    data = Some(&bytes[header_end..chunk_end]);
                }
                _ => {}
            }
            cursor = next;
        }
        let (channels, sample_rate_hz, block_align) = format.ok_or(SignalError::InvalidParameter)?;
        let samples = data.ok_or(SignalError::InvalidParameter)?;
        if samples.len() % usize::from(block_align) != 0 { return Err(SignalError::LengthMismatch); }
        Ok(Self { samples, channels, sample_rate_hz, frames: samples.len() / usize::from(block_align) })
    }

    pub const fn samples(self) -> &'a [u8] { self.samples }
    pub const fn channels(self) -> u16 { self.channels }
    pub const fn sample_rate_hz(self) -> u32 { self.sample_rate_hz }
    pub const fn frames(self) -> usize { self.frames }
}

#[cfg(test)]
#[path = "wave_tests.rs"]
mod tests;
