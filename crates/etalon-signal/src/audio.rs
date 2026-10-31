//! Caller-selected PCM audio features. Digital values are not sound pressure
//! levels without a separately qualified sensor and calibration.
use crate::{wave::Pcm16Wave, SignalError};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelPower {
    pub channel: u16,
    pub sample_rate_hz: u32,
    pub sample_count: usize,
    /// Mean square of signed PCM codes normalized by 32768.
    pub mean_square: f64,
}

/// Compute normalized mean-square sample power for one channel in a PCM16
/// recording. Channel numbering is zero-based. This is an electrical/digital
/// feature; it does not infer acoustic pressure, species or speech content.
pub fn pcm16_channel_power(wave: Pcm16Wave<'_>, channel: u16) -> Result<ChannelPower, SignalError> {
    if channel >= wave.channels() { return Err(SignalError::InvalidParameter); }
    if wave.frames() == 0 { return Err(SignalError::EmptyInput); }
    let channels = usize::from(wave.channels());
    let stride = channels * 2; // PCM parser checked u16 block alignment.
    let offset = usize::from(channel) * 2;
    let mut sum = 0.0f64;
    for frame in wave.samples().chunks_exact(stride) {
        let sample = i16::from_le_bytes([frame[offset], frame[offset + 1]]);
        let normalized = f64::from(sample) / 32768.0;
        sum += normalized * normalized;
    }
    let mean_square = sum / wave.frames() as f64;
    if !mean_square.is_finite() { return Err(SignalError::NumericRange); }
    Ok(ChannelPower { channel, sample_rate_hz: wave.sample_rate_hz(), sample_count: wave.frames(), mean_square })
}

#[cfg(test)]
#[path = "audio_tests.rs"]
mod tests;
