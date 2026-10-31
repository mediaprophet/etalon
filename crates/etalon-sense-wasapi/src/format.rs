use core::ptr;

use etalon_sense_core::{adapter::AdapterError, quantity::SampleEncoding, ContractError};
use windows::{
    core::GUID,
    Win32::Media::Audio::{WAVEFORMATEX, WAVEFORMATEXTENSIBLE},
};

const WAVE_FORMAT_PCM_TAG: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xfffe;
const PCM_SUBFORMAT: GUID = GUID::from_u128(0x00000001_0000_0010_8000_00aa00389b71);
const FLOAT_SUBFORMAT: GUID = GUID::from_u128(0x00000003_0000_0010_8000_00aa00389b71);

/// Supported negotiated WASAPI mix formats. The adapter copies these samples
/// unchanged; it does not calibrate microphone voltage or sound pressure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WasapiFormat {
    sample_rate_hz: u32,
    channels: u16,
    block_align: u16,
    bits_per_sample: u16,
    encoding: SampleEncoding,
}

impl WasapiFormat {
    pub const fn sample_rate_hz(self) -> u32 {
        self.sample_rate_hz
    }
    pub const fn channels(self) -> u16 {
        self.channels
    }
    pub const fn block_align(self) -> u16 {
        self.block_align
    }
    pub const fn bits_per_sample(self) -> u16 {
        self.bits_per_sample
    }
    pub const fn encoding(self) -> SampleEncoding {
        self.encoding
    }

    pub(super) unsafe fn parse(pointer: *const WAVEFORMATEX) -> Result<Self, AdapterError> {
        if pointer.is_null() {
            return Err(AdapterError::Unsupported);
        }
        let base = unsafe { ptr::read_unaligned(pointer) };
        let subformat = if base.wFormatTag == WAVE_FORMAT_EXTENSIBLE {
            if base.cbSize < 22 {
                return Err(AdapterError::Unsupported);
            }
            let extended = unsafe { ptr::read_unaligned(pointer.cast::<WAVEFORMATEXTENSIBLE>()) };
            let valid_bits = unsafe { extended.Samples.wValidBitsPerSample };
            if valid_bits != base.wBitsPerSample {
                return Err(AdapterError::Unsupported);
            }
            Some(extended.SubFormat)
        } else {
            None
        };

        let is_pcm = (base.wFormatTag == WAVE_FORMAT_PCM_TAG && subformat.is_none())
            || (base.wFormatTag == WAVE_FORMAT_EXTENSIBLE && subformat == Some(PCM_SUBFORMAT));
        let is_float = (base.wFormatTag == WAVE_FORMAT_IEEE_FLOAT && subformat.is_none())
            || (base.wFormatTag == WAVE_FORMAT_EXTENSIBLE && subformat == Some(FLOAT_SUBFORMAT));
        let encoding = if is_pcm && base.wBitsPerSample == 16 {
            SampleEncoding::I16
        } else if is_float && base.wBitsPerSample == 32 {
            SampleEncoding::F32
        } else {
            return Err(AdapterError::Unsupported);
        };
        let sample_bytes = u32::from(base.wBitsPerSample / 8);
        let expected_align = u32::from(base.nChannels)
            .checked_mul(sample_bytes)
            .ok_or(AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
        let expected_rate = base
            .nSamplesPerSec
            .checked_mul(expected_align)
            .ok_or(AdapterError::Buffer(ContractError::ArithmeticOverflow))?;
        if base.nChannels == 0
            || base.nSamplesPerSec == 0
            || sample_bytes == 0
            || u32::from(base.nBlockAlign) != expected_align
            || base.nAvgBytesPerSec != expected_rate
        {
            return Err(AdapterError::Unsupported);
        }
        Ok(Self {
            sample_rate_hz: base.nSamplesPerSec,
            channels: base.nChannels,
            block_align: base.nBlockAlign,
            bits_per_sample: base.wBitsPerSample,
            encoding,
        })
    }
}

#[cfg(test)]
#[path = "format_tests.rs"]
mod tests;
