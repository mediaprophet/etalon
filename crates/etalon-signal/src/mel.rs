//! Row-major power-spectrum projection onto caller-supplied mel weights and HTK mel conversions.
use crate::{
    filter::{log10_f64, pow10_f64},
    SignalError,
};

/// Convert frequency in Hz to the HTK Mel scale:
/// $\text{mel} = 2595 \cdot \log_{10}(1 + f / 700)$.
#[inline]
pub fn hz_to_mel(hz: f32) -> f32 {
    (2595.0 * log10_f64(1.0 + f64::from(hz.max(0.0)) / 700.0)) as f32
}

/// Convert HTK Mel frequency back to Hz:
/// $f = 700 \cdot (10^{\text{mel} / 2595} - 1)$.
#[inline]
pub fn mel_to_hz(mel: f32) -> f32 {
    (700.0 * (pow10_f64(f64::from(mel) / 2595.0) - 1.0)) as f32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MelShape {
    pub n_frames: usize,
    pub n_bins: usize,
    pub n_mel: usize,
}
impl MelShape {
    pub fn spectrum_len(self) -> Result<usize, SignalError> {
        self.n_frames.checked_mul(self.n_bins).ok_or(SignalError::NumericRange)
    }
    pub fn filterbank_len(self) -> Result<usize, SignalError> {
        self.n_mel.checked_mul(self.n_bins).ok_or(SignalError::NumericRange)
    }
    pub fn output_len(self) -> Result<usize, SignalError> {
        self.n_frames.checked_mul(self.n_mel).ok_or(SignalError::NumericRange)
    }
}
/// Borrows a constructed filterbank; preparation and execution allocate nothing.
#[derive(Clone, Copy, Debug)]
pub struct PreparedMel<'a> {
    shape: MelShape,
    filterbank: &'a [f32],
}
impl<'a> PreparedMel<'a> {
    pub fn new(shape: MelShape, filterbank: &'a [f32]) -> Result<Self, SignalError> {
        shape.spectrum_len()?;
        shape.output_len()?;
        if filterbank.len() != shape.filterbank_len()? {
            return Err(SignalError::LengthMismatch);
        }
        Ok(Self { shape, filterbank })
    }
    pub fn shape(self) -> MelShape { self.shape }
    pub fn filterbank(self) -> &'a [f32] { self.filterbank }
    /// Writes the output prefix in increasing-bin f32 accumulation order.
    /// IEEE non-finite values propagate as in the original scalar oracle.
    pub fn apply_into(self, spectrum: &[f32], out: &mut [f32]) -> Result<usize, SignalError> {
        let shape = self.shape;
        let len = shape.output_len()?;
        if spectrum.len() != shape.spectrum_len()? || out.len() < len {
            return Err(SignalError::LengthMismatch);
        }
        if len == 0 { return Ok(0); }
        for frame in 0..shape.n_frames {
            let spec_base = frame * shape.n_bins;
            for m in 0..shape.n_mel {
                let fb_base = m * shape.n_bins;
                let mut acc = 0.0f32;
                for b in 0..shape.n_bins {
                    acc += spectrum[spec_base + b] * self.filterbank[fb_base + b];
                }
                out[frame * shape.n_mel + m] = acc;
            }
        }
        Ok(len)
    }
}
pub fn mel_apply_into(spectrum: &[f32], filterbank: &[f32], shape: MelShape,
                      out: &mut [f32]) -> Result<usize, SignalError> {
    PreparedMel::new(shape, filterbank)?.apply_into(spectrum, out)
}
#[cfg(test)]
#[path = "mel_tests.rs"]
mod tests;
