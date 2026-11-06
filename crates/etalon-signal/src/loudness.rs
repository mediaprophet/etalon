//! Zero-heap loudness metrics (RMS, dBFS, crest factor, ITU-R BS.1770 LUFS),
//! auditory critical-band scales (Traunmüller Bark, Glasberg-Moore ERB),
//! and orthonormal DCT-II cepstral projection.

use crate::filter::{cos_f64, log10_f64, pow10_f64, sqrt_f64};

/// Linear root-mean-square (RMS) of a mono block: $\sqrt{\frac{1}{N} \sum x_i^2}$.
/// Returns `0.0` for an empty slice.
pub fn rms_f32(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for &x in samples {
        let xd = f64::from(x);
        acc += xd * xd;
    }
    sqrt_f64(acc / (samples.len() as f64)) as f32
}

/// RMS level in dBFS ($20 \log_{10}(\text{rms})$), returning `f32::NEG_INFINITY` on silence.
#[inline]
pub fn rms_dbfs(samples: &[f32]) -> f32 {
    let r = rms_f32(samples);
    if r <= 0.0 {
        f32::NEG_INFINITY
    } else {
        (20.0 * log10_f64(f64::from(r))) as f32
    }
}

/// Linear crest factor $\max |x_i| / \text{rms}(x)$ of a mono block.
/// Returns `0.0` for an empty or silent block.
pub fn crest_factor_f32(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut peak = 0.0f32;
    let mut sq = 0.0f64;
    for &x in samples {
        let a = x.abs();
        if a > peak {
            peak = a;
        }
        let xd = f64::from(x);
        sq += xd * xd;
    }
    let rms = sqrt_f64(sq / (samples.len() as f64)) as f32;
    if rms <= 0.0 {
        0.0
    } else {
        peak / rms
    }
}

/// Crest factor in dB ($20 \log_{10}(\text{peak} / \text{rms})$).
/// Returns `f32::NEG_INFINITY` for an empty or silent block.
#[inline]
pub fn crest_factor_db(samples: &[f32]) -> f32 {
    let c = crest_factor_f32(samples);
    if c <= 0.0 {
        f32::NEG_INFINITY
    } else {
        (20.0 * log10_f64(f64::from(c))) as f32
    }
}

/// Convert K-weighted mean-square energy $z$ to ITU-R BS.1770 / EBU R128 loudness in LUFS:
/// $L = -0.691 + 10 \log_{10}(z)$.
#[inline]
pub fn energy_to_lufs(mean_square: f64) -> f64 {
    if !mean_square.is_finite() || mean_square <= 0.0 {
        return f64::NEG_INFINITY;
    }
    -0.691 + 10.0 * log10_f64(mean_square)
}

/// Convert frequency in Hz to the Traunmüller (1990) Bark critical-band rate:
/// $z = \frac{26.81 f}{1960 + f} - 0.53$.
#[inline]
pub fn hz_to_bark(hz: f32) -> f32 {
    let f = hz.max(0.0);
    26.81 * f / (1960.0 + f) - 0.53
}

/// Convert Traunmüller Bark rate back to Hz:
/// $f = \frac{1960 (b + 0.53)}{26.81 - (b + 0.53)}$.
#[inline]
pub fn bark_to_hz(bark: f32) -> f32 {
    let z = bark + 0.53;
    let denom = 26.81 - z;
    if denom <= 0.0 {
        f32::INFINITY
    } else {
        1960.0 * z / denom
    }
}

/// Convert frequency in Hz to Glasberg & Moore (1990) ERB-rate number:
/// $E = 21.4 \log_{10}(1 + 0.00437 f)$.
#[inline]
pub fn hz_to_erb(hz: f32) -> f32 {
    let f = f64::from(hz.max(0.0));
    (21.4 * log10_f64(1.0 + 0.00437 * f)) as f32
}

/// Convert Glasberg & Moore ERB-rate number back to Hz:
/// $f = (10^{E / 21.4} - 1) / 0.00437$.
#[inline]
pub fn erb_to_hz(erb: f32) -> f32 {
    ((pow10_f64(f64::from(erb) / 21.4) - 1.0) / 0.00437) as f32
}

/// Orthonormal Discrete Cosine Transform Type-II (`norm="ortho"`) from `input` into `out`:
/// $$X[0] = \sqrt{\frac{1}{N}} \sum_{n=0}^{N-1} x[n], \quad
///   X[k] = \sqrt{\frac{2}{N}} \sum_{n=0}^{N-1} x[n] \cos\!\left(\frac{\pi}{N}(n + \tfrac{1}{2})k\right)$$
/// Returns `N` on success, or `0` if `input` is empty or `out.len() < input.len()`.
pub fn dct_ii_into(input: &[f32], out: &mut [f32]) -> usize {
    let n = input.len();
    if n == 0 || out.len() < n {
        return 0;
    }
    let inv_n = 1.0 / (n as f64);
    let scale0 = sqrt_f64(inv_n);
    let scale_k = sqrt_f64(2.0 * inv_n);
    let pi_over_n = core::f64::consts::PI * inv_n;

    for (k, slot) in out.iter_mut().enumerate().take(n) {
        let mut acc = 0.0f64;
        for (nn, &x) in input.iter().enumerate() {
            acc += f64::from(x) * cos_f64(pi_over_n * ((nn as f64) + 0.5) * (k as f64));
        }
        *slot = (if k == 0 { scale0 * acc } else { scale_k * acc }) as f32;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_crest_and_auditory_scales() {
        let s = [0.5f32, -0.5, 0.5, -0.5];
        assert!((rms_f32(&s) - 0.5).abs() < 1e-6);
        assert!((crest_factor_f32(&s) - 1.0).abs() < 1e-6);
        assert!((energy_to_lufs(1.0) - (-0.691)).abs() < 1e-6);

        assert!((bark_to_hz(hz_to_bark(1000.0)) - 1000.0).abs() < 0.5);
        assert!((erb_to_hz(hz_to_erb(1000.0)) - 1000.0).abs() < 0.5);

        let inp = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        assert_eq!(dct_ii_into(&inp, &mut out), 4);
        assert!((out[0] - 5.0).abs() < 1e-4);
    }
}
