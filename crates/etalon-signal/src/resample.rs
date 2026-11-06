//! Zero-heap sample-rate conversion (linear, Lanczos/Kaiser windowed-sinc, polyphase)
//! and PCM format/channel conversion primitives.

use crate::filter::{floor_f64, sin_cos_f64, sqrt_f64};

#[inline]
fn ceil_f64(x: f64) -> f64 {
    -floor_f64(-x)
}

#[inline]
fn sinc_f64(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = core::f64::consts::PI * x;
        let (s, _) = sin_cos_f64(px);
        s / px
    }
}

/// Modified zeroth-order Bessel function of the first kind $I_0(x)$ via 24-term power series:
/// $$I_0(x) = \sum_{k=0}^{24} \left(\frac{(x/2)^k}{k!}\right)^2$$
#[inline]
pub fn kaiser_bessel_i0(x: f64) -> f64 {
    let half_x = 0.5 * x;
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    for k in 1..=24 {
        term *= half_x / (k as f64);
        sum += term * term;
    }
    sum
}

/// Anti-aliasing low-pass cutoff frequency in Hz (`0.5 * min(src_rate, dst_rate)`).
#[inline]
pub fn anti_alias_cutoff_hz(src_rate: f64, dst_rate: f64) -> f64 {
    if !src_rate.is_finite() || !dst_rate.is_finite() || src_rate <= 0.0 || dst_rate <= 0.0 {
        return 0.0;
    }
    0.5 * src_rate.min(dst_rate)
}

/// Anti-alias low-pass cutoff in cycles per source sample (`[0.0, 0.5]`).
#[inline]
pub fn antialias_cutoff_cycles(src_rate: u32, dst_rate: u32) -> f32 {
    if src_rate == 0 || dst_rate == 0 {
        return 0.0;
    }
    let ratio = dst_rate as f32 / src_rate as f32;
    0.5 * ratio.min(1.0)
}

/// Linear-interpolation mono resampler into caller-supplied `out`.
/// Returns the number of output frames written.
pub fn resample_linear_mono_into(
    input: &[f32],
    src_rate_hz: u32,
    dst_rate_hz: u32,
    out: &mut [f32],
) -> usize {
    if src_rate_hz == 0 || dst_rate_hz == 0 || input.is_empty() {
        return 0;
    }
    if src_rate_hz == dst_rate_hz {
        let n = input.len().min(out.len());
        out[..n].copy_from_slice(&input[..n]);
        return n;
    }
    let out_n = (((input.len() as u64 * dst_rate_hz as u64) / src_rate_hz as u64) as usize)
        .min(out.len());
    if out_n == 0 {
        return 0;
    }
    let ratio = src_rate_hz as f64 / dst_rate_hz as f64;
    for (i, slot) in out.iter_mut().enumerate().take(out_n) {
        let pos = i as f64 * ratio;
        let i0 = floor_f64(pos) as usize;
        let frac = (pos - i0 as f64) as f32;
        let s0 = input.get(i0).copied().unwrap_or(0.0);
        let s1 = input.get(i0 + 1).copied().unwrap_or(s0);
        *slot = s0 * (1.0 - frac) + s1 * frac;
    }
    out_n
}

/// Convert `f32` samples in `[-1.0, 1.0]` to little-endian 16-bit PCM bytes in `out_bytes`.
/// Returns the number of samples encoded.
pub fn pcm_f32_to_i16_le_into(samples: &[f32], out_bytes: &mut [u8]) -> usize {
    let n = samples.len().min(out_bytes.len() / 2);
    for (i, &s) in samples.iter().enumerate().take(n) {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        let b = v.to_le_bytes();
        out_bytes[i * 2] = b[0];
        out_bytes[i * 2 + 1] = b[1];
    }
    n
}

/// Downmix interleaved multi-channel `f32` PCM into mono `out`.
/// Returns the number of mono frames written.
pub fn pcm_to_mono_f32_into(interleaved: &[f32], channels: usize, out: &mut [f32]) -> usize {
    if channels == 0 {
        return 0;
    }
    let frames = (interleaved.len() / channels).min(out.len());
    let inv_ch = 1.0 / (channels as f32);
    for (i, slot) in out.iter_mut().enumerate().take(frames) {
        let base = i * channels;
        let mut acc = 0.0f32;
        for c in 0..channels {
            acc += interleaved[base + c];
        }
        *slot = acc * inv_ch;
    }
    frames
}

/// Evaluate a DC-normalized band-limited windowed-sinc kernel at fractional source index `pos`.
///
/// - `cutoff_ratio`: `min(1.0, dst_rate / src_rate)` (`2.0 * cutoff_cycles`).
/// - `half_width`: kernel half-width in scaled taps ($\ge 1$).
/// - `kaiser_beta`: if `<= 0.0`, uses a Lanczos window $\text{sinc}(d / R)$; if `> 0.0`,
///   uses a Kaiser-Bessel window $I_0(\beta \sqrt{1 - (d/R)^2}) / I_0(\beta)$.
#[inline]
pub fn windowed_sinc_sample(
    input: &[f32],
    pos: f64,
    cutoff_ratio: f64,
    half_width: usize,
    kaiser_beta: f64,
) -> f32 {
    if input.is_empty() || half_width == 0 || cutoff_ratio <= 0.0 {
        return 0.0;
    }
    let filt_scale = cutoff_ratio.min(1.0);
    let radius = (half_width as f64) / filt_scale;
    let j_min = ceil_f64(pos - radius) as i64;
    let j_max = floor_f64(pos + radius) as i64;
    let len = input.len() as i64;
    let inv_i0_beta = if kaiser_beta > 0.0 {
        1.0 / kaiser_bessel_i0(kaiser_beta)
    } else {
        1.0
    };

    let mut acc = 0.0f64;
    let mut wsum = 0.0f64;
    let mut j = j_min;
    while j <= j_max {
        if j >= 0 && j < len {
            let d = pos - (j as f64);
            let u = d / radius;
            let win = if kaiser_beta > 0.0 {
                let arg = (1.0 - u * u).max(0.0);
                kaiser_bessel_i0(kaiser_beta * sqrt_f64(arg)) * inv_i0_beta
            } else {
                sinc_f64(u)
            };
            let tap = filt_scale * sinc_f64(filt_scale * d) * win;
            acc += f64::from(input[j as usize]) * tap;
            wsum += tap;
        }
        j += 1;
    }
    if wsum.abs() < 1e-12 {
        0.0
    } else {
        (acc / wsum) as f32
    }
}

/// Band-limited windowed-sinc resampler from `src_rate` to `dst_rate` into caller-supplied `out`.
/// Returns the number of output frames written.
pub fn windowed_sinc_resample_into(
    input: &[f32],
    src_rate: f64,
    dst_rate: f64,
    half_width: usize,
    kaiser_beta: f64,
    out: &mut [f32],
) -> usize {
    if !src_rate.is_finite()
        || !dst_rate.is_finite()
        || src_rate <= 0.0
        || dst_rate <= 0.0
        || half_width == 0
        || input.is_empty()
    {
        return 0;
    }
    if (src_rate - dst_rate).abs() < 1e-9 {
        let n = input.len().min(out.len());
        out[..n].copy_from_slice(&input[..n]);
        return n;
    }
    let out_len_total = floor_f64((input.len() as f64) * dst_rate / src_rate) as usize;
    let out_n = out_len_total.min(out.len());
    if out_n == 0 {
        return 0;
    }
    let cutoff_ratio = (dst_rate / src_rate).min(1.0);
    let step = src_rate / dst_rate;
    for (i, slot) in out.iter_mut().enumerate().take(out_n) {
        let center = (i as f64) * step;
        *slot = windowed_sinc_sample(input, center, cutoff_ratio, half_width, kaiser_beta);
    }
    out_n
}

/// Evaluate a single polyphase FIR filter branch centered at `center` in `input`.
#[inline]
pub fn polyphase_fir_step(input: &[f32], center: usize, phase_coeffs: &[f32]) -> f32 {
    if input.is_empty() || phase_coeffs.is_empty() {
        return 0.0;
    }
    let half = phase_coeffs.len() / 2;
    let len = input.len() as isize;
    let c = center as isize;
    let mut acc = 0.0f64;
    for (k, &w) in phase_coeffs.iter().enumerate() {
        let idx = c + (k as isize) - (half as isize);
        if idx >= 0 && idx < len {
            acc += f64::from(input[idx as usize]) * f64::from(w);
        }
    }
    acc as f32
}

#[inline]
fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Rational-ratio polyphase windowed-sinc resampler (`L = dst/gcd`, `M = src/gcd`).
pub fn resample_polyphase_into(
    src: &[f32],
    src_rate: u32,
    dst_rate: u32,
    half_taps: usize,
    out: &mut [f32],
) -> usize {
    if src_rate == 0 || dst_rate == 0 || half_taps == 0 || src.is_empty() {
        return 0;
    }
    if src_rate == dst_rate {
        let n = src.len().min(out.len());
        out[..n].copy_from_slice(&src[..n]);
        return n;
    }
    let g = gcd_u32(src_rate, dst_rate);
    let l = (dst_rate / g) as u64;
    let m = (src_rate / g) as u64;
    let out_len_total = ((src.len() as u64) * l / m) as usize;
    let out_n = out_len_total.min(out.len());
    if out_n == 0 {
        return 0;
    }
    let cutoff_ratio = (2.0 * antialias_cutoff_cycles(src_rate, dst_rate)) as f64;
    for (i, slot) in out.iter_mut().enumerate().take(out_n) {
        let num = (i as u64) * m;
        let base = (num / l) as f64;
        let phase = ((num % l) as f64) / (l as f64);
        *slot = windowed_sinc_sample(src, base + phase, cutoff_ratio, half_taps, 0.0);
    }
    out_n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_and_sinc_preserve_dc() {
        let src = [0.5f32; 64];
        let mut out_lin = [0.0f32; 32];
        let n1 = resample_linear_mono_into(&src, 48_000, 24_000, &mut out_lin);
        assert_eq!(n1, 32);
        assert!((out_lin[16] - 0.5).abs() < 1e-5);

        let mut out_sinc = [0.0f32; 32];
        let n2 = windowed_sinc_resample_into(&src, 48_000.0, 24_000.0, 16, 6.0, &mut out_sinc);
        assert_eq!(n2, 32);
        assert!((out_sinc[16] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn kaiser_i0_at_zero_is_one() {
        assert!((kaiser_bessel_i0(0.0) - 1.0).abs() < 1e-12);
        assert!(kaiser_bessel_i0(5.0) > 27.0);
    }
}
