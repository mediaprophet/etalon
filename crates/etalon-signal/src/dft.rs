//! Bounded reference complex transform with caller-supplied coefficients, plus
//! zero-heap in-place radix-2 FFT/IFFT and direct DFT primitives.
//!
//! This is a CPU reference operation, not an acceleration backend,
//! calibrated spectrum, or a guarantee that caller-supplied coefficients are DFT twiddles.
use crate::{fft::MAX_FFT_SAMPLES, filter::sin_cos_f64, SignalError};

/// Upper bound on the quadratic work and coefficient matrix size.
pub const MAX_DFT_SAMPLES: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DftScale {
    Raw,
    DivideBySampleCount,
}

/// Transform interleaved complex samples using row-major interleaved complex
/// coefficients: `output[k] = sum(input[n] * coefficient[k,n])`. The caller
/// selects sign, coefficient generation, frequency labeling, and windowing.
///
/// All inputs and results are checked before modifying output. Coefficient
/// components must lie in [-1, 1], but unit magnitude is not established.
/// `out` must contain exactly two values per input complex sample.
pub fn transform_into(
    input: &[f32],
    coefficients: &[f32],
    out: &mut [f64],
    scale: DftScale,
) -> Result<usize, SignalError> {
    if input.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if !input.len().is_multiple_of(2) {
        return Err(SignalError::LengthMismatch);
    }
    let count = input.len() / 2;
    if count > MAX_DFT_SAMPLES {
        return Err(SignalError::InvalidParameter);
    }
    let expected = count
        .checked_mul(count)
        .and_then(|v| v.checked_mul(2))
        .ok_or(SignalError::NumericRange)?;
    if coefficients.len() != expected || out.len() != input.len() {
        return Err(SignalError::LengthMismatch);
    }
    if input.iter().any(|v| !v.is_finite()) || coefficients.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    if coefficients.iter().any(|v| *v < -1.0 || *v > 1.0) {
        return Err(SignalError::InvalidParameter);
    }
    for bin in 0..count {
        let (real, imag) = calculate_bin(input, coefficients, bin, count, scale);
        if !real.is_finite() || !imag.is_finite() {
            return Err(SignalError::NumericRange);
        }
    }
    for bin in 0..count {
        let (real, imag) = calculate_bin(input, coefficients, bin, count, scale);
        out[bin * 2] = real;
        out[bin * 2 + 1] = imag;
    }
    Ok(count)
}

fn calculate_bin(
    input: &[f32],
    coefficients: &[f32],
    bin: usize,
    count: usize,
    scale: DftScale,
) -> (f64, f64) {
    let mut real = 0.0;
    let mut imag = 0.0;
    for index in 0..count {
        let a = f64::from(input[index * 2]);
        let b = f64::from(input[index * 2 + 1]);
        let c = f64::from(coefficients[(bin * count + index) * 2]);
        let d = f64::from(coefficients[(bin * count + index) * 2 + 1]);
        real += a * c - b * d;
        imag += a * d + b * c;
    }
    if scale == DftScale::DivideBySampleCount {
        (real / count as f64, imag / count as f64)
    } else {
        (real, imag)
    }
}

/// In-place iterative Cooley-Tukey radix-2 FFT (`inverse == false`, $e^{-i 2\pi k n / N}$)
/// or $1/N$-normalized IFFT (`inverse == true`, $\frac{1}{N} e^{+i 2\pi k n / N}$) over
/// interleaved complex `[re, im, ...]` samples for any power-of-two $N \ge 1$.
pub fn fft_radix2_interleaved_f32(data: &mut [f32], inverse: bool) -> Result<(), SignalError> {
    if data.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if !data.len().is_multiple_of(2) {
        return Err(SignalError::LengthMismatch);
    }
    let n = data.len() / 2;
    if !n.is_power_of_two() {
        return Err(SignalError::InvalidParameter);
    }
    if n == 1 {
        return Ok(());
    }

    // Bit-reversal permutation over complex pairs.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            data.swap(2 * i, 2 * j);
            data.swap(2 * i + 1, 2 * j + 1);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut width = 2usize;
    while width <= n {
        let half = width / 2;
        let theta = sign * core::f64::consts::TAU / (width as f64);
        for offset in 0..half {
            let (wi, wr) = sin_cos_f64(theta * (offset as f64));
            for base in (0..n).step_by(width) {
                let left = 2 * (base + offset);
                let right = 2 * (base + offset + half);
                let lr = f64::from(data[left]);
                let li = f64::from(data[left + 1]);
                let rr = f64::from(data[right]);
                let ri = f64::from(data[right + 1]);

                let tr = wr * rr - wi * ri;
                let ti = wr * ri + wi * rr;

                data[left] = (lr + tr) as f32;
                data[left + 1] = (li + ti) as f32;
                data[right] = (lr - tr) as f32;
                data[right + 1] = (li - ti) as f32;
            }
        }
        width <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / (n as f64);
        for v in data.iter_mut() {
            *v = (f64::from(*v) * inv_n) as f32;
        }
    }
    Ok(())
}

/// In-place iterative Cooley-Tukey radix-2 FFT (`inverse == false`, $e^{-i 2\pi k n / N}$)
/// or $1/N$-normalized IFFT (`inverse == true`, $\frac{1}{N} e^{+i 2\pi k n / N}$) over
/// interleaved complex `[re, im, ...]` samples with `MAX_FFT_SAMPLES` and finite validation.
pub fn fft_radix2_in_place(interleaved: &mut [f32], inverse: bool) -> Result<usize, SignalError> {
    if interleaved.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if !interleaved.len().is_multiple_of(2) {
        return Err(SignalError::LengthMismatch);
    }
    let n = interleaved.len() / 2;
    if !n.is_power_of_two() || n > MAX_FFT_SAMPLES {
        return Err(SignalError::InvalidParameter);
    }
    if interleaved.iter().any(|x| !x.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    fft_radix2_interleaved_f32(interleaved, inverse)?;
    if interleaved.iter().any(|x| !x.is_finite()) {
        return Err(SignalError::NumericRange);
    }
    Ok(n)
}

/// Compute the forward complex DFT ($e^{-i 2\pi k j / N}$) of interleaved complex `input`
/// into `out[..input.len()]`. Uses $O(N \log N)$ radix-2 FFT when $N$ is a power of two,
/// and $O(N^2)$ direct evaluation in `f64` otherwise.
pub fn dft_interleaved_into(input: &[f32], out: &mut [f32]) -> Result<usize, SignalError> {
    if input.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if !input.len().is_multiple_of(2) || out.len() < input.len() {
        return Err(SignalError::LengthMismatch);
    }
    let n = input.len() / 2;
    if input.iter().any(|x| !x.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    if n.is_power_of_two() && n <= MAX_FFT_SAMPLES {
        out[..input.len()].copy_from_slice(input);
        return fft_radix2_in_place(&mut out[..input.len()], false);
    }
    if n > MAX_DFT_SAMPLES {
        return Err(SignalError::InvalidParameter);
    }

    let scale_ang = -core::f64::consts::TAU / (n as f64);
    for k in 0..n {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for j in 0..n {
            let xr = f64::from(input[2 * j]);
            let xi = f64::from(input[2 * j + 1]);
            let (s, c) = sin_cos_f64(scale_ang * (k as f64) * (j as f64));
            re += xr * c - xi * s;
            im += xr * s + xi * c;
        }
        let re_f32 = re as f32;
        let im_f32 = im as f32;
        if !re_f32.is_finite() || !im_f32.is_finite() {
            return Err(SignalError::NumericRange);
        }
        out[2 * k] = re_f32;
        out[2 * k + 1] = im_f32;
    }
    Ok(n)
}

#[cfg(test)]
#[path = "dft_tests.rs"]
mod tests;
