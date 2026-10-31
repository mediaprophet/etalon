//! Iterative radix-2 complex transform with caller-supplied twiddles.
//!
//! This CPU routine does not generate twiddles, establish physical frequency
//! calibration, select an accelerator, or assert numerical equivalence across
//! devices. The caller supplies scratch and output buffers.
use crate::SignalError;

pub const MAX_FFT_SAMPLES: usize = 4096;

/// Transform interleaved complex f32 values using a radix-2, decimation-in-time
/// butterfly. `twiddles` is interleaved complex exp(-2πik/N), k=0..N/2, supplied
/// by the caller. Forward transform is unnormalized. Finite components must
/// be in [-1,1], but this routine cannot prove the supplied table is correct.
///
/// Input/output each contain 2N scalars; scratch has 2N f64 scalars; twiddles
/// contain N scalars (zero when N=1). N must be a power of two, 1..=4096.
/// On error output is unchanged, but scratch may have been overwritten.
pub fn forward_into(
    input: &[f32],
    twiddles: &[f32],
    scratch: &mut [f64],
    output: &mut [f32],
) -> Result<usize, SignalError> {
    if input.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if input.len() % 2 != 0 {
        return Err(SignalError::LengthMismatch);
    }
    let n = input.len() / 2;
    if !n.is_power_of_two() || n > MAX_FFT_SAMPLES {
        return Err(SignalError::InvalidParameter);
    }
    let expected_twiddles = if n == 1 { 0 } else { n };
    if output.len() != input.len()
        || scratch.len() != input.len()
        || twiddles.len() != expected_twiddles
    {
        return Err(SignalError::LengthMismatch);
    }
    if input.iter().any(|x| !x.is_finite()) || twiddles.iter().any(|x| !x.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    if twiddles.iter().any(|x| *x < -1.0 || *x > 1.0) {
        return Err(SignalError::InvalidParameter);
    }
    let mut reversed = 0usize;
    for index in 0..n {
        scratch[2 * reversed] = f64::from(input[2 * index]);
        scratch[2 * reversed + 1] = f64::from(input[2 * index + 1]);
        let mut bit = n >> 1;
        while bit > 0 && reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
    }
    let mut width = 2usize;
    while width <= n {
        let half = width / 2;
        let twiddle_step = n / width;
        for base in (0..n).step_by(width) {
            for offset in 0..half {
                let twiddle_index = 2 * offset * twiddle_step;
                let wr = f64::from(twiddles[twiddle_index]);
                let wi = f64::from(twiddles[twiddle_index + 1]);
                let left = 2 * (base + offset);
                let right = 2 * (base + offset + half);
                let rr = scratch[right] * wr - scratch[right + 1] * wi;
                let ri = scratch[right] * wi + scratch[right + 1] * wr;
                let lr = scratch[left];
                let li = scratch[left + 1];
                scratch[left] = lr + rr;
                scratch[left + 1] = li + ri;
                scratch[right] = lr - rr;
                scratch[right + 1] = li - ri;
            }
        }
        width *= 2;
    }
    if scratch
        .iter()
        .any(|x| !x.is_finite() || !(*x as f32).is_finite())
    {
        return Err(SignalError::NumericRange);
    }
    for (target, &value) in output.iter_mut().zip(scratch.iter()) {
        *target = value as f32;
    }
    Ok(n)
}

#[cfg(test)]
#[path = "fft_tests.rs"]
mod tests;
