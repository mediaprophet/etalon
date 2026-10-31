//! Caller-buffered streaming primitives for finite scalar signals.
//!
//! `MovingAverage` uses partial-window means during warm-up. Its history slice
//! is caller-owned and must be preserved exactly between calls; finite-value
//! checks cannot detect altered but finite ring contents. Do not share one
//! history slice between filter instances. A discontinuity requires an explicit
//! reset; samples across a gap are never joined implicitly.
//! `subtract_baseline` is suitable for subtracting the moving-average output.

use crate::SignalError;

/// Fixed-width running mean state. The sample ring is supplied by the caller.
#[derive(Debug, PartialEq)]
pub struct MovingAverage {
    window_len: usize,
    next: usize,
    filled: usize,
    sum: f64,
}

impl MovingAverage {
    /// Create a running mean with a nonzero window length.
    pub const fn new(window_len: usize) -> Result<Self, SignalError> {
        if window_len == 0 {
            return Err(SignalError::InvalidParameter);
        }
        Ok(Self {
            window_len,
            next: 0,
            filled: 0,
            sum: 0.0,
        })
    }

    pub const fn window_len(&self) -> usize {
        self.window_len
    }

    pub const fn warmup_remaining(&self) -> usize {
        self.window_len - self.filled
    }

    /// Process a block into an equally sized output slice.
    ///
    /// `history` must have exactly `window_len()` elements and be preserved
    /// unchanged between calls except for updates made by this filter. During
    /// warm-up, each output is divided by the number of
    /// samples seen so far. All inputs and active history values are validated
    /// and the entire block is preflighted before either output or state changes.
    pub fn process(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        history: &mut [f32],
    ) -> Result<usize, SignalError> {
        if output.len() != input.len() || history.len() != self.window_len {
            return Err(SignalError::LengthMismatch);
        }
        if self.next >= self.window_len || self.filled > self.window_len || !self.sum.is_finite() {
            return Err(SignalError::InvalidParameter);
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(SignalError::NonFiniteInput);
        }
        for offset in 0..self.filled {
            let backward = offset + 1;
            let index = if backward <= self.next {
                self.next - backward
            } else {
                self.window_len - (backward - self.next)
            };
            if !history[index].is_finite() {
                return Err(SignalError::NonFiniteInput);
            }
        }
        if input.is_empty() {
            return Ok(0);
        }

        // Preflight using the retained ring and the input block as an implicit
        // continuation. This pass detects numeric failures before publishing.
        let mut check_sum = self.sum;
        let mut check_filled = self.filled;
        for (index, &sample) in input.iter().enumerate() {
            if check_filled == self.window_len {
                let first_eviction = self.window_len - self.filled;
                let evictions_before = index - first_eviction;
                let evicted = if evictions_before < self.filled {
                    let oldest = if self.next >= self.filled {
                        self.next - self.filled
                    } else {
                        self.window_len - (self.filled - self.next)
                    };
                    history[ring_advance(oldest, evictions_before, self.window_len)]
                } else {
                    input[evictions_before - self.filled]
                };
                check_sum -= f64::from(evicted);
            } else {
                check_filled += 1;
            }
            check_sum += f64::from(sample);
            let mean = check_sum / check_filled as f64;
            if !check_sum.is_finite() || !mean.is_finite() || !(mean as f32).is_finite() {
                return Err(SignalError::NumericRange);
            }
        }

        let mut sum = self.sum;
        let mut filled = self.filled;
        let mut next = self.next;
        for (index, &sample) in input.iter().enumerate() {
            if filled == self.window_len {
                sum -= f64::from(history[next]);
            } else {
                filled += 1;
            }
            sum += f64::from(sample);
            output[index] = (sum / filled as f64) as f32;
            history[next] = sample;
            next += 1;
            if next == self.window_len {
                next = 0;
            }
        }
        self.sum = sum;
        self.filled = filled;
        self.next = next;
        Ok(input.len())
    }

    /// Clear the running window. On failure, history and state are unchanged.
    pub fn reset(&mut self, history: &mut [f32]) -> Result<(), SignalError> {
        if history.len() != self.window_len {
            return Err(SignalError::LengthMismatch);
        }
        history.fill(0.0);
        self.next = 0;
        self.filled = 0;
        self.sum = 0.0;
        Ok(())
    }

    /// Begin a fresh window after a known stream gap or discontinuity.
    pub fn reset_after_gap(&mut self, history: &mut [f32]) -> Result<(), SignalError> {
        self.reset(history)
    }
}

fn ring_advance(index: usize, advance: usize, len: usize) -> usize {
    let until_wrap = len - index;
    if advance >= until_wrap {
        advance - until_wrap
    } else {
        index + advance
    }
}

/// Subtract a caller-supplied, sample-aligned baseline from finite scalar data.
///
/// Both slices and the output must have equal lengths. The full operation is
/// preflighted in f64, so a non-finite input or f32-range overflow leaves output
/// unchanged. No baseline estimate or calibration is inferred here.
pub fn subtract_baseline(
    input: &[f32],
    baseline: &[f32],
    output: &mut [f32],
) -> Result<usize, SignalError> {
    if input.len() != baseline.len() || input.len() != output.len() {
        return Err(SignalError::LengthMismatch);
    }
    for (&sample, &base) in input.iter().zip(baseline) {
        if !sample.is_finite() || !base.is_finite() {
            return Err(SignalError::NonFiniteInput);
        }
        let delta = f64::from(sample) - f64::from(base);
        if !delta.is_finite() || (delta as f32).is_infinite() {
            return Err(SignalError::NumericRange);
        }
    }
    for ((target, &sample), &base) in output.iter_mut().zip(input).zip(baseline) {
        *target = (f64::from(sample) - f64::from(base)) as f32;
    }
    Ok(input.len())
}

#[cfg(test)]
#[path = "temporal_tests.rs"]
mod tests;
