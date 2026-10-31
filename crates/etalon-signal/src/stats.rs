//! Allocation-free basic statistics for a finite sample slice.

use crate::SignalError;

/// Summary statistics in f64 so reductions do not overflow for finite f32 input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SignalSummary {
    mean: f64,
    mean_square: f64,
    variance: f64,
    minimum: f32,
    maximum: f32,
    sample_count: usize,
}

impl SignalSummary {
    pub const fn mean(self) -> f64 {
        self.mean
    }

    /// Mean square (RMS squared), preserving an exact nonnegative power metric
    /// without depending on a platform math library for square root.
    pub const fn mean_square(self) -> f64 {
        self.mean_square
    }

    pub const fn variance(self) -> f64 {
        self.variance
    }

    pub const fn minimum(self) -> f32 {
        self.minimum
    }

    pub const fn maximum(self) -> f32 {
        self.maximum
    }

    pub const fn sample_count(self) -> usize {
        self.sample_count
    }
}

/// Compute mean, population variance, mean-square power, and extrema.
///
/// The input must be nonempty and finite. Accumulators use f64; no heap or
/// hidden normalization is performed.
pub fn summarize(samples: &[f32]) -> Result<SignalSummary, SignalError> {
    if samples.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }

    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    let mut sum_squares = 0.0f64;
    let mut minimum = samples[0];
    let mut maximum = samples[0];

    for (index, &sample) in samples.iter().enumerate() {
        let value = f64::from(sample);
        let count = (index + 1) as f64;
        let delta = value - mean;
        mean += delta / count;
        let delta_after = value - mean;
        m2 += delta * delta_after;
        sum_squares += value * value;
        if sample < minimum {
            minimum = sample;
        }
        if sample > maximum {
            maximum = sample;
        }
    }

    let count = samples.len() as f64;
    let mean_square = sum_squares / count;
    let variance = m2 / count;
    if !mean.is_finite()
        || !mean_square.is_finite()
        || !variance.is_finite()
        || mean_square < 0.0
        || variance < 0.0
    {
        return Err(SignalError::NumericRange);
    }
    Ok(SignalSummary {
        mean,
        mean_square,
        variance,
        minimum,
        maximum,
        sample_count: samples.len(),
    })
}

#[cfg(test)]
#[path = "stats_tests.rs"]
mod tests;
