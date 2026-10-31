//! Bounded cross-correlation for interleaved complex I/Q samples.

use crate::SignalError;

/// Explicit scaling for the finite-overlap complex correlation sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrelationScaling {
    /// Return the raw sum over the overlap.
    RawSum,
    /// Divide the raw sum by the number of overlapping complex samples.
    MeanByOverlap,
}

/// Complex result of one cross-correlation at a selected lag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplexCorrelation {
    real: f64,
    imag: f64,
    overlap_samples: usize,
}

impl ComplexCorrelation {
    pub const fn real(self) -> f64 {
        self.real
    }

    pub const fn imag(self) -> f64 {
        self.imag
    }

    pub const fn overlap_samples(self) -> usize {
        self.overlap_samples
    }
}

/// Cross-correlate interleaved complex sample slices at `lag`.
///
/// A complex sample is `(I, Q)` in adjacent f32 elements. Positive lag aligns
/// `observed[n + lag]` with `reference[n]`; negative lag advances the reference
/// slice. The fixed conjugation convention is
/// `sum(conj(reference[n]) * observed[n + lag])`, over only the valid overlap.
/// Inputs must be nonempty, even-length, and finite. Scaling is explicit; no
/// amplitude normalization, calibration, or physical-unit conversion occurs.
pub fn correlate_iq(
    reference: &[f32],
    observed: &[f32],
    lag: i64,
    scaling: CorrelationScaling,
) -> Result<ComplexCorrelation, SignalError> {
    if reference.is_empty() || observed.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if reference.len() % 2 != 0 || observed.len() % 2 != 0 {
        return Err(SignalError::LengthMismatch);
    }
    if reference.iter().any(|sample| !sample.is_finite())
        || observed.iter().any(|sample| !sample.is_finite())
    {
        return Err(SignalError::NonFiniteInput);
    }

    let reference_count = reference.len() / 2;
    let observed_count = observed.len() / 2;
    let (reference_start, observed_start, overlap) = if lag >= 0 {
        let observed_start = usize::try_from(lag).map_err(|_| SignalError::InvalidParameter)?;
        if observed_start >= observed_count {
            return Err(SignalError::InvalidParameter);
        }
        (
            0,
            observed_start,
            reference_count.min(observed_count - observed_start),
        )
    } else {
        let reference_start = usize::try_from(lag.unsigned_abs())
            .map_err(|_| SignalError::InvalidParameter)?;
        if reference_start >= reference_count {
            return Err(SignalError::InvalidParameter);
        }
        (
            reference_start,
            0,
            (reference_count - reference_start).min(observed_count),
        )
    };
    if overlap == 0 {
        return Err(SignalError::InvalidParameter);
    }

    let mut real = 0.0f64;
    let mut imag = 0.0f64;
    for index in 0..overlap {
        let reference_index = 2 * (reference_start + index);
        let observed_index = 2 * (observed_start + index);
        let ri = f64::from(reference[reference_index]);
        let rq = f64::from(reference[reference_index + 1]);
        let oi = f64::from(observed[observed_index]);
        let oq = f64::from(observed[observed_index + 1]);

        real += ri * oi + rq * oq;
        imag += ri * oq - rq * oi;
        if !real.is_finite() || !imag.is_finite() {
            return Err(SignalError::NumericRange);
        }
    }

    if scaling == CorrelationScaling::MeanByOverlap {
        let divisor = overlap as f64;
        real /= divisor;
        imag /= divisor;
    }
    if !real.is_finite() || !imag.is_finite() {
        return Err(SignalError::NumericRange);
    }

    Ok(ComplexCorrelation {
        real,
        imag,
        overlap_samples: overlap,
    })
}

#[cfg(test)]
#[path = "correlation_tests.rs"]
mod tests;
