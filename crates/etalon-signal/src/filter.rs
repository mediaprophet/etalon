//! Stateful one-pole low-pass filter with caller-owned output storage.

use crate::SignalError;

/// A one-pole low-pass filter. The caller provides the initial state explicitly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnePoleLowPass {
    alpha: f64,
    state: f64,
}

impl OnePoleLowPass {
    /// `alpha` is the new-sample weight in `y = alpha*x + (1-alpha)*y_prev`.
    /// Values must be finite and in `[0, 1]`; `initial_state` is never defaulted.
    pub fn new(alpha: f32, initial_state: f32) -> Result<Self, SignalError> {
        if !alpha.is_finite()
            || !initial_state.is_finite()
            || !(0.0..=1.0).contains(&alpha)
        {
            return Err(SignalError::InvalidParameter);
        }
        Ok(Self {
            alpha: f64::from(alpha),
            state: f64::from(initial_state),
        })
    }

    pub fn state(&self) -> f64 {
        self.state
    }

    /// Process all samples and write results into `out[..samples.len()]`.
    ///
    /// Length and every input value are checked before output or filter state
    /// changes, so invalid input is transactional. Extra output capacity is
    /// untouched. Returns the number of samples written.
    pub fn process_into(&mut self, samples: &[f32], out: &mut [f32]) -> Result<usize, SignalError> {
        if out.len() < samples.len() {
            return Err(SignalError::LengthMismatch);
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SignalError::NonFiniteInput);
        }

        let mut state = self.state;
        for (index, &sample) in samples.iter().enumerate() {
            state = self.alpha * f64::from(sample) + (1.0 - self.alpha) * state;
            let output = state as f32;
            if !state.is_finite() || !output.is_finite() {
                return Err(SignalError::NumericRange);
            }
            out[index] = output;
        }
        self.state = state;
        Ok(samples.len())
    }
}

#[cfg(test)]
#[path = "filter_tests.rs"]
mod tests;
