//! Hysteresis transitions over caller-selected scalar features.
//!
//! A transition is not a classification of a person, object, or event.
use crate::SignalError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThresholdState { Below, Above }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThresholdTransition {
    pub sample_index: usize,
    pub state: ThresholdState,
}

/// Caller-owned detector state. Thresholds use the same units as input.
#[derive(Debug)]
pub struct Hysteresis {
    low: f32,
    high: f32,
    state: ThresholdState,
}

impl Hysteresis {
    pub fn new(low: f32, high: f32, initial: ThresholdState) -> Result<Self, SignalError> {
        if !low.is_finite() || !high.is_finite() || low >= high {
            return Err(SignalError::InvalidParameter);
        }
        Ok(Self { low, high, state: initial })
    }

    pub const fn state(&self) -> ThresholdState { self.state }

    /// Each sample can produce at most one transition. Callers supply at
    /// least `samples.len()` slots, making capacity failure transactional.
    /// A rising edge occurs at >= high; a falling edge occurs at <= low.
    /// Slots after the returned count are untouched. Indices are chunk-local.
    pub fn process_into(&mut self, samples: &[f32], out: &mut [Option<ThresholdTransition>]) -> Result<usize, SignalError> {
        if out.len() < samples.len() { return Err(SignalError::LengthMismatch); }
        if samples.iter().any(|x| !x.is_finite()) { return Err(SignalError::NonFiniteInput); }
        let mut count = 0;
        let mut state = self.state;
        for (sample_index, &value) in samples.iter().enumerate() {
            let next = match state {
                ThresholdState::Below if value >= self.high => ThresholdState::Above,
                ThresholdState::Above if value <= self.low => ThresholdState::Below,
                _ => state,
            };
            if next != state {
                out[count] = Some(ThresholdTransition { sample_index, state: next });
                count += 1;
                state = next;
            }
        }
        self.state = state;
        Ok(count)
    }
}

#[cfg(test)]
#[path = "detection_tests.rs"]
mod tests;
