//! Stateful phase unwrapping for caller-normalized radians.

use crate::SignalError;

const PI: f64 = core::f32::consts::PI as f64;
const TAU: f64 = 2.0 * PI;

/// Explicit phase state at a chunk boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseState {
    previous_wrapped: f64,
    previous_unwrapped: f64,
    initialized: bool,
}

impl PhaseState {
    /// Start without a prior sample. The first valid phase becomes an anchor.
    pub const fn uninitialized() -> Self {
        Self {
            previous_wrapped: 0.0,
            previous_unwrapped: 0.0,
            initialized: false,
        }
    }

    /// Continue from an explicit previous wrapped/unwrapped phase pair.
    pub fn at(wrapped_radians: f32, unwrapped_radians: f64) -> Result<Self, SignalError> {
        if !is_normalized(wrapped_radians) || !unwrapped_radians.is_finite() {
            return Err(SignalError::InvalidParameter);
        }
        Ok(Self {
            previous_wrapped: f64::from(wrapped_radians),
            previous_unwrapped: unwrapped_radians,
            initialized: true,
        })
    }

    pub const fn is_initialized(self) -> bool {
        self.initialized
    }

    pub const fn previous_wrapped(self) -> Option<f64> {
        if self.initialized {
            Some(self.previous_wrapped)
        } else {
            None
        }
    }

    pub const fn previous_unwrapped(self) -> Option<f64> {
        if self.initialized {
            Some(self.previous_unwrapped)
        } else {
            None
        }
    }

    pub const fn reset(&mut self) {
        self.previous_wrapped = 0.0;
        self.previous_unwrapped = 0.0;
        self.initialized = false;
    }
}

/// Unwrap finite phases in radians into `out`, updating caller-owned state.
///
/// Each wrapped input must be normalized to `[-π, π]` using f32 precision.
/// The shortest signed delta is chosen; exact ±π differences preserve their
/// input sign. A `true` discontinuity flag starts a new phase segment anchored
/// at that sample's wrapped value. No discontinuities are inferred from data.
/// Input, flag, and output lengths are validated before state/output changes.
pub fn unwrap_phase_into(
    wrapped_radians: &[f32],
    discontinuities: &[bool],
    state: &mut PhaseState,
    out: &mut [f64],
) -> Result<usize, SignalError> {
    if discontinuities.len() != wrapped_radians.len() || out.len() < wrapped_radians.len() {
        return Err(SignalError::LengthMismatch);
    }
    if wrapped_radians.iter().any(|&phase| !is_normalized(phase)) {
        return Err(SignalError::InvalidParameter);
    }

    let mut next_state = *state;
    for (&phase, &discontinuity) in wrapped_radians.iter().zip(discontinuities) {
        let wrapped = f64::from(phase);
        if !next_state.initialized || discontinuity {
            next_state.previous_wrapped = wrapped;
            next_state.previous_unwrapped = wrapped;
            next_state.initialized = true;
            continue;
        }

        let mut delta = wrapped - next_state.previous_wrapped;
        if delta > PI {
            delta -= TAU;
        } else if delta < -PI {
            delta += TAU;
        }
        let unwrapped = next_state.previous_unwrapped + delta;
        if !unwrapped.is_finite() {
            return Err(SignalError::NumericRange);
        }
        next_state.previous_wrapped = wrapped;
        next_state.previous_unwrapped = unwrapped;
    }

    // Compute output in a second pass to preserve transactional state/output
    // semantics even if a future numeric change makes the preflight fail.
    next_state = *state;
    for (index, (&phase, &discontinuity)) in wrapped_radians.iter().zip(discontinuities).enumerate()
    {
        let wrapped = f64::from(phase);
        let unwrapped = if !next_state.initialized || discontinuity {
            wrapped
        } else {
            let mut delta = wrapped - next_state.previous_wrapped;
            if delta > PI {
                delta -= TAU;
            } else if delta < -PI {
                delta += TAU;
            }
            next_state.previous_unwrapped + delta
        };
        if !unwrapped.is_finite() {
            return Err(SignalError::NumericRange);
        }
        out[index] = unwrapped;
        next_state.previous_wrapped = wrapped;
        next_state.previous_unwrapped = unwrapped;
        next_state.initialized = true;
    }

    *state = next_state;
    Ok(wrapped_radians.len())
}

fn is_normalized(phase: f32) -> bool {
    phase.is_finite() && f64::from(phase) >= -PI && f64::from(phase) <= PI
}

#[cfg(test)]
#[path = "phase_tests.rs"]
mod tests;
