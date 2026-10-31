//! Explicit, bounded mapping between two monotonic sensor clock domains.
//!
//! This contract does not identify either clock as UTC and does not infer a
//! mapping from timestamps alone. Callers provide two calibration anchors, a
//! source-clock validity interval, and a reference-clock model-error bound.
//! Mapping assumes an affine clock relationship over that interval. The
//! reported uncertainty includes the caller's error allowance, but this module
//! cannot verify it or bound time-varying drift/model error on the caller's
//! behalf.

use crate::time::{ClockStamp, TimeInterval};
use crate::ContractError;

const PPB_SCALE: i128 = 1_000_000_000;

/// Result of validating or applying a clock synchronization calibration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynchronizationError {
    Contract(ContractError),
    UnknownSynchronization,
    InvalidAnchors,
    ExcessiveDrift,
    UncertaintyExceeded,
    OutsideValidity,
}

impl From<ContractError> for SynchronizationError {
    fn from(value: ContractError) -> Self {
        Self::Contract(value)
    }
}

/// A paired timestamp observation recorded during calibration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClockAnchor {
    source: ClockStamp,
    reference: ClockStamp,
}

impl ClockAnchor {
    pub const fn new(source: ClockStamp, reference: ClockStamp) -> Self {
        Self { source, reference }
    }

    pub const fn source(self) -> ClockStamp {
        self.source
    }

    pub const fn reference(self) -> ClockStamp {
        self.reference
    }
}

/// Validated affine mapping from a source monotonic clock to a reference
/// monotonic clock, calibrated from two explicit anchor pairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClockSynchronization {
    synchronization_id: u64,
    source_clock_id: u64,
    source_epoch: u32,
    reference_clock_id: u64,
    reference_epoch: u32,
    first: ClockAnchor,
    second: ClockAnchor,
    valid_source: TimeInterval,
    max_drift_ppb: u64,
    measured_drift_ppb: u64,
    model_error_bound_ns: u64,
    max_uncertainty_ns: u64,
    anchor_uncertainty_ns: u64,
}

impl ClockSynchronization {
    /// Validity must lie between the latest possible first source anchor and
    /// earliest possible second anchor, so every admitted time is interpolation.
    /// `max_drift_ppb` limits the nominal two-center fitted slope, not the
    /// worst-case physical oscillator drift permitted by uncertain anchors.
    pub fn new(
        synchronization_id: u64,
        first: ClockAnchor,
        second: ClockAnchor,
        valid_source: TimeInterval,
        max_drift_ppb: u64,
        model_error_bound_ns: u64,
        max_uncertainty_ns: u64,
    ) -> Result<Self, SynchronizationError> {
        if synchronization_id == 0 {
            return Err(SynchronizationError::Contract(
                ContractError::InvalidIdentifier,
            ));
        }
        let src_a = first.source;
        let src_b = second.source;
        let ref_a = first.reference;
        let ref_b = second.reference;

        if src_a.clock_id() != src_b.clock_id()
            || src_a.restart_epoch() != src_b.restart_epoch()
            || ref_a.clock_id() != ref_b.clock_id()
            || ref_a.restart_epoch() != ref_b.restart_epoch()
            || (src_a.clock_id() == ref_a.clock_id()
                && src_a.restart_epoch() == ref_a.restart_epoch())
        {
            return Err(SynchronizationError::InvalidAnchors);
        }

        let src_span = src_b
            .ticks_ns()
            .checked_sub(src_a.ticks_ns())
            .filter(|&span| span > 0)
            .ok_or(SynchronizationError::InvalidAnchors)?;
        let ref_span = ref_b
            .ticks_ns()
            .checked_sub(ref_a.ticks_ns())
            .filter(|&span| span > 0)
            .ok_or(SynchronizationError::InvalidAnchors)?;
        if src_a.interval().upper_ns() >= src_b.interval().lower_ns()
            || ref_a.interval().upper_ns() >= ref_b.interval().lower_ns()
        {
            return Err(SynchronizationError::InvalidAnchors);
        }

        if valid_source.lower_ns() >= valid_source.upper_ns()
            || valid_source.lower_ns() < src_a.interval().upper_ns()
            || valid_source.upper_ns() > src_b.interval().lower_ns()
        {
            return Err(SynchronizationError::OutsideValidity);
        }

        let drift_delta = i128::from(ref_span) - i128::from(src_span);
        let drift_numerator =
            drift_delta
                .checked_mul(PPB_SCALE)
                .ok_or(SynchronizationError::Contract(
                    ContractError::ArithmeticOverflow,
                ))?;
        let drift_abs = if drift_numerator < 0 {
            -drift_numerator
        } else {
            drift_numerator
        };
        let drift_denominator = i128::from(src_span);
        let drift_ppb = (drift_abs / drift_denominator)
            + if drift_abs % drift_denominator != 0 {
                1
            } else {
                0
            };
        let measured_drift_ppb = u64::try_from(drift_ppb)
            .map_err(|_| SynchronizationError::Contract(ContractError::ArithmeticOverflow))?;
        if measured_drift_ppb > max_drift_ppb {
            return Err(SynchronizationError::ExcessiveDrift);
        }

        let first_source_uncertainty =
            scale_uncertainty(src_a.uncertainty_ns(), ref_span, src_span)?;
        let second_source_uncertainty =
            scale_uncertainty(src_b.uncertainty_ns(), ref_span, src_span)?;
        let first_uncertainty = first_source_uncertainty
            .checked_add(ref_a.uncertainty_ns())
            .ok_or(SynchronizationError::Contract(
                ContractError::ArithmeticOverflow,
            ))?;
        let second_uncertainty = second_source_uncertainty
            .checked_add(ref_b.uncertainty_ns())
            .ok_or(SynchronizationError::Contract(
                ContractError::ArithmeticOverflow,
            ))?;
        let anchor_uncertainty_ns = first_uncertainty.max(second_uncertainty);
        let minimum_bound = anchor_uncertainty_ns
            .checked_add(model_error_bound_ns)
            .and_then(|n| n.checked_add(1));
        if max_uncertainty_ns == 0
            || match minimum_bound {
                Some(bound) => bound > max_uncertainty_ns,
                None => true,
            }
        {
            return Err(SynchronizationError::UncertaintyExceeded);
        }

        Ok(Self {
            synchronization_id,
            source_clock_id: src_a.clock_id(),
            source_epoch: src_a.restart_epoch(),
            reference_clock_id: ref_a.clock_id(),
            reference_epoch: ref_a.restart_epoch(),
            first,
            second,
            valid_source,
            max_drift_ppb,
            measured_drift_ppb,
            model_error_bound_ns,
            max_uncertainty_ns,
            anchor_uncertainty_ns,
        })
    }

    /// Map a timestamp in the calibrated source domain to the reference domain.
    /// The entire source uncertainty interval must be valid for this mapping.
    pub fn map(&self, source: ClockStamp) -> Result<ClockStamp, SynchronizationError> {
        if source.clock_id() != self.source_clock_id || source.restart_epoch() != self.source_epoch
        {
            return Err(SynchronizationError::Contract(
                ContractError::ClockDomainMismatch,
            ));
        }

        let source_interval = source.interval();
        if source_interval.lower_ns() < self.valid_source.lower_ns()
            || source_interval.upper_ns() > self.valid_source.upper_ns()
        {
            return Err(SynchronizationError::OutsideValidity);
        }

        let source_anchor = self.first.source.ticks_ns();
        let target_anchor = self.first.reference.ticks_ns();
        let source_span = self.second.source.ticks_ns() - source_anchor;
        let target_span = self.second.reference.ticks_ns() - target_anchor;
        let source_offset = source
            .ticks_ns()
            .checked_sub(source_anchor)
            .ok_or(SynchronizationError::OutsideValidity)?;

        let scaled_offset = u128::from(source_offset)
            .checked_mul(u128::from(target_span))
            .ok_or(SynchronizationError::Contract(
                ContractError::ArithmeticOverflow,
            ))?
            / u128::from(source_span);
        let mapped_ticks =
            target_anchor
                .checked_add(u64::try_from(scaled_offset).map_err(|_| {
                    SynchronizationError::Contract(ContractError::ArithmeticOverflow)
                })?)
                .ok_or(SynchronizationError::Contract(
                    ContractError::ArithmeticOverflow,
                ))?;

        let input_uncertainty =
            scale_uncertainty(source.uncertainty_ns(), target_span, source_span)?;
        let uncertainty = self
            .anchor_uncertainty_ns
            .checked_add(input_uncertainty)
            .and_then(|n| n.checked_add(self.model_error_bound_ns))
            .and_then(|n| n.checked_add(1)) // integer interpolation rounding
            .ok_or(SynchronizationError::UncertaintyExceeded)?;
        if uncertainty > self.max_uncertainty_ns {
            return Err(SynchronizationError::UncertaintyExceeded);
        }

        Ok(ClockStamp::new(
            self.reference_clock_id,
            self.reference_epoch,
            mapped_ticks,
            uncertainty,
        )?)
    }

    pub const fn synchronization_id(&self) -> u64 {
        self.synchronization_id
    }

    pub const fn source_domain(&self) -> (u64, u32) {
        (self.source_clock_id, self.source_epoch)
    }

    pub const fn reference_domain(&self) -> (u64, u32) {
        (self.reference_clock_id, self.reference_epoch)
    }

    pub const fn anchors(&self) -> (ClockAnchor, ClockAnchor) {
        (self.first, self.second)
    }

    pub const fn valid_source(&self) -> TimeInterval {
        self.valid_source
    }

    /// Admission limit for the nominal fitted slope relative to unit slope.
    pub const fn max_drift_ppb(&self) -> u64 {
        self.max_drift_ppb
    }

    /// Nominal two-center slope deviation, rounded upward in parts per billion.
    /// This is not a worst-case bound on the physical clock's drift.
    pub const fn measured_drift_ppb(&self) -> u64 {
        self.measured_drift_ppb
    }

    /// Caller-supplied allowance for non-affine residual/drift error, in the
    /// reference clock's nanoseconds. It is included in every mapped bound but
    /// is not measured or verified here.
    pub const fn model_error_bound_ns(&self) -> u64 {
        self.model_error_bound_ns
    }

    pub const fn max_uncertainty_ns(&self) -> u64 {
        self.max_uncertainty_ns
    }
}

fn scale_uncertainty(
    source_uncertainty_ns: u64,
    target_span_ns: u64,
    source_span_ns: u64,
) -> Result<u64, SynchronizationError> {
    let scaled = u128::from(source_uncertainty_ns)
        .checked_mul(u128::from(target_span_ns))
        .ok_or(SynchronizationError::Contract(
            ContractError::ArithmeticOverflow,
        ))?;
    let denominator = u128::from(source_span_ns);
    if denominator == 0 {
        return Err(SynchronizationError::InvalidAnchors);
    }
    let whole = scaled / denominator;
    let rounded = whole + if scaled % denominator != 0 { 1 } else { 0 };
    u64::try_from(rounded).map_err(|_| SynchronizationError::UncertaintyExceeded)
}

/// Apply an optional synchronization without guessing when none is supplied.
pub fn map_synchronized(
    synchronization: Option<&ClockSynchronization>,
    source: ClockStamp,
) -> Result<ClockStamp, SynchronizationError> {
    synchronization
        .ok_or(SynchronizationError::UnknownSynchronization)?
        .map(source)
}

#[cfg(test)]
#[path = "synchronization_tests.rs"]
mod tests;
