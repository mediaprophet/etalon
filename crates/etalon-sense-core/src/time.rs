//! Validated, integer-only sensor clock stamps.
//!
//! A clock identifier names one clock domain. It does not imply UTC, device
//! host time, or a mapping to any other clock. Cross-domain conversion needs a
//! separately qualified synchronization contract.

use core::cmp::Ordering;

use crate::ContractError;

/// A timestamp interval with inclusive nanosecond bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeInterval {
    lower_ns: u64,
    upper_ns: u64,
}

impl TimeInterval {
    /// Construct a bounded interval. The lower bound must not exceed the upper.
    pub const fn new(lower_ns: u64, upper_ns: u64) -> Result<Self, ContractError> {
        if lower_ns > upper_ns {
            return Err(ContractError::InvalidTime);
        }
        Ok(Self { lower_ns, upper_ns })
    }

    /// Build an interval around a sample time. Uncertainty before zero clamps
    /// explicitly to zero; upper-bound overflow is rejected.
    pub const fn around(center_ns: u64, uncertainty_ns: u64) -> Result<Self, ContractError> {
        let upper_ns = match center_ns.checked_add(uncertainty_ns) {
            Some(value) => value,
            None => return Err(ContractError::InvalidTime),
        };
        Ok(Self {
            lower_ns: center_ns.saturating_sub(uncertainty_ns),
            upper_ns,
        })
    }

    pub const fn lower_ns(self) -> u64 {
        self.lower_ns
    }

    pub const fn upper_ns(self) -> u64 {
        self.upper_ns
    }

    /// Compare inclusive uncertainty intervals. Overlapping intervals do not
    /// establish a definite temporal order.
    pub const fn relation_to(self, other: Self) -> IntervalRelation {
        if self.upper_ns < other.lower_ns {
            IntervalRelation::Before
        } else if self.lower_ns > other.upper_ns {
            IntervalRelation::After
        } else {
            IntervalRelation::Overlaps
        }
    }
}

/// Ordering supported by two uncertainty intervals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntervalRelation {
    Before,
    Overlaps,
    After,
}

/// A validated timestamp in a single, explicitly named monotonic clock epoch.
///
/// The tuple `(clock_id, restart_epoch)` is the comparison domain. A restarted
/// device must advance `restart_epoch`; stamps from different domains cannot
/// be compared or aligned by this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClockStamp {
    clock_id: u64,
    restart_epoch: u32,
    ticks_ns: u64,
    uncertainty_ns: u64,
}

impl ClockStamp {
    /// Create a stamp, rejecting empty clock identities and overflowing
    /// uncertainty intervals.
    pub const fn new(
        clock_id: u64,
        restart_epoch: u32,
        ticks_ns: u64,
        uncertainty_ns: u64,
    ) -> Result<Self, ContractError> {
        if clock_id == 0 {
            return Err(ContractError::InvalidIdentifier);
        }
        if ticks_ns.checked_add(uncertainty_ns).is_none() {
            return Err(ContractError::InvalidTime);
        }
        Ok(Self {
            clock_id,
            restart_epoch,
            ticks_ns,
            uncertainty_ns,
        })
    }

    pub const fn clock_id(self) -> u64 {
        self.clock_id
    }

    pub const fn restart_epoch(self) -> u32 {
        self.restart_epoch
    }

    pub const fn ticks_ns(self) -> u64 {
        self.ticks_ns
    }

    pub const fn uncertainty_ns(self) -> u64 {
        self.uncertainty_ns
    }

    pub const fn interval(self) -> TimeInterval {
        // Construction validated this checked addition already.
        TimeInterval {
            lower_ns: self.ticks_ns.saturating_sub(self.uncertainty_ns),
            upper_ns: self.ticks_ns + self.uncertainty_ns,
        }
    }

    /// Compare this stamp's uncertainty interval to another stamp in the same
    /// clock domain and restart epoch.
    pub const fn relation_to(self, other: Self) -> Result<IntervalRelation, ContractError> {
        if self.clock_id != other.clock_id || self.restart_epoch != other.restart_epoch {
            return Err(ContractError::ClockDomainMismatch);
        }
        Ok(self.interval().relation_to(other.interval()))
    }

    /// Exact center-time delta (`self - other`) in nanoseconds. `i128` retains
    /// the full difference between any two `u64` ticks without truncation.
    pub const fn delta_ns(self, other: Self) -> Result<i128, ContractError> {
        if self.clock_id != other.clock_id || self.restart_epoch != other.restart_epoch {
            return Err(ContractError::ClockDomainMismatch);
        }
        Ok(self.ticks_ns as i128 - other.ticks_ns as i128)
    }

    /// Exact center-time ordering, subject to a matching clock domain.
    pub fn compare_center(self, other: Self) -> Result<Ordering, ContractError> {
        if self.clock_id != other.clock_id || self.restart_epoch != other.restart_epoch {
            return Err(ContractError::ClockDomainMismatch);
        }
        Ok(self.ticks_ns.cmp(&other.ticks_ns))
    }
}

#[cfg(test)]
#[path = "time_tests.rs"]
mod tests;
