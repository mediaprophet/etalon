//! Classified byte reservations against caller-supplied budget snapshots.
//!
//! Concurrent reservations share one atomic total. External budgets and recorded
//! usage are snapshots, not a transaction with changing caller policy. Existing
//! guards remain releasable after a budget shrink; new admissions use new limits.
//! Separate total/class reads are diagnostic observations, not one atomic snapshot.

mod pool;

pub use pool::{Reservation, ReservationPool};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionLimits {
    pub global_capacity: u64,
    pub global_recorded: u64,
    pub partition_capacity: u64,
    pub partition_recorded: u64,
}

impl AdmissionLimits {
    pub const fn remaining_capacity(self) -> u64 {
        let global = self.global_capacity.saturating_sub(self.global_recorded);
        let partition = self
            .partition_capacity
            .saturating_sub(self.partition_recorded);
        if global < partition {
            global
        } else {
            partition
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReservationError {
    InvalidClass {
        class: usize,
        classes: usize,
    },
    BudgetExceeded {
        requested_bytes: u64,
        available_bytes: u64,
    },
    ArithmeticOverflow {
        requested_bytes: u64,
        available_bytes: u64,
    },
}

impl core::fmt::Display for ReservationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidClass { class, classes } => write!(f, "reservation class {class} outside {classes} classes"),
            Self::BudgetExceeded { requested_bytes, available_bytes } => write!(f, "reservation refused: requested {requested_bytes} bytes, {available_bytes} available"),
            Self::ArithmeticOverflow { requested_bytes, available_bytes } => write!(f, "reservation arithmetic overflow: requested {requested_bytes} bytes, {available_bytes} available"),
        }
    }
}

impl core::error::Error for ReservationError {}

#[cfg(test)]
mod tests;
