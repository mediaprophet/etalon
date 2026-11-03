//! Allocation-free resource accounting; device execution and budget policy belong to callers.
#![no_std]
#![forbid(unsafe_code)]

pub mod reservation;
pub mod residency;

pub use reservation::{AdmissionLimits, Reservation, ReservationError, ReservationPool};

#[cfg(test)]
extern crate std;
