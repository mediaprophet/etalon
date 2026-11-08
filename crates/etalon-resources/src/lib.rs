//! Allocation-free resource accounting; device execution and budget policy belong to callers.
#![cfg_attr(not(feature = "device"), no_std)]
#![forbid(unsafe_code)]

pub mod reservation;
pub mod residency;

pub use reservation::{AdmissionLimits, Reservation, ReservationError, ReservationPool};

#[cfg(test)]
extern crate std;

#[cfg(feature = "device")]
pub mod device_caps;
#[cfg(feature = "device")]
pub mod device_lease;
#[cfg(feature = "device")]
pub mod provider_scope;
#[cfg(feature = "device")]
pub mod shared_device;

#[cfg(feature = "device")]
pub mod submission;
