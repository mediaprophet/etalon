//! Fixed-capacity residency metadata and deterministic LRU admission.
//!
//! Inline storage belongs to the caller; no heap or device operations occur.
//! Pinning prevents automatic eviction, not explicit release or cancellation.
//! Failed admission leaves all slots and counters unchanged. Ticks are opaque
//! caller values, ordered numerically with lowest-slot-index tie breaking.
//! This governor does not retire physical resources or synchronize GPU work.

mod governor;
mod types;

pub use governor::ResidencyGovernor;
pub use types::{ResidencyError, ResidencyPoolCaps, ResidencySlot};

#[cfg(test)]
mod tests;
