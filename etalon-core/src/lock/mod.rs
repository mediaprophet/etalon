//! Instrumented lock types.
//!
//! This module defines a set of instrumented wrappers for the lock
//! types used in `etalon-core` ([`Mutex`], [`RwLock`], and
//! [`SnatchLock`]) that help us understand and validate `etalon-core`
//! synchronization.
//!
//! - The [`ranked`] module defines lock types that perform run-time
//!   checks to ensure that each thread acquires locks only in a
//!   specific order, to prevent deadlocks.
//!
//! - The [`observing`] module defines lock types that record
//!   `etalon-core`'s lock acquisition activity to disk, for later
//!   analysis by the `lock-analyzer` binary.
//!
//! - The [`vanilla`] module defines lock types that are
//!   uninstrumented, no-overhead wrappers around the standard lock
//!   types.
//!
//! If the `etalon_validate_locks` config is set (either directly, i.e. with
//! `RUSTFLAGS='--cfg etalon_validate_locks'`, or via the
//! `etalon_validate_locks_debug` `cfg` alias in `build.rs` that conditionally
//! enables it when debug assertions are also enabled), `etalon-core` uses the
//! [`ranked`] module's locks. `.config/cargo.toml` in the `etalon` tree specifies
//! `etalon_validate_locks_debug`, so lock validation is active by default in
//! `etalon` CI and `etalon` development trees, but not in `etalon`-using applications
//! unless specifically requested.
//!
//! If the `observe_locks` feature is enabled, `etalon-core` uses the
//! [`observing`] module's locks.
//!
//! Otherwise, `etalon-core` uses the [`vanilla`] module's locks.
//!
//! [`Mutex`]: etalon_sync::Mutex
//! [`RwLock`]: etalon_sync::RwLock
//! [`SnatchLock`]: crate::snatch::SnatchLock

pub mod rank;

#[cfg(feature = "std")] // requires thread-locals to work
#[cfg_attr(not(etalon_validate_locks), allow(dead_code))]
mod ranked;

#[cfg(feature = "observe_locks")]
#[cfg_attr(etalon_validate_locks, allow(dead_code))]
mod observing;

#[cfg_attr(
    any(etalon_validate_locks, feature = "observe_locks"),
    allow(dead_code)
)]
mod vanilla;

#[cfg(etalon_validate_locks)]
use ranked as chosen;

#[cfg(all(not(etalon_validate_locks), feature = "observe_locks"))]
use observing as chosen;

#[cfg(not(any(etalon_validate_locks, feature = "observe_locks")))]
use vanilla as chosen;

pub use chosen::{Mutex, MutexGuard, RankData, RwLock, RwLockReadGuard, RwLockWriteGuard};
