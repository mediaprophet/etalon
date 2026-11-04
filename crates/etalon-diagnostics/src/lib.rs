//! Native benchmark mechanisms; callers retain ranking and attestation policy.
#![cfg(not(target_family = "wasm"))]

#[cfg(feature = "gpu")]
pub mod device;
#[cfg(feature = "gpu")]
mod gpu;
pub mod ipc;
pub mod measurement;
mod pipe;
pub mod process;
mod types;

pub use types::{CircuitBench, CircuitKind, DeviceBenchmarkRequest, DeviceBenchmarkResponse};
