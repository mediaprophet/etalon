//! Explicit local RTL-SDR receive; synchronous native calls may block indefinitely.
//! Native memory is unbounded. No timestamps, sample continuity or RF calibration
//! are inferred. Host authority must cover the entire open device lifetime.
#![deny(unsafe_op_in_unsafe_fn)]

mod abi;
pub mod adapter;
pub mod capture_payload;
mod config;
pub mod conversion;
mod device;
mod identity;
pub mod observation;
mod runtime;

#[cfg(test)]
mod tests;

pub use config::{AppliedConfig, Config, GainMode};
pub use device::{DeviceState, ReadReport, RtlSdrDevice};
pub use identity::UsbIdentity;
pub use runtime::RtlSdrRuntime;

pub const MAX_PAIRS: usize = 1_048_576;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Open,
    PostOpen,
    Configure,
    PostConfigure,
    Read,
    ReadReady,
    PostRead,
}

pub struct GateContext<'a> {
    pub stage: Stage,
    pub target_id: u64,
    pub generation: u32,
    pub device_index: u32,
    pub requested: Option<Config>,
    pub applied: Option<AppliedConfig>,
    pub identity: Option<&'a UsbIdentity>,
    pub pair_count: Option<usize>,
}

pub trait RtlSdrGate {
    /// Authorize the exact operation and its side effects. The host must hold
    /// an immutable policy/device lease, including between synchronous reads.
    fn allow(&self, context: &GateContext<'_>) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    UnsupportedPlatform,
    Admission,
    LibraryLoad,
    MissingSymbol,
    Synchronization,
    NotAuthorized,
    InvalidConfig,
    InvalidState,
    InvalidBuffer,
    InvalidIdentity,
    InvalidNativeHandle,
    InvalidNativeReport,
    ConfigurationChanged,
    SequenceExhausted,
    Native {
        operation: &'static str,
        code: i32,
    },
    PartialRead {
        actual_bytes: i32,
        expected_bytes: usize,
    },
    CloseOutcomeUnknown,
    CleanupFailed {
        primary_operation: &'static str,
        primary_code: Option<i32>,
        cleanup_code: i32,
    },
}
