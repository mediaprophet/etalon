//! Synchronous host offload to an explicitly admitted Etalon DSP service.
//!
//! No library discovery, unsigned-domain control, CPU fallback or hardware
//! attestation is performed. Native runtime allocations are not bounded by
//! caller I/O limits: this provider declares `ColdUnbounded`. Synchronous calls
//! can block indefinitely; timeout or cancellation must never release an
//! in-flight buffer. Use an isolated host process for hard resource limits.
#![deny(unsafe_op_in_unsafe_fn)]

mod abi;
mod provider;
mod session;

pub use provider::FastRpcIqPowerProvider;
pub use session::{FastRpcSession, SessionState};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastRpcError {
    UnsupportedPlatform,
    DeploymentAdmission,
    ServiceSchema,
    LibraryLoad,
    MissingSymbol,
    RuntimeSynchronization,
    InvalidLimits,
    BufferSize,
    InvalidState,
    Open(i32),
    AdmissionCleanupFailed(i32),
    Invoke(i32),
    Close(i32),
    CloseOutcomeUnknown,
    ArgumentMutation,
    Wire,
    RequestExhausted,
    Contract(etalon_sense_core::ContractError),
}
