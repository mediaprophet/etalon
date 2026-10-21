//! Silicon capability verification and truthful state machine.

/// Errors returned by silicon capability validation and compute lane operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AgentError {
    #[error("Null host or device pointer encountered")]
    NullPointer,
    #[error("Unaligned pointer encountered")]
    UnalignedPointer,
    #[error("Zero-length allocation or registration requested")]
    ZeroLength,
    #[error("Requested silicon capability is unavailable or unverified")]
    CapabilityUnavailable,
    #[error("Compute lane is not prepared for execution")]
    NotPrepared,
}

/// State progression for hardware acceleration capabilities.
/// Workloads must only be routed to a lane once it reaches `Verified`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CapabilityState {
    /// Capability is physically unsupported or unavailable on this device.
    Unsupported,
    /// Hardware extension or adapter feature bit is present in enumeration.
    Detected,
    /// Shader, PTX, or MSL module compiles cleanly without driver rejection.
    Compilable,
    /// Pipeline, bind groups, and memory slabs are allocated and bound.
    Prepared,
    /// 1-step deterministic micro-oracle self-check passed on live hardware.
    Verified,
}

impl CapabilityState {
    /// Alias for [`Self::Unsupported`] when a driver or runtime capability is unavailable.
    #[allow(non_upper_case_globals)]
    pub const Unavailable: Self = Self::Unsupported;

    /// Returns true if the capability has been verified by the hardware oracle.
    #[inline]
    pub fn is_verified(&self) -> bool {
        matches!(self, Self::Verified)
    }

    /// Returns true if the capability is at least compilable.
    #[inline]
    pub fn is_compilable(&self) -> bool {
        *self >= Self::Compilable
    }
}

/// Description of an identified silicon capability.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SiliconCapability {
    pub name: &'static str,
    pub state: CapabilityState,
    pub details: Option<String>,
}

impl SiliconCapability {
    pub const fn new(name: &'static str, state: CapabilityState) -> Self {
        Self {
            name,
            state,
            details: None,
        }
    }
}
