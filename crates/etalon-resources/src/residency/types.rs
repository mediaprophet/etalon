use core::fmt;

/// Maximum byte capacities for resident CPU heap/staging and GPU VRAM buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResidencyPoolCaps {
    pub max_cpu_bytes: u64,
    pub max_gpu_vram_bytes: u64,
}

/// A resident asset slot tracking resource identity, footprint, and LRU recency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResidencySlot {
    pub resource_hash: u64,
    pub cpu_bytes: u64,
    pub gpu_vram_bytes: u64,
    pub last_used_tick: u64,
    pub pinned: bool,
    pub active: bool,
}

/// Typed error returned during residency evaluation, admission, or release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyError {
    ResidencyBudgetExceeded {
        requested_cpu: u64,
        requested_gpu: u64,
        available_cpu: u64,
        available_gpu: u64,
    },
    SlotCapacityFull,
    ResourceNotFound(u64),
    ResourcePinned(u64),
    Cancelled,
}

impl fmt::Display for ResidencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResidencyBudgetExceeded {
                requested_cpu,
                requested_gpu,
                available_cpu,
                available_gpu,
            } => {
                write!(
                    f,
                    "residency budget exceeded: requested (CPU {requested_cpu}, GPU {requested_gpu}) > available (CPU {available_cpu}, GPU {available_gpu})"
                )
            }
            Self::SlotCapacityFull => write!(f, "residency governor slot capacity full"),
            Self::ResourceNotFound(hash) => {
                write!(f, "resource {hash:#018x} not found in residency pool")
            }
            Self::ResourcePinned(hash) => {
                write!(f, "resource {hash:#018x} is pinned and cannot be evicted")
            }
            Self::Cancelled => write!(f, "residency admission cancelled"),
        }
    }
}

impl core::error::Error for ResidencyError {}
