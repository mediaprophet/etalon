//! Silicon steward, capability state machine, and safe resource lease manager.
//!
//! Provides the coordination layer between cold memory-mapping / kernel preparation
//! and zero-heap hot-path hardware submission across Vulkan, DX12, Metal, and CUDA.
//! CUDA and OS memory interop are native-only. Accelerator reports and portable
//! compute, lease and slab contracts remain available on WebAssembly.

pub mod accelerator;
pub mod capability;
pub mod compute_lane;
#[cfg(not(target_family = "wasm"))]
pub mod cuda_lane;
pub mod hal_lane;
#[cfg(not(target_family = "wasm"))]
pub mod interop;
pub mod lease;
pub mod slab;

/// Optional sensing facade. Individual device and acceleration providers are
/// selected separately; enabling this module does not discover or open devices.
#[cfg(feature = "sensing")]
pub use etalon_sense as sensing;

pub use capability::{AgentError, CapabilityState, SiliconCapability};
pub use compute_lane::{ComputeLane, PerQueueCompletionToken, QueueId, SubmissionToken};
#[cfg(not(target_family = "wasm"))]
pub use cuda_lane::{
    CudaComputeLane, CudaDriverApi, CudaPreparedPlan, CudaSubmission, PinBackend, PinnedRegion,
};
pub use hal_lane::{HalComputeLane, HalOracleError, HalPreparedPlan, HalSubmission};
pub use lease::{AccessRights, AllocationId, BufferRegion, SubmissionLease};
pub use slab::{BindingUsage, BufferView, MemoryTopology, QualiaSlabAllocator};
