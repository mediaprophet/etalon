//! Silicon steward, capability state machine, and safe resource lease manager.
//!
//! Provides the coordination layer between cold memory-mapping / kernel preparation
//! and zero-heap hot-path hardware submission across Vulkan, DX12, Metal, and CUDA.

pub mod accelerator;
pub mod capability;
pub mod compute_lane;
pub mod cuda_lane;
pub mod hal_lane;
pub mod interop;
pub mod lease;
pub mod slab;

pub use capability::{AgentError, CapabilityState, SiliconCapability};
pub use compute_lane::{ComputeLane, PerQueueCompletionToken, QueueId, SubmissionToken};
pub use cuda_lane::{
    CudaComputeLane, CudaDriverApi, CudaPreparedPlan, CudaSubmission, PinBackend, PinnedRegion,
};
pub use hal_lane::{HalComputeLane, HalOracleError, HalPreparedPlan, HalSubmission};
pub use lease::{AccessRights, AllocationId, BufferRegion, SubmissionLease};
pub use slab::{BindingUsage, BufferView, MemoryTopology, QualiaSlabAllocator};
