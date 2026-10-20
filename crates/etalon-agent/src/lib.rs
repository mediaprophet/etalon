//! Silicon steward, capability state machine, and safe resource lease manager.
//!
//! Provides the coordination layer between cold memory-mapping / kernel preparation
//! and zero-heap hot-path hardware submission across Vulkan, DX12, Metal, and CUDA.

pub mod capability;
pub mod lease;
pub mod slab;
pub mod compute_lane;
pub mod hal_lane;
pub mod cuda_lane;
pub mod interop;

pub use capability::{CapabilityState, SiliconCapability};
pub use compute_lane::{ComputeLane, PerQueueCompletionToken, QueueId};
pub use hal_lane::{HalComputeLane, HalPreparedPlan, HalSubmission};
pub use cuda_lane::{CudaComputeLane, CudaPreparedPlan, CudaSubmission};
pub use lease::{AccessRights, AllocationId, BufferRegion, SubmissionLease};
pub use slab::{BindingUsage, BufferView, MemoryTopology, QualiaSlabAllocator};
