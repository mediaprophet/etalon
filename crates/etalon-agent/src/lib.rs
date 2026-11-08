//! Silicon steward, capability state machine, and safe resource lease manager.
//!
//! Provides the coordination layer between cold memory-mapping / kernel preparation
//! and hardware submission across Etalon backends. The `cuda` feature exposes owned
//! CUDA execution of shared Forge kernels; the legacy raw-pointer lane cannot execute kernels.
//! CUDA and OS memory interop are native-only. Accelerator reports and portable
//! compute, lease and slab contracts remain available on WebAssembly.

pub mod accelerator;
pub mod capability;
pub mod compute_lane;
pub mod device_queue;
#[cfg(all(feature = "cuda", not(target_family = "wasm")))]
pub use etalon_cuda as cuda;
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
pub use device_queue::{DeviceQueueError, SharedDeviceQueue};
pub use hal_lane::{HalComputeLane, HalOracleError, HalPreparedPlan, HalSubmission};
pub use lease::{AccessRights, AllocationId, BufferRegion, SubmissionLease};
pub use slab::{BindingUsage, BufferView, MemoryTopology, QualiaSlabAllocator};

pub mod device_caps;
pub mod shared_device;
pub use etalon_resources::{device_lease, provider_scope};

#[cfg(all(feature = "diagnostics", not(target_family = "wasm")))]
pub use etalon_diagnostics as diagnostics;
