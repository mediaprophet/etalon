//! Native CUDA execution of shared Forge kernels with owned device resources.
//!
//! Built-in operations validate shapes before dispatch. Supplied PTX loading and
//! launch require explicit unsafe contracts; CUDA does not validate shader memory access.
#![cfg(not(target_family = "wasm"))]

mod iq;
mod kernels;
mod runtime;
mod supplied;

pub use iq::CudaIqPowerPlan;
pub use kernels::CudaKernels;
pub use runtime::{
    CudaArgument, CudaBuffer, CudaContext, CudaDeviceInfo, CudaError, CudaKernel,
    CudaLaunchWorkspace, CudaModule,
};
pub use supplied::{CudaParameter, CudaPreparedKernel, CudaProgram};
