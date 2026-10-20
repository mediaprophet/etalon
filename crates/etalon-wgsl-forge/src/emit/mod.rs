//! Target backend emission dispatch and code generation.

pub mod wgsl;
pub mod msl;
pub mod hlsl_wave;
pub mod cuda_c;
pub mod cache;

use super::ir::{ForgeError, Op};
use super::schedule::Schedule;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TargetBackend {
    Wgsl,
    Msl,
    Hlsl,
    Cuda,
}

pub fn emit_kernel(op: &Op, sched: &Schedule, target: TargetBackend) -> Result<String, ForgeError> {
    match target {
        TargetBackend::Wgsl => wgsl::emit_wgsl_kernel(op, sched),
        TargetBackend::Msl => msl::emit_msl_kernel(op, sched),
        TargetBackend::Hlsl => hlsl_wave::emit_hlsl_wave_kernel(op, sched),
        TargetBackend::Cuda => cuda_c::emit_cuda_kernel(op, sched),
    }
}
