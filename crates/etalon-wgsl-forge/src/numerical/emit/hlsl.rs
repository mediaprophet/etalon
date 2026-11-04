use super::GeneratedShader;
use crate::numerical::{ForgeError, KernelSpec, Schedule};
pub fn emit_hlsl(kernel: &KernelSpec, schedule: Schedule) -> Result<GeneratedShader, ForgeError> {
    super::emit_canonical(kernel, schedule, super::TargetBackend::Hlsl)
}

pub use super::hlsl_wave::emit_gemv_wavematrix_hlsl;

#[cfg(test)]
include!("hlsl_migration_tests.rs");
