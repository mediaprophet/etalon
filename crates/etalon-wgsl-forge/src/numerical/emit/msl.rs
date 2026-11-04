use super::GeneratedShader;
use crate::numerical::{ForgeError, KernelSpec, Schedule};
pub fn emit_msl(kernel: &KernelSpec, schedule: Schedule) -> Result<GeneratedShader, ForgeError> {
    super::emit_canonical(kernel, schedule, super::TargetBackend::Msl)
}
