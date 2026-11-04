use super::{GeneratedShader, TargetBackend};
use crate::numerical::{ForgeError, KernelSpec, Schedule};
pub(super) fn is_iq(kernel: &KernelSpec) -> bool {
    matches!(kernel.id.as_str(), "iq-power-f32" | "iq-power-i16")
}
pub(super) fn emit(
    kernel: &KernelSpec,
    schedule: Schedule,
    target: TargetBackend,
) -> Result<GeneratedShader, ForgeError> {
    super::emit_canonical(kernel, schedule, target)
}
