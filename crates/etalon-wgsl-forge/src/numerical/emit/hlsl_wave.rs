use crate::numerical::{ForgeError, KernelSpec, Schedule};
pub fn emit_gemv_wave_hlsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let spec: crate::KernelSpec = serde_json::from_value(serde_json::to_value(kernel)?)?;
    crate::emit::hlsl_wave::emit_gemv_wave_hlsl(source, &spec, schedule.into()).map_err(Into::into)
}
pub fn emit_fused_ffn_wave_hlsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let spec: crate::KernelSpec = serde_json::from_value(serde_json::to_value(kernel)?)?;
    crate::emit::hlsl_wave::emit_fused_ffn_wave_hlsl(source, &spec, schedule.into())
        .map_err(Into::into)
}
pub fn emit_topk_wave_hlsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let spec: crate::KernelSpec = serde_json::from_value(serde_json::to_value(kernel)?)?;
    crate::emit::hlsl_wave::emit_topk_wave_hlsl(source, &spec, schedule.into()).map_err(Into::into)
}
pub fn emit_gemv_wavematrix_hlsl(
    source: &mut String,
    kernel: &KernelSpec,
    schedule: Schedule,
) -> Result<(), ForgeError> {
    let spec: crate::KernelSpec = serde_json::from_value(serde_json::to_value(kernel)?)?;
    crate::emit::hlsl_wave::emit_gemv_wavematrix_hlsl(source, &spec, schedule.into())
        .map_err(Into::into)
}

#[cfg(test)]
include!("hlsl_wave_migration_tests.rs");
