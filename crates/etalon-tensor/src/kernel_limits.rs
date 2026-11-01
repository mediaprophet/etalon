use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum TensorPipelineError {
    #[error("tensor kernels require a 64-invocation compute workgroup")]
    WorkgroupSize,
    #[error("tensor kernel exceeds device workgroup storage")]
    WorkgroupStorage,
    #[error("tensor kernel exceeds compute-stage storage buffer limit")]
    StorageBindings,
}

pub(crate) fn check(
    limits: &etalon::Limits,
    storage_buffers: u32,
    scratch_bytes: u32,
) -> Result<(), TensorPipelineError> {
    if limits.max_compute_invocations_per_workgroup < 64 || limits.max_compute_workgroup_size_x < 64
    {
        return Err(TensorPipelineError::WorkgroupSize);
    }
    if limits.max_compute_workgroup_storage_size < scratch_bytes {
        return Err(TensorPipelineError::WorkgroupStorage);
    }
    if limits.max_storage_buffers_per_shader_stage < storage_buffers {
        return Err(TensorPipelineError::StorageBindings);
    }
    Ok(())
}
