use etalon_wgsl_forge::numerical::execute::cuda_lease::CudaProviderLease;

pub fn prepare_provider() -> Result<Option<CudaProviderLease>, String> {
    let _ = super::device::warm_cuda_context();
    let guard = super::device::multi_weight_device()
        .lock()
        .map_err(|_| "CUDA owner mutex poisoned")?;
    Ok(guard.as_ref().map(|owner| owner.ctx.provider_lease()))
}

pub fn validate_provider(lease: Option<&CudaProviderLease>) -> Result<(), String> {
    let guard = super::device::multi_weight_device()
        .lock()
        .map_err(|_| "CUDA owner mutex poisoned")?;
    match (guard.as_ref(), lease) {
        (Some(owner), Some(lease)) if lease.matches(&owner.ctx) => Ok(()),
        (None, None) => Ok(()),
        _ => Err("CUDA numerical owner changed after preparation".into()),
    }
}
