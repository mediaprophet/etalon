use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaStream};
use super::CudaComputeContext;

#[derive(Clone, Debug)]
pub struct CudaProviderLease {
    owner: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    generation: u64,
}

impl CudaProviderLease {
    pub fn capture(context: &CudaComputeContext) -> Self {
        Self { owner: context.ctx.clone(), stream: context.stream.clone(), generation: context.owner_generation }
    }
    pub fn generation(&self) -> u64 { self.generation }
    pub fn matches(&self, context: &CudaComputeContext) -> bool {
        self.generation == context.owner_generation && Arc::ptr_eq(&self.owner, &context.ctx) && Arc::ptr_eq(&self.stream, &context.stream)
    }
    pub fn live(&self) -> Result<(), String> {
        self.stream.synchronize().map_err(|error| format!("CUDA provider lost: {error:?}"))
    }
}
