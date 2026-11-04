//! Exact routed/shared NVFP4 dispatch over caller buffers and fixed stack scratch.
//! Inherited shape/panic/partial-state behavior is retained. No callback rollback.
//! Allocation guarantees require a separately qualified nonallocating fetcher.
mod error;
mod expert;
mod step;
pub use error::MoeDispatchError;
pub use expert::{evaluate_swiglu_expert_nvfp4, silu, ExpertWeightView};
pub use step::dispatch_moe_step;
pub const MAX_ROUTED_EXPERTS: usize = 512;
pub const MAX_INTERMEDIATE_DIM: usize = 4096;
#[cfg(test)]
mod tests;
