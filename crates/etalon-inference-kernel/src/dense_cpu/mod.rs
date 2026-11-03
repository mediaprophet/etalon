//! Dense CPU transformer execution over caller-owned state and tensor kernels.
//!
//! Tensor adapters supply real row decoding and GEMV, including target-specific
//! quantization. This module executes the forward math; it supplies no tokenizer,
//! architecture policy, backend fallback, sampler, publication or state rollback.
//! A failed step may have changed KV and scratch. Positions remain caller-owned.
//! Byte budgets cover the thirteen execution buffers, excluding borrowed model
//! bindings, weights and callback-owned storage. Adapters must separately qualify
//! their codec/target behavior and allocation contract; the trait cannot enforce it.

mod attention;
mod forward;
mod math;
mod plan;
mod types;

pub use forward::{ingest_token, reset_kv, run_token};
pub use plan::{DensePlan, DenseRequirements};
pub use types::*;

#[cfg(test)]
mod tests;
