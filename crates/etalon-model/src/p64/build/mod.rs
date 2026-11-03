//! Caller-buffered P64 planning and writing for already prepared borrowed payloads.
//!
//! No sorting, role inference, dtype conversion or semantic manifold construction is
//! performed. Streaming transform callbacks and specialized FFN rewrite profiles are
//! outside this interface. Structural refusals leave caller buffers unchanged.

mod descriptors;
mod plan;
mod write;

pub use descriptors::{BuildBudget, BuildError, BuildInput, BuildProfile, TensorDescriptor};
pub use plan::{plan_into, BuildPlan};
pub use write::write_into;

#[cfg(test)]
mod tests;
