//! Generic independent and clustered expert execution.
mod types;
mod execution;
pub use types::*;
pub use execution::{dispatch_clustered_moe_step, gemv_q4k, silu};
#[cfg(test)]
mod tests;
