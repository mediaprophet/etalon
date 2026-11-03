//! Allocation-free session metadata for approval-before-publication protocols.
//! This crate does not execute models, authenticate controllers, publish output,
//! or restore numerical state. Embedding adapters perform those operations.
//! A restoration acknowledgment records the caller's assertion, not proof that
//! KV, recurrent or sampler state was restored. Existing post-turn protocols
//! require a separate contract and are explicitly rejected here.
#![no_std]
#![forbid(unsafe_code)]

pub mod artifact;

mod control;
mod mode;
mod session;

pub use control::*;
pub use mode::*;
pub use session::*;

#[cfg(test)]
mod tests;
