#![cfg_attr(not(feature = "ggml"), no_std)]
#![forbid(unsafe_code)]
#![allow(
    clippy::disallowed_types,
    clippy::manual_clamp,
    clippy::manual_range_patterns,
    clippy::wrong_self_convention
)]

extern crate alloc;

#[cfg(feature = "ggml")]
pub mod ggml;
pub mod gguf;
pub mod p64;
pub mod tokenizer;

#[cfg(feature = "inference-model")]
pub mod tensor_roles;

#[cfg(feature = "inference-model")]
pub mod inference;

#[cfg(feature = "inference-model")]
pub mod operator_package;
