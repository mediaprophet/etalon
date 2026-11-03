#![cfg_attr(not(feature = "ggml"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod gguf;
#[cfg(feature = "ggml")]
pub mod ggml;
pub mod p64;
