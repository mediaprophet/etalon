#![deny(unsafe_code)]
//! Numerical inference storage and budget mechanics; host policy remains external.
pub mod runtime;
pub mod backends;
#[cfg(all(not(target_arch = "wasm32"), feature = "gpu-attention"))]
pub mod native_attention;
#[cfg(feature = "gguf-cpu")]
pub mod cpu_gguf;

pub mod qwen;

pub mod moe;
#[cfg(all(feature = "native-weights", not(target_arch = "wasm32")))]
pub mod weights;

#[cfg(feature = "inference-model")]
pub mod ordinary;

#[cfg(feature = "cpu-inference")]
pub mod cpu_model;

#[cfg(all(not(target_arch = "wasm32"), feature = "gpu-attention"))]
pub mod native_resident;

#[cfg(feature = "gpu-attention")]
pub mod gpu_resources;

#[cfg(feature = "inference-model")]
pub mod projections;

#[cfg(feature = "gpu-attention")]
pub mod gpu_attention;

#[cfg(feature = "gpu-attention")]
pub mod browser_prefill;

#[cfg(feature = "gpu-attention")]
pub mod browser_projection;

#[cfg(feature = "inference-model")]
pub mod operator_runtime;

#[cfg(feature = "inference-model")]
pub mod operator_converter;
