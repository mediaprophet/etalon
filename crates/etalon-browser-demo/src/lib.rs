//! Bounded browser smoke tests through Etalon's Rust-to-WebGPU backend.

#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(target_arch = "wasm32")]
mod compute;
#[cfg(target_arch = "wasm32")]
mod extensions;
#[cfg(target_arch = "wasm32")]
mod readback;
#[cfg(target_arch = "wasm32")]
mod render;

#[cfg(target_arch = "wasm32")]
pub use browser::{create_session, create_session_for_profile, Session};

#[cfg(any(target_arch = "wasm32", test))]
mod requirements;

#[cfg(target_arch = "wasm32")]
mod adapter;
#[cfg(target_arch = "wasm32")]
mod domain_agent;
#[cfg(target_arch = "wasm32")]
mod domain_nquin;
#[cfg(target_arch = "wasm32")]
mod domain_tensor;

#[cfg(target_arch = "wasm32")]
mod backend_contracts;
#[cfg(target_arch = "wasm32")]
pub use backend_contracts::inspect_adapter_contract;

#[cfg(target_arch = "wasm32")]
mod adapter_control;
#[cfg(target_arch = "wasm32")]
pub use adapter_control::adapter_acquisition_control;
