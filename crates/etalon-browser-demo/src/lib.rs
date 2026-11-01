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
