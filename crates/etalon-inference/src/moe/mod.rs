//! Reusable expert routing, residency metadata, bounded tiles and streamed SwiGLU.
//! Architecture adapters select geometry and shared-gate semantics. Tile format
//! identifiers retain their historical names; model admission remains external.
pub mod residency;
pub mod streamed_routing;
pub use etalon_inference_kernel::moe::{dispatch, nvfp4, routing};
#[cfg(all(feature = "native-weights", not(target_arch = "wasm32")))]
pub mod expert_tile;

pub mod offload;
pub mod placement;

#[cfg(all(feature = "native-weights", not(target_arch = "wasm32")))]
pub mod streamed_executor;
#[cfg(all(feature = "native-weights", not(target_arch = "wasm32")))]
pub mod streamed_profile;

#[cfg(all(feature = "native-weights", not(target_arch = "wasm32")))]
pub use streamed_executor::{execute_streamed_swiglu_with_tiles, StreamedMoeError};
#[cfg(all(feature = "native-weights", not(target_arch = "wasm32")))]
pub use streamed_profile::{SharedExpertGate, StreamedSwiGluProfile};
