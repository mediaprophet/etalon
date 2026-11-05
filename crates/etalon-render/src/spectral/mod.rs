//! Spectral colour, continuous EMF basis, metamer fibre, gamut mapping, LUT inversion, and deferred resolve.

pub mod basis;
pub mod blend;
pub mod colour_kernel;
pub mod deferred_resolve;
pub mod gamut;
#[cfg(test)]
pub mod harness;
pub mod kernel;
pub mod lut;
pub mod metamer;
pub mod oracle;
pub mod sigma;

pub use basis::*;
pub use blend::*;
pub use colour_kernel::*;
pub use deferred_resolve::{
    evaluate_cmf_xyz, evaluate_deferred_resolve_pixel, generate_hero_wavelengths,
    DeferredCameraParams, GBufferPixel, ResolvedLight, DEFERRED_RESOLVE_WGSL,
};
pub use gamut::*;
pub use kernel::*;
pub use lut::*;
pub use metamer::*;
pub use oracle::*;
pub use sigma::{sigma_to_cie_xyz, sigma_to_linear_rgb};
