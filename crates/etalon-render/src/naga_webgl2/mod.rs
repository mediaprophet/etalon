//! Naga IR sanitizer and WGSL → GLSL ES 300 bridge for WebGL2 compatibility.

pub mod bridge;
pub mod sanitize;

pub use bridge::*;
pub use sanitize::*;
