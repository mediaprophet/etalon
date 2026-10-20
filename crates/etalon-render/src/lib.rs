//! Decoupled volumetric, HUD, and EMF rendering substrate.

pub mod scene_contract;
pub mod volumetric;
pub mod hud;
pub mod emf;

pub use scene_contract::{FrameCameraPod, ManifoldProjectionPod};
