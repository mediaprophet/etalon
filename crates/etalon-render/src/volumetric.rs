//! Volumetric ray-marcher and holographic projection pass.

use super::scene_contract::FrameCameraPod;

#[derive(Clone, Debug)]
pub struct VolumetricPassDesc {
    pub max_steps: u32,
    pub step_size: f32,
    pub absorption: f32,
}
