//! Kinematic Character Controller Engine (`character`).
//!
//! Provides a zero-heap, deterministic 64 Hz kinematic capsule character controller:
//! - Smooth ground movement, acceleration, braking, slope-limiting, step-climbing (`config`),
//! - Fluid submersion locomotion (wading drag, swimming, 6-DOF diving) (`water_step`),
//! - Full 3D XPBD capsule sweeps, stair auto-stepping, moving platform velocity inheritance (`physics_step`).

pub mod config;
pub mod physics_step;
pub mod water_step;

#[cfg(test)]
mod tests;

pub use config::*;
pub use physics_step::*;
