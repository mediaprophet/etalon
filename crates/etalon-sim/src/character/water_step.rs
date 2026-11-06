//! Character Controller Fluid & Water Locomotion Coupling (`water_step.rs`).
//!
//! Evaluates water surface at feet, handles wading drag, swimming, diving 6-DOF locomotion,
//! Archimedes buoyancy lifting, and dynamic surface wetness tracking.

use etalon_render::water_surface::SubmersionState;
use super::config::{lerp_accel, CharacterController};

impl CharacterController {
    /// Advances the character simulation with two-way fluid coupling:
    /// - Evaluates water surface at feet
    /// - Applies wading drag and swim locomotion
    /// - Computes Archimedes buoyancy lifting the character in deep water
    /// - Updates dynamic wetness and returns any new [`SubmersionState`] transition
    pub fn step_with_water<F, W>(
        &mut self,
        wish_dir: [f32; 2],
        sprint: bool,
        jump: bool,
        crouch: bool,
        dt: f32,
        ground_probe: F,
        water_probe: W,
    ) -> Option<SubmersionState>
    where
        F: Fn(f32, f32) -> (f32, [f32; 3]),
        W: Fn(f32, f32) -> f32,
    {
        let prev_submersion = self.state.submersion;
        let water_y = water_probe(self.state.position[0], self.state.position[2]);
        let raw_depth = (water_y - self.state.position[1]).max(0.0);
        self.state.water_depth = raw_depth;

        // Submersion classification
        let new_submersion = if raw_depth <= 0.05 {
            SubmersionState::Dry
        } else if raw_depth < 0.65 {
            SubmersionState::Wading
        } else if raw_depth < 1.65 {
            SubmersionState::Swimming
        } else {
            SubmersionState::Diving
        };
        self.state.submersion = new_submersion;

        // Dynamic wetness accumulation and drying
        if raw_depth > 0.05 {
            let target_wetness = (raw_depth / 1.5).min(1.0);
            if target_wetness > self.state.wetness {
                self.state.wetness = target_wetness;
            }
        } else {
            self.state.wetness = (self.state.wetness - dt / 30.0).max(0.0);
        }

        match new_submersion {
            SubmersionState::Dry => {
                self.step(wish_dir, sprint, jump, crouch, dt, ground_probe);
            }
            SubmersionState::Wading => {
                self.step(wish_dir, sprint, jump, crouch, dt, ground_probe);
                let drag = (1.0 - (raw_depth / 0.65) * 0.45).clamp(0.4, 1.0);
                self.state.velocity[0] *= drag;
                self.state.velocity[2] *= drag;
            }
            SubmersionState::Swimming => {
                self.step(wish_dir, false, false, false, dt, ground_probe);
                let target_y = water_y - 1.2;
                let vert_diff = target_y - self.state.position[1];
                self.state.velocity[1] = lerp_accel(self.state.velocity[1], vert_diff * 4.0, 15.0, dt);
                self.state.position[1] += self.state.velocity[1] * dt;
                self.state.is_grounded = false;
            }
            SubmersionState::Diving => {
                self.state.is_crouching = crouch;
                let cos_yaw = self.state.yaw.cos();
                let sin_yaw = self.state.yaw.sin();
                let cos_pitch = self.state.pitch.cos();
                let sin_pitch = self.state.pitch.sin();

                let forward = [sin_yaw * cos_pitch, -sin_pitch, cos_yaw * cos_pitch];
                let right = [cos_yaw, 0.0, -sin_yaw];

                let mut dir = [
                    forward[0] * wish_dir[0] + right[0] * wish_dir[1],
                    forward[1] * wish_dir[0],
                    forward[2] * wish_dir[0] + right[2] * wish_dir[1],
                ];
                if jump {
                    dir[1] += 0.8;
                }
                if crouch {
                    dir[1] -= 0.8;
                }

                let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
                if len > 0.01 {
                    let inv = 1.0 / len;
                    dir[0] *= inv;
                    dir[1] *= inv;
                    dir[2] *= inv;
                }

                let swim_speed = 3.2;
                self.state.velocity[0] = lerp_accel(self.state.velocity[0], dir[0] * swim_speed, 12.0, dt);
                self.state.velocity[1] = lerp_accel(self.state.velocity[1], dir[1] * swim_speed + 0.3, 12.0, dt);
                self.state.velocity[2] = lerp_accel(self.state.velocity[2], dir[2] * swim_speed, 12.0, dt);

                self.state.position[0] += self.state.velocity[0] * dt;
                self.state.position[1] += self.state.velocity[1] * dt;
                self.state.position[2] += self.state.velocity[2] * dt;
                self.state.is_grounded = false;
            }
        }

        if new_submersion != prev_submersion {
            Some(new_submersion)
        } else {
            None
        }
    }
}
