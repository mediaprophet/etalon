//! 3D Physics & XPBD Capsule Sweep Coupling (`physics_step.rs`).
//!
//! Couples [`CharacterController`] with 3D physics narrowphase/broadphase:
//! - Iterative slide-and-collide capsule sweeps against [`BroadphaseBvh`] and [`ColliderShape3D`],
//! - Stair auto-stepping up to configured `max_step_height`,
//! - Ceiling downward velocity clamping on upward head impacts,
//! - Moving platform velocity inheritance from contacted kinematic/dynamic bodies.

use super::config::{lerp_accel, CharacterController};
use crate::physics3d::{cast_capsule, BroadphaseBvh, ColliderShape3D};
use etalon_render::spatial_math::pga::Motor;

impl CharacterController {
    /// Advances the character controller in full 3D space against a [`BroadphaseBvh`] and
    /// 3D collider shapes (`SdfAnalytical`, `Cuboid`, `Sphere`, `Capsule`, `ConvexHull`):
    /// - 3D capsule sweep (`cast_capsule`) with iterative slide-and-collide plane projection,
    /// - Stair auto-stepping up to `config.max_step_height`,
    /// - Ceiling / overhang downward velocity clamping on upward head impact,
    /// - Moving platform velocity inheritance from contacted kinematic/dynamic bodies.
    #[allow(clippy::too_many_arguments)]
    pub fn step_character_3d<const MAX_LEAVES: usize>(
        &mut self,
        wish_dir: [f32; 2],
        sprint: bool,
        jump: bool,
        crouch: bool,
        dt: f32,
        bvh: &BroadphaseBvh<MAX_LEAVES>,
        shapes: &[ColliderShape3D],
        motors: &[Motor],
        body_velocities: Option<&[[f32; 3]]>,
    ) {
        self.state.is_crouching = crouch;
        let active_height = if crouch {
            self.config.crouch_height
        } else {
            self.config.height
        };
        let radius = self.config.radius.max(0.05);
        let half_cyl = (active_height * 0.5 - radius).max(0.05);

        let target_speed = if crouch {
            self.config.walk_speed * 0.5
        } else if sprint {
            self.config.run_speed
        } else {
            self.config.walk_speed
        };

        // 1. Transform wish_dir by yaw
        let cos_yaw = self.state.yaw.cos();
        let sin_yaw = self.state.yaw.sin();
        let forward = [sin_yaw, 0.0, cos_yaw];
        let right = [cos_yaw, 0.0, -sin_yaw];

        let mut world_wish = [
            forward[0] * wish_dir[0] + right[0] * wish_dir[1],
            0.0,
            forward[2] * wish_dir[0] + right[2] * wish_dir[1],
        ];
        let wish_len = (world_wish[0] * world_wish[0] + world_wish[2] * world_wish[2]).sqrt();
        if wish_len > 1.0 {
            world_wish[0] /= wish_len;
            world_wish[2] /= wish_len;
        }

        let accel = if self.state.is_grounded {
            if wish_len > 0.01 {
                self.config.ground_accel
            } else {
                self.config.ground_decel
            }
        } else {
            self.config.air_accel
        };

        self.state.velocity[0] =
            lerp_accel(self.state.velocity[0], world_wish[0] * target_speed, accel, dt);
        self.state.velocity[2] =
            lerp_accel(self.state.velocity[2], world_wish[2] * target_speed, accel, dt);

        // 2. Jump & Gravity
        if self.state.is_grounded && jump {
            self.state.velocity[1] = self.config.jump_impulse;
            self.state.is_grounded = false;
        } else if !self.state.is_grounded {
            self.state.velocity[1] =
                (self.state.velocity[1] + self.config.gravity * dt).max(self.config.terminal_velocity);
        }

        let mut cap_center = [
            self.state.position[0] + self.state.platform_velocity[0] * dt,
            self.state.position[1] + active_height * 0.5 + self.state.platform_velocity[1] * dt,
            self.state.position[2] + self.state.platform_velocity[2] * dt,
        ];

        // 3. Horizontal 3D capsule sweep with stair auto-stepping & slide-and-collide planes
        let horiz_disp = [self.state.velocity[0] * dt, 0.0, self.state.velocity[2] * dt];
        let horiz_dist =
            (horiz_disp[0] * horiz_disp[0] + horiz_disp[2] * horiz_disp[2]).sqrt();
        if horiz_dist > 1e-5 {
            let h_dir = [horiz_disp[0] / horiz_dist, 0.0, horiz_disp[2] / horiz_dist];
            if let Some(hit) = cast_capsule(
                bvh,
                shapes,
                motors,
                cap_center,
                radius,
                half_cyl,
                h_dir,
                horiz_dist,
                None,
            ) {
                // Attempt stair auto-step if grounded and obstacle is steep
                let mut stepped = false;
                if self.state.is_grounded && self.config.max_step_height > 0.01 {
                    let step_up_center = [
                        cap_center[0],
                        cap_center[1] + self.config.max_step_height,
                        cap_center[2],
                    ];
                    let clear_up = cast_capsule(
                        bvh,
                        shapes,
                        motors,
                        cap_center,
                        radius,
                        half_cyl,
                        [0.0, 1.0, 0.0],
                        self.config.max_step_height,
                        None,
                    )
                    .is_none();

                    if clear_up {
                        let clear_fwd = cast_capsule(
                            bvh,
                            shapes,
                            motors,
                            step_up_center,
                            radius,
                            half_cyl,
                            h_dir,
                            horiz_dist,
                            None,
                        )
                        .is_none();

                        if clear_fwd {
                            let fwd_center = [
                                step_up_center[0] + horiz_disp[0],
                                step_up_center[1],
                                step_up_center[2] + horiz_disp[2],
                            ];
                            if let Some(down_hit) = cast_capsule(
                                bvh,
                                shapes,
                                motors,
                                fwd_center,
                                radius,
                                half_cyl,
                                [0.0, -1.0, 0.0],
                                self.config.max_step_height + 0.05,
                                None,
                            ) {
                                if down_hit.normal[1].acos() <= self.config.max_slope_angle {
                                    cap_center = [
                                        fwd_center[0],
                                        fwd_center[1] - (down_hit.toi - 0.002).max(0.0),
                                        fwd_center[2],
                                    ];
                                    stepped = true;
                                }
                            }
                        }
                    }
                }

                if !stepped {
                    // Advance up to contact and slide along the obstacle plane
                    let safe_toi = (hit.toi - 0.002).max(0.0);
                    cap_center[0] += h_dir[0] * safe_toi;
                    cap_center[2] += h_dir[2] * safe_toi;

                    let rem = horiz_dist - safe_toi;
                    let n = hit.normal;
                    let vn = h_dir[0] * n[0] + h_dir[2] * n[2];
                    let slide = [
                        (h_dir[0] - n[0] * vn) * rem,
                        0.0,
                        (h_dir[2] - n[2] * vn) * rem,
                    ];
                    let slide_len = (slide[0] * slide[0] + slide[2] * slide[2]).sqrt();
                    if slide_len > 1e-5 {
                        let s_dir = [slide[0] / slide_len, 0.0, slide[2] / slide_len];
                        if let Some(hit2) = cast_capsule(
                            bvh,
                            shapes,
                            motors,
                            cap_center,
                            radius,
                            half_cyl,
                            s_dir,
                            slide_len,
                            None,
                        ) {
                            let s_toi = (hit2.toi - 0.002).max(0.0);
                            cap_center[0] += s_dir[0] * s_toi;
                            cap_center[2] += s_dir[2] * s_toi;
                        } else {
                            cap_center[0] += slide[0];
                            cap_center[2] += slide[2];
                        }
                    }
                }
            } else {
                cap_center[0] += horiz_disp[0];
                cap_center[2] += horiz_disp[2];
            }
        }

        // 4. Vertical sweep: ceiling overhang stop (when vy > 0) and ground snap (when vy <= 0)
        let vy = self.state.velocity[1];
        if vy > 0.0 {
            let up_dist = vy * dt;
            if let Some(ceil_hit) = cast_capsule(
                bvh,
                shapes,
                motors,
                cap_center,
                radius,
                half_cyl,
                [0.0, 1.0, 0.0],
                up_dist,
                None,
            ) {
                cap_center[1] += (ceil_hit.toi - 0.002).max(0.0);
                self.state.velocity[1] = 0.0;
            } else {
                cap_center[1] += up_dist;
            }
            self.state.is_grounded = false;
            self.state.platform_velocity = [0.0; 3];
        } else {
            let down_dist = (-vy * dt).max(0.06);
            if let Some(ground_hit) = cast_capsule(
                bvh,
                shapes,
                motors,
                cap_center,
                radius,
                half_cyl,
                [0.0, -1.0, 0.0],
                down_dist,
                None,
            ) {
                cap_center[1] -= (ground_hit.toi - 0.001).max(0.0);
                self.state.ground_normal = ground_hit.normal;
                let slope_angle = ground_hit.normal[1].clamp(0.0, 1.0).acos();
                if slope_angle <= self.config.max_slope_angle {
                    self.state.is_grounded = true;
                    self.state.velocity[1] = 0.0;
                    if let Some(vels) = body_velocities {
                        let cid = ground_hit.collider_id as usize;
                        if cid < vels.len() {
                            self.state.platform_velocity = vels[cid];
                        } else {
                            self.state.platform_velocity = [0.0; 3];
                        }
                    }
                } else {
                    self.state.is_grounded = false;
                    self.state.platform_velocity = [0.0; 3];
                }
            } else {
                cap_center[1] += vy * dt;
                self.state.is_grounded = false;
                self.state.platform_velocity = [0.0; 3];
            }
        }

        if self.state.is_grounded {
            self.state.stride_distance += horiz_dist;
        }

        self.state.position = [
            cap_center[0],
            cap_center[1] - active_height * 0.5,
            cap_center[2],
        ];
    }
}

/// Convenience wrapper for [`CharacterController::step_character_3d`].
#[allow(clippy::too_many_arguments)]
pub fn step_character_3d<const MAX_LEAVES: usize>(
    controller: &mut CharacterController,
    wish_dir: [f32; 2],
    sprint: bool,
    jump: bool,
    crouch: bool,
    dt: f32,
    bvh: &BroadphaseBvh<MAX_LEAVES>,
    shapes: &[ColliderShape3D],
    motors: &[Motor],
    body_velocities: Option<&[[f32; 3]]>,
) {
    controller.step_character_3d(
        wish_dir,
        sprint,
        jump,
        crouch,
        dt,
        bvh,
        shapes,
        motors,
        body_velocities,
    );
}
