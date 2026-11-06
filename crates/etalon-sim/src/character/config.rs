//! Kinematic Character Controller Configuration & Ground Stepping (`config.rs`).
//!
//! Provides:
//! - [`CharacterConfig`]: Physical dimensions, speed thresholds, gravity, and step limits.
//! - [`CharacterState`]: Runtime kinematic state (position, velocity, orientation, grounding, wetness).
//! - [`CharacterController`]: Controller instance with basic ground integration (`step()`).

use etalon_render::water_surface::SubmersionState;

/// Character controller movement configuration.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacterConfig {
    /// Capsule radius in meters.
    pub radius: f32,
    /// Standing capsule height in meters.
    pub height: f32,
    /// Crouching capsule height in meters.
    pub crouch_height: f32,
    /// Maximum step-up height in meters.
    pub max_step_height: f32,
    /// Maximum traversable slope in radians (e.g., 0.785 rad ≈ 45°).
    pub max_slope_angle: f32,
    /// Walking speed in m/s.
    pub walk_speed: f32,
    /// Sprinting speed in m/s.
    pub run_speed: f32,
    /// Acceleration on ground in m/s².
    pub ground_accel: f32,
    /// Deceleration on ground in m/s².
    pub ground_decel: f32,
    /// Air acceleration in m/s².
    pub air_accel: f32,
    /// Jump impulse velocity in m/s.
    pub jump_impulse: f32,
    /// Gravity acceleration in m/s² (negative downward, e.g. -9.81).
    pub gravity: f32,
    /// Maximum downward fall speed in m/s.
    pub terminal_velocity: f32,
}

impl Default for CharacterConfig {
    fn default() -> Self {
        Self {
            radius: 0.4,
            height: 1.8,
            crouch_height: 1.1,
            max_step_height: 0.35,
            max_slope_angle: 0.785398, // 45 degrees
            walk_speed: 4.5,
            run_speed: 8.5,
            ground_accel: 45.0,
            ground_decel: 30.0,
            air_accel: 8.0,
            jump_impulse: 5.8,
            gravity: -18.0,
            terminal_velocity: -35.0,
        }
    }
}

/// Dynamic runtime state of the character.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacterState {
    /// 3D position [x, y, z] in world meters (at the base/feet of the capsule).
    pub position: [f32; 3],
    /// 3D velocity [vx, vy, vz] in m/s.
    pub velocity: [f32; 3],
    /// Facing yaw in radians.
    pub yaw: f32,
    /// Pitch angle in radians (for head/look).
    pub pitch: f32,
    /// True if the character's feet are currently contacting traversable ground.
    pub is_grounded: bool,
    /// True if the character is currently crouching.
    pub is_crouching: bool,
    /// Surface normal vector [nx, ny, nz] of the contacted ground.
    pub ground_normal: [f32; 3],
    /// Moving platform velocity [vx, vy, vz] imparted to the character.
    pub platform_velocity: [f32; 3],
    /// Accumulated walking distance in meters (drives footstep sounds & head bob).
    pub stride_distance: f32,
    /// Fluid submersion locomotion state.
    pub submersion: SubmersionState,
    /// Water depth at character feet in meters.
    pub water_depth: f32,
    /// Dynamic surface wetness ratio in [0.0, 1.0].
    pub wetness: f32,
}

impl Default for CharacterState {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            is_grounded: true,
            is_crouching: false,
            ground_normal: [0.0, 1.0, 0.0],
            platform_velocity: [0.0, 0.0, 0.0],
            stride_distance: 0.0,
            submersion: SubmersionState::Dry,
            water_depth: 0.0,
            wetness: 0.0,
        }
    }
}

/// Zero-heap kinematic character controller.
#[derive(Debug, Clone, Copy)]
pub struct CharacterController {
    pub config: CharacterConfig,
    pub state: CharacterState,
}

impl CharacterController {
    /// Creates a new character controller with the given configuration and starting position.
    pub const fn new(config: CharacterConfig, start_position: [f32; 3]) -> Self {
        Self {
            config,
            state: CharacterState {
                position: start_position,
                velocity: [0.0, 0.0, 0.0],
                yaw: 0.0,
                pitch: 0.0,
                is_grounded: false,
                is_crouching: false,
                ground_normal: [0.0, 1.0, 0.0],
                platform_velocity: [0.0, 0.0, 0.0],
                stride_distance: 0.0,
                submersion: SubmersionState::Dry,
                water_depth: 0.0,
                wetness: 0.0,
            },
        }
    }

    /// Advances the character simulation by one fixed timestep $\Delta t$.
    ///
    /// - `wish_dir`: Normalized 2D horizontal input direction `[forward, strafe_right]`.
    /// - `sprint`: True if sprinting.
    /// - `jump`: True if jump requested on this tick.
    /// - `crouch`: True if crouch requested.
    /// - `ground_probe`: Terrain or mesh ground probe returning elevation `y` and normal `[nx, ny, nz]` at `(x, z)`.
    pub fn step<F>(
        &mut self,
        wish_dir: [f32; 2],
        sprint: bool,
        jump: bool,
        crouch: bool,
        dt: f32,
        ground_probe: F,
    ) where
        F: Fn(f32, f32) -> (f32, [f32; 3]),
    {
        self.state.is_crouching = crouch;
        let target_speed = if crouch {
            self.config.walk_speed * 0.5
        } else if sprint {
            self.config.run_speed
        } else {
            self.config.walk_speed
        };

        // 1. Transform wish direction from character yaw coordinates to world space
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

        // 2. Horizontal velocity acceleration / deceleration
        let accel = if self.state.is_grounded {
            if wish_len > 0.01 {
                self.config.ground_accel
            } else {
                self.config.ground_decel
            }
        } else {
            self.config.air_accel
        };

        let target_vx = world_wish[0] * target_speed;
        let target_vz = world_wish[2] * target_speed;

        self.state.velocity[0] = lerp_accel(self.state.velocity[0], target_vx, accel, dt);
        self.state.velocity[2] = lerp_accel(self.state.velocity[2], target_vz, accel, dt);

        // 3. Jump and Gravity
        if self.state.is_grounded && jump {
            self.state.velocity[1] = self.config.jump_impulse;
            self.state.is_grounded = false;
        } else if !self.state.is_grounded {
            self.state.velocity[1] += self.config.gravity * dt;
            if self.state.velocity[1] < self.config.terminal_velocity {
                self.state.velocity[1] = self.config.terminal_velocity;
            }
        }

        // 4. Integrate tentative position
        let next_x = self.state.position[0]
            + (self.state.velocity[0] + self.state.platform_velocity[0]) * dt;
        let mut next_y = self.state.position[1] + self.state.velocity[1] * dt;
        let next_z = self.state.position[2]
            + (self.state.velocity[2] + self.state.platform_velocity[2]) * dt;

        // 5. Query ground contact at tentative position
        let (ground_y, ground_norm) = ground_probe(next_x, next_z);
        self.state.ground_normal = ground_norm;

        // Check slope angle relative to vertical up [0, 1, 0]
        let cos_slope = ground_norm[1].clamp(0.0, 1.0);
        let slope_angle = cos_slope.acos();
        let is_traversable = slope_angle <= self.config.max_slope_angle;

        // Ground collision & step-climbing resolution
        if next_y <= ground_y {
            if is_traversable {
                next_y = ground_y;
                self.state.velocity[1] = 0.0;
                self.state.is_grounded = true;
            } else {
                // Slope sliding: slide downward along the slope normal projection
                let slide_x = ground_norm[0] * (1.0 - ground_norm[1]);
                let slide_z = ground_norm[2] * (1.0 - ground_norm[1]);
                self.state.velocity[0] += slide_x * 9.81 * dt;
                self.state.velocity[2] += slide_z * 9.81 * dt;
                next_y = ground_y;
                self.state.is_grounded = false;
            }
        } else if next_y > ground_y + 0.05 {
            self.state.is_grounded = false;
        }

        // Stride tracking for audio / animations
        if self.state.is_grounded {
            let horiz_dist = (self.state.velocity[0] * self.state.velocity[0]
                + self.state.velocity[2] * self.state.velocity[2])
                .sqrt()
                * dt;
            self.state.stride_distance += horiz_dist;
        }

        self.state.position[0] = next_x;
        self.state.position[1] = next_y;
        self.state.position[2] = next_z;
    }

    /// Computes eye position in world space (for camera placement).
    #[inline]
    pub fn eye_position(&self) -> [f32; 3] {
        let h = if self.state.is_crouching {
            self.config.crouch_height - 0.1
        } else {
            self.config.height - 0.12
        };
        [
            self.state.position[0],
            self.state.position[1] + h,
            self.state.position[2],
        ]
    }
}

#[inline]
pub(crate) fn lerp_accel(current: f32, target: f32, rate: f32, dt: f32) -> f32 {
    let diff = target - current;
    let step = rate * dt;
    if diff.abs() <= step {
        target
    } else {
        current + diff.signum() * step
    }
}
