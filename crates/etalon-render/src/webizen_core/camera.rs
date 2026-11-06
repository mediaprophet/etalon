//! World-to-screen projection camera and zero-heap spherical orbit/zoom/pan controls.

/// 3D vector for world-space coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    /// Creates a new 3D vector.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// Screen-space point (2D coordinates in physical pixels).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenPoint {
    pub x: f64,
    pub y: f64,
}

impl ScreenPoint {
    /// Creates a new 2D screen-space point.
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Camera for world-to-screen projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 5.0,
            },
            target: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            up: Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            fov: 60.0,
        }
    }
}

impl Camera {
    /// Creates a camera with explicit position, target, up vector, and field-of-view (degrees).
    #[inline]
    pub const fn new(position: Vec3, target: Vec3, up: Vec3, fov: f64) -> Self {
        Self {
            position,
            target,
            up,
            fov,
        }
    }

    /// Resets the camera to its canonical default state.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Project a world-space point to screen space.
    pub fn project(&self, world: Vec3, viewport: (f64, f64)) -> Option<ScreenPoint> {
        let (width, height) = viewport;
        let aspect_ratio = width / height;

        // Transform to camera space
        let dx = world.x - self.position.x;
        let dy = world.y - self.position.y;
        let dz = world.z - self.position.z;

        // Simple perspective divide
        if dz <= 0.1 {
            return None; // Behind camera
        }

        let fov_rad = self.fov * std::f64::consts::PI / 180.0;
        let scale = 1.0 / (dz * (fov_rad / 2.0).tan());

        let screen_x = (dx * scale * aspect_ratio + 1.0) * width / 2.0;
        let screen_y = (-dy * scale + 1.0) * height / 2.0;

        Some(ScreenPoint {
            x: screen_x,
            y: screen_y,
        })
    }

    /// Alias for [`Self::project`].
    #[inline]
    pub fn project_point(&self, world: Vec3, viewport: (f64, f64)) -> Option<ScreenPoint> {
        self.project(world, viewport)
    }

    /// Rotate projection viewpoint around target using yaw (horizontal) and pitch (vertical) angles.
    /// This adjusts the geometric examination viewpoint of the 10D epistemic manifold.
    /// All calculations use stack-allocated f64 values for zero-heap compliance.
    pub fn orbit(&mut self, yaw: f64, pitch: f64) {
        // Calculate vector from target to eye (camera radius)
        let dx = self.position.x - self.target.x;
        let dy = self.position.y - self.target.y;
        let dz = self.position.z - self.target.z;

        // Current spherical coordinates
        let radius = (dx * dx + dy * dy + dz * dz).sqrt();
        let current_yaw = dz.atan2(dx);
        let current_pitch = (dy / radius).asin();

        // Apply rotation with pitch clamping to avoid gimbal lock
        let new_yaw = current_yaw + yaw;
        let new_pitch = (current_pitch + pitch).clamp(
            -std::f64::consts::FRAC_PI_2 + 0.01,
            std::f64::consts::FRAC_PI_2 - 0.01,
        );

        // Convert back to Cartesian coordinates (stack-allocated)
        let cos_pitch = new_pitch.cos();
        let sin_pitch = new_pitch.sin();
        let cos_yaw = new_yaw.cos();
        let sin_yaw = new_yaw.sin();

        // Update camera position
        self.position.x = self.target.x + radius * cos_pitch * cos_yaw;
        self.position.y = self.target.y + radius * sin_pitch;
        self.position.z = self.target.z + radius * cos_pitch * sin_yaw;
    }

    /// Zoom camera in/out by adjusting distance to target.
    /// Uses stack-allocated f64 values for zero-heap compliance.
    pub fn zoom(&mut self, delta: f64) {
        // Calculate current distance to target
        let dx = self.position.x - self.target.x;
        let dy = self.position.y - self.target.y;
        let dz = self.position.z - self.target.z;
        let current_distance = (dx * dx + dy * dy + dz * dz).sqrt();

        // Apply zoom with minimum distance clamp
        let zoom_factor = (-delta * 0.1).exp(); // Smooth exponential zoom
        let new_distance = (current_distance * zoom_factor).max(0.5);

        // Scale the position vector
        let scale = new_distance / current_distance;
        self.position.x = self.target.x + dx * scale;
        self.position.y = self.target.y + dy * scale;
        self.position.z = self.target.z + dz * scale;
    }

    /// Pan camera in screen space (dx, dy are screen-relative deltas).
    /// Uses stack-allocated f64 values for zero-heap compliance.
    pub fn pan(&mut self, dx: f64, dy: f64) {
        // Calculate camera forward and right vectors
        let forward_x = self.target.x - self.position.x;
        let forward_y = self.target.y - self.position.y;
        let forward_z = self.target.z - self.position.z;
        let forward_len =
            (forward_x * forward_x + forward_y * forward_y + forward_z * forward_z).sqrt();

        // Normalize forward vector
        let forward_x = forward_x / forward_len;
        let forward_y = forward_y / forward_len;
        let forward_z = forward_z / forward_len;

        // Calculate right vector (cross product of forward and up)
        let right_x = forward_y * self.up.z - forward_z * self.up.y;
        let right_y = forward_z * self.up.x - forward_x * self.up.z;
        let right_z = forward_x * self.up.y - forward_y * self.up.x;
        let right_len = (right_x * right_x + right_y * right_y + right_z * right_z).sqrt();

        // Normalize right vector
        let right_x = right_x / right_len;
        let right_y = right_y / right_len;
        let right_z = right_z / right_len;

        // Calculate up vector (cross product of right and forward)
        let up_x = right_y * forward_z - right_z * forward_y;
        let up_y = right_z * forward_x - right_x * forward_z;
        let up_z = right_x * forward_y - right_y * forward_x;

        // Scale pan by distance to target for consistent feel
        let pan_scale = forward_len * 0.001;

        // Apply pan to both position and target
        let pan_x = (right_x * dx + up_x * dy) * pan_scale;
        let pan_y = (right_y * dx + up_y * dy) * pan_scale;
        let pan_z = (right_z * dx + up_z * dy) * pan_scale;

        self.position.x += pan_x;
        self.position.y += pan_y;
        self.position.z += pan_z;

        self.target.x += pan_x;
        self.target.y += pan_y;
        self.target.z += pan_z;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_projects_and_rejects_points_behind_near_plane() {
        let cam = Camera::default();
        assert!(cam.project(Vec3::new(0.0, 0.0, 5.0), (800.0, 600.0)).is_none());
        let pt = cam
            .project_point(Vec3::new(0.0, 0.0, 10.0), (800.0, 600.0))
            .expect("point in front of camera projects");
        assert!((pt.x - 400.0).abs() < 1e-6);
        assert!((pt.y - 300.0).abs() < 1e-6);
    }

    #[test]
    fn camera_orbit_zoom_pan_and_reset_maintain_invariants() {
        let mut cam = Camera::default();
        cam.orbit(0.5, 0.25);
        let r = (cam.position.x.powi(2) + cam.position.y.powi(2) + cam.position.z.powi(2)).sqrt();
        assert!((r - 5.0).abs() < 1e-6);
        cam.zoom(2.0);
        let r_zoomed =
            (cam.position.x.powi(2) + cam.position.y.powi(2) + cam.position.z.powi(2)).sqrt();
        assert!(r_zoomed < 5.0 && r_zoomed >= 0.5);
        cam.pan(10.0, -5.0);
        assert!(cam.target.x.is_finite() && cam.target.y.is_finite());
        cam.reset();
        assert_eq!(cam, Camera::default());
    }
}
