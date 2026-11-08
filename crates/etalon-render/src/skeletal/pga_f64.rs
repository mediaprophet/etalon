//! 3D Projective Geometric Algebra (PGA $\mathcal{G}_{3,0,1}$) / Dual Quaternion Motor Algebra (`f64`).
//!
//! Represents rigid 3D spatial transformations (rotation + translation) as motors $M = R + \varepsilon D$.
//! Enables constant-speed Screw Linear Interpolation (ScLERP) along Riemannian geodesics
//! in $SE(3)$ without singularity, gimbal lock, or heap allocation.
//! Includes lossless/bounded conversion bridges to/from `crate::spatial_math::pga::Motor` (`f32`).

/// A 3D PGA Motor (even multivector / unit dual quaternion) in `f64` precision.
/// Real part $R = (r_w, r_x, r_y, r_z)$, Dual part $D = (d_w, d_x, d_y, d_z)$.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Motor {
    pub r_w: f64,
    pub r_x: f64,
    pub r_y: f64,
    pub r_z: f64,
    pub d_w: f64,
    pub d_x: f64,
    pub d_y: f64,
    pub d_z: f64,
}

impl Default for Motor {
    fn default() -> Self {
        Self::identity()
    }
}

/// Lie algebra $\mathfrak{se}(3)$ screw bivector (rotation axis/angle + translation along screw axis).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[repr(C)]
pub struct MotorBivector {
    pub rx: f64,
    pub ry: f64,
    pub rz: f64,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

impl Motor {
    /// Identity motor (no rotation, zero translation).
    pub const fn identity() -> Self {
        Self {
            r_w: 1.0,
            r_x: 0.0,
            r_y: 0.0,
            r_z: 0.0,
            d_w: 0.0,
            d_x: 0.0,
            d_y: 0.0,
            d_z: 0.0,
        }
    }

    /// Construct a motor from a rotation quaternion `[w, x, y, z]` and translation `[x, y, z]`.
    pub fn from_rotation_translation(rot: [f64; 4], trans: [f64; 3]) -> Self {
        let (rw, rx, ry, rz) = (rot[0], rot[1], rot[2], rot[3]);
        let (tx, ty, tz) = (trans[0], trans[1], trans[2]);

        // Dual part D = 0.5 * T * R
        let dw = -0.5 * (tx * rx + ty * ry + tz * rz);
        let dx = 0.5 * (tx * rw + ty * rz - tz * ry);
        let dy = 0.5 * (-tx * rz + ty * rw + tz * rx);
        let dz = 0.5 * (tx * ry - ty * rx + tz * rw);

        Self {
            r_w: rw,
            r_x: rx,
            r_y: ry,
            r_z: rz,
            d_w: dw,
            d_x: dx,
            d_y: dy,
            d_z: dz,
        }
        .normalize()
    }

    /// Construct a pure translation motor.
    pub fn from_translation(tx: f64, ty: f64, tz: f64) -> Self {
        Self::from_rotation_translation([1.0, 0.0, 0.0, 0.0], [tx, ty, tz])
    }

    /// Converts to `(rotation [w, x, y, z], translation [x, y, z])`.
    pub fn to_rotation_translation(&self) -> ([f64; 4], [f64; 3]) {
        let rot = [self.r_w, self.r_x, self.r_y, self.r_z];
        // Translation T = 2 * D * R^(-1)
        let tx = 2.0
            * (self.d_x * self.r_w - self.d_w * self.r_x + self.d_z * self.r_y
                - self.d_y * self.r_z);
        let ty = 2.0
            * (self.d_y * self.r_w - self.d_z * self.r_x - self.d_w * self.r_y
                + self.d_x * self.r_z);
        let tz = 2.0
            * (self.d_z * self.r_w + self.d_y * self.r_x
                - self.d_x * self.r_y
                - self.d_w * self.r_z);
        let trans = [tx, ty, tz];
        (rot, trans)
    }

    /// Converts this `f64` dual-quaternion `Motor` (`[w, x, y, z]` order) into an `f32`
    /// `crate::spatial_math::pga::Motor` (`blade4` `(s, e12, e13, e23)` order).
    #[inline]
    pub fn to_pga_f32(&self) -> crate::spatial_math::pga::Motor {
        let qr = [
            self.r_w as f32,
            self.r_x as f32,
            self.r_y as f32,
            self.r_z as f32,
        ];
        let qd = [
            self.d_w as f32,
            self.d_x as f32,
            self.d_y as f32,
            self.d_z as f32,
        ];
        crate::spatial_math::pga::Motor {
            r: crate::spatial_math::pga::quat_to_blade4(qr),
            d: crate::spatial_math::pga::quat_to_blade4(qd),
        }
    }

    /// Constructs an `f64` dual-quaternion `Motor` from an `f32` `crate::spatial_math::pga::Motor`.
    #[inline]
    pub fn from_pga_f32(m: crate::spatial_math::pga::Motor) -> Self {
        let qr = crate::spatial_math::pga::blade4_to_quat(m.r);
        let qd = crate::spatial_math::pga::blade4_to_quat(m.d);
        Self {
            r_w: qr[0] as f64,
            r_x: qr[1] as f64,
            r_y: qr[2] as f64,
            r_z: qr[3] as f64,
            d_w: qd[0] as f64,
            d_x: qd[1] as f64,
            d_y: qd[2] as f64,
            d_z: qd[3] as f64,
        }
        .normalize()
    }

    /// Geometric product (motor multiplication: $M_a M_b$).
    pub fn mul(&self, other: &Self) -> Self {
        let rw = self.r_w * other.r_w
            - self.r_x * other.r_x
            - self.r_y * other.r_y
            - self.r_z * other.r_z;
        let rx = self.r_w * other.r_x + self.r_x * other.r_w + self.r_y * other.r_z
            - self.r_z * other.r_y;
        let ry = self.r_w * other.r_y - self.r_x * other.r_z
            + self.r_y * other.r_w
            + self.r_z * other.r_x;
        let rz = self.r_w * other.r_z + self.r_x * other.r_y - self.r_y * other.r_x
            + self.r_z * other.r_w;

        let dw = self.r_w * other.d_w
            - self.r_x * other.d_x
            - self.r_y * other.d_y
            - self.r_z * other.d_z
            + self.d_w * other.r_w
            - self.d_x * other.r_x
            - self.d_y * other.r_y
            - self.d_z * other.r_z;

        let dx = self.r_w * other.d_x + self.r_x * other.d_w + self.r_y * other.d_z
            - self.r_z * other.d_y
            + self.d_w * other.r_x
            + self.d_x * other.r_w
            + self.d_y * other.r_z
            - self.d_z * other.r_y;

        let dy = self.r_w * other.d_y - self.r_x * other.d_z
            + self.r_y * other.d_w
            + self.r_z * other.d_x
            + self.d_w * other.r_y
            - self.d_x * other.r_z
            + self.d_y * other.r_w
            + self.d_z * other.r_x;

        let dz = self.r_w * other.d_z + self.r_x * other.d_y - self.r_y * other.d_x
            + self.r_z * other.d_w
            + self.d_w * other.r_z
            + self.d_x * other.r_y
            - self.d_y * other.r_x
            + self.d_z * other.r_w;

        Self {
            r_w: rw,
            r_x: rx,
            r_y: ry,
            r_z: rz,
            d_w: dw,
            d_x: dx,
            d_y: dy,
            d_z: dz,
        }
    }

    /// Conjugate of motor ($R^*, D^*$).
    pub fn conjugate(&self) -> Self {
        Self {
            r_w: self.r_w,
            r_x: -self.r_x,
            r_y: -self.r_y,
            r_z: -self.r_z,
            d_w: self.d_w,
            d_x: -self.d_x,
            d_y: -self.d_y,
            d_z: -self.d_z,
        }
    }

    /// Normalize motor so that $|R| = 1$ and $R \cdot D = 0$.
    pub fn normalize(&self) -> Self {
        let mag_r_sq =
            self.r_w * self.r_w + self.r_x * self.r_x + self.r_y * self.r_y + self.r_z * self.r_z;
        if mag_r_sq < 1e-12 {
            return Self::identity();
        }
        let inv_mag_r = 1.0 / mag_r_sq.sqrt();
        let dot_rd =
            self.r_w * self.d_w + self.r_x * self.d_x + self.r_y * self.d_y + self.r_z * self.d_z;

        let rw = self.r_w * inv_mag_r;
        let rx = self.r_x * inv_mag_r;
        let ry = self.r_y * inv_mag_r;
        let rz = self.r_z * inv_mag_r;

        let dw = (self.d_w - rw * dot_rd) * inv_mag_r;
        let dx = (self.d_x - rx * dot_rd) * inv_mag_r;
        let dy = (self.d_y - ry * dot_rd) * inv_mag_r;
        let dz = (self.d_z - rz * dot_rd) * inv_mag_r;

        Self {
            r_w: rw,
            r_x: rx,
            r_y: ry,
            r_z: rz,
            d_w: dw,
            d_x: dx,
            d_y: dy,
            d_z: dz,
        }
    }

    /// Logarithmic map: $\log(M) \in \mathfrak{se}(3)$.
    pub fn log(&self) -> MotorBivector {
        let m = if self.r_w < 0.0 {
            Self {
                r_w: -self.r_w,
                r_x: -self.r_x,
                r_y: -self.r_y,
                r_z: -self.r_z,
                d_w: -self.d_w,
                d_x: -self.d_x,
                d_y: -self.d_y,
                d_z: -self.d_z,
            }
        } else {
            *self
        };

        let sin_sq = m.r_x * m.r_x + m.r_y * m.r_y + m.r_z * m.r_z;
        if sin_sq < 1e-12 {
            return MotorBivector {
                rx: 0.0,
                ry: 0.0,
                rz: 0.0,
                dx: m.d_x,
                dy: m.d_y,
                dz: m.d_z,
            };
        }

        let sin_theta = sin_sq.sqrt();
        let theta = sin_theta.atan2(m.r_w);
        let inv_sin = 1.0 / sin_theta;

        let axis_x = m.r_x * inv_sin;
        let axis_y = m.r_y * inv_sin;
        let axis_z = m.r_z * inv_sin;

        let pitch = -m.d_w * inv_sin;

        let rx = theta * axis_x;
        let ry = theta * axis_y;
        let rz = theta * axis_z;

        let dx = m.d_x - pitch * m.r_x;
        let dy = m.d_y - pitch * m.r_y;
        let dz = m.d_z - pitch * m.r_z;

        MotorBivector {
            rx,
            ry,
            rz,
            dx,
            dy,
            dz,
        }
    }

    /// Screw Linear Interpolation (ScLERP): Geodesic shortest-path $SE(3)$ interpolation.
    /// Returns $M(t) = M_0 (M_0^{-1} M_1)^t$.
    pub fn sclerp(m0: &Self, m1: &Self, t: f64) -> Self {
        if t <= 0.0 {
            return *m0;
        }
        if t >= 1.0 {
            return *m1;
        }

        let m0_inv = m0.conjugate();
        let mut delta = m0_inv.mul(m1);

        if delta.r_w < 0.0 {
            delta = Self {
                r_w: -delta.r_w,
                r_x: -delta.r_x,
                r_y: -delta.r_y,
                r_z: -delta.r_z,
                d_w: -delta.d_w,
                d_x: -delta.d_x,
                d_y: -delta.d_y,
                d_z: -delta.d_z,
            };
        }

        let bivector = delta.log();
        let scaled_bivector = MotorBivector {
            rx: bivector.rx * t,
            ry: bivector.ry * t,
            rz: bivector.rz * t,
            dx: bivector.dx * t,
            dy: bivector.dy * t,
            dz: bivector.dz * t,
        };

        let delta_t = scaled_bivector.exp();
        m0.mul(&delta_t).normalize()
    }

    /// Transform a 3D point using the motor: $p' = M p \widetilde{M}$.
    pub fn transform_point(&self, p: [f64; 3]) -> [f64; 3] {
        let (rot, trans) = self.to_rotation_translation();
        let (rw, rx, ry, rz) = (rot[0], rot[1], rot[2], rot[3]);
        let (px, py, pz) = (p[0], p[1], p[2]);

        let ix = rw * px + ry * pz - rz * py;
        let iy = rw * py - rx * pz + rz * px;
        let iz = rw * pz + rx * py - ry * px;
        let iw = -rx * px - ry * py - rz * pz;

        let rx_out = ix * rw + iw * -rx + iy * -rz - iz * -ry;
        let ry_out = iy * rw + iw * -ry + iz * -rx - ix * -rz;
        let rz_out = iz * rw + iw * -rz + ix * -ry - iy * -rx;

        [rx_out + trans[0], ry_out + trans[1], rz_out + trans[2]]
    }
}

impl MotorBivector {
    /// Exponential map: $\exp(B) \in SE(3)$.
    pub fn exp(&self) -> Motor {
        let theta_sq = self.rx * self.rx + self.ry * self.ry + self.rz * self.rz;
        if theta_sq < 1e-12 {
            return Motor {
                r_w: 1.0,
                r_x: 0.0,
                r_y: 0.0,
                r_z: 0.0,
                d_w: 0.0,
                d_x: self.dx,
                d_y: self.dy,
                d_z: self.dz,
            };
        }

        let theta = theta_sq.sqrt();
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let inv_theta = 1.0 / theta;

        let rw = cos_theta;
        let rx = self.rx * inv_theta * sin_theta;
        let ry = self.ry * inv_theta * sin_theta;
        let rz = self.rz * inv_theta * sin_theta;

        Motor {
            r_w: rw,
            r_x: rx,
            r_y: ry,
            r_z: rz,
            d_w: 0.0,
            d_x: self.dx,
            d_y: self.dy,
            d_z: self.dz,
        }
        .normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motor_identity_transforms_point_invariant() {
        let m = Motor::identity();
        let p = [10.0, -5.0, 3.5];
        let p_out = m.transform_point(p);
        for i in 0..3 {
            assert!((p[i] - p_out[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn motor_pure_translation_and_sclerp() {
        let m0 = Motor::identity();
        let m1 = Motor::from_rotation_translation([1.0, 0.0, 0.0, 0.0], [10.0, 20.0, 30.0]);
        let mid = Motor::sclerp(&m0, &m1, 0.5);
        let (_, trans) = mid.to_rotation_translation();
        assert!((trans[0] - 5.0).abs() < 1e-4);
        assert!((trans[1] - 10.0).abs() < 1e-4);
        assert!((trans[2] - 15.0).abs() < 1e-4);
    }

    #[test]
    fn motor_f64_to_pga_f32_roundtrip_matches_sandwich_point() {
        let half = std::f64::consts::FRAC_PI_4;
        let m64 =
            Motor::from_rotation_translation([half.cos(), 0.0, half.sin(), 0.0], [2.0, -1.5, 4.0]);
        let m32 = m64.to_pga_f32();
        let p64 = [1.0, 2.0, -0.5];
        let out64 = m64.transform_point(p64);
        let out32 = crate::spatial_math::pga::sandwich_point(m32, [1.0, 2.0, -0.5]);
        for i in 0..3 {
            assert!((out64[i] as f32 - out32[i]).abs() < 1e-4);
        }
        let roundtrip = Motor::from_pga_f32(m32);
        let out_rt = roundtrip.transform_point(p64);
        for i in 0..3 {
            assert!((out64[i] - out_rt[i]).abs() < 1e-4);
        }
    }
}
