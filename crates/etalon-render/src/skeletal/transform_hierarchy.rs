//! Zero-Heap Skeletal Transform Hierarchy & Dual-Quaternion Bridge (`transform_hierarchy.rs`).
//!
//! Provides `Transform3D`, `ZeroHeapSkeleton`, `Keyframe3D`, `BoneTrack`, `AnimationEvent`,
//! and `AnimationClip`, uplifted with direct `Transform3D` ↔ PGA `Motor` (`f64`/`f32`)
//! conversion and hemisphere-aligned `GpuJointMotorPair` dual-quaternion palette export.

use super::gpu_skin::{hemisphere_align_motor, GpuJointMotorPair};
use super::Motor;

/// Maximum bones supported per skeleton instance (fits in GPU uniform blocks).
pub const MAX_BONES: usize = 64;

/// 3D affine transform representation (`rotation` stored as `[x, y, z, w]`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform3D {
    /// Translation `[x, y, z]` in meters.
    pub translation: [f32; 3],
    /// Unit quaternion rotation `[x, y, z, w]`.
    pub rotation: [f32; 4],
    /// Scale factor `[sx, sy, sz]`.
    pub scale: [f32; 3],
}

impl Default for Transform3D {
    fn default() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

impl Transform3D {
    /// Computes $4 \times 4$ column-major affine transformation matrix.
    pub fn to_mat4(&self) -> [f32; 16] {
        let [x, y, z, w] = self.rotation;
        let [sx, sy, sz] = self.scale;
        let [tx, ty, tz] = self.translation;

        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;
        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;

        [
            (1.0 - (yy + zz)) * sx,
            (xy + wz) * sx,
            (xz - wy) * sx,
            0.0,
            (xy - wz) * sy,
            (1.0 - (xx + zz)) * sy,
            (yz + wx) * sy,
            0.0,
            (xz + wy) * sz,
            (yz - wx) * sz,
            (1.0 - (xx + yy)) * sz,
            0.0,
            tx,
            ty,
            tz,
            1.0,
        ]
    }

    /// Spherical linear interpolation between two transforms.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let translation = [
            self.translation[0] + (other.translation[0] - self.translation[0]) * t,
            self.translation[1] + (other.translation[1] - self.translation[1]) * t,
            self.translation[2] + (other.translation[2] - self.translation[2]) * t,
        ];
        let scale = [
            self.scale[0] + (other.scale[0] - self.scale[0]) * t,
            self.scale[1] + (other.scale[1] - self.scale[1]) * t,
            self.scale[2] + (other.scale[2] - self.scale[2]) * t,
        ];
        let rotation = quat_slerp(self.rotation, other.rotation, t);

        Self {
            translation,
            rotation,
            scale,
        }
    }

    /// Converts the rigid rotation (`[x, y, z, w]`) and translation (`[x, y, z]`) of this
    /// `Transform3D` into an `f64` PGA dual-quaternion [`Motor`] (`[w, x, y, z]` rotor order).
    #[inline]
    pub fn to_motor_f64(&self) -> Motor {
        let [rx, ry, rz, rw] = self.rotation;
        let [tx, ty, tz] = self.translation;
        Motor::from_rotation_translation(
            [rw as f64, rx as f64, ry as f64, rz as f64],
            [tx as f64, ty as f64, tz as f64],
        )
    }

    /// Reconstructs a unit-scale `Transform3D` from an `f64` PGA dual-quaternion [`Motor`].
    #[inline]
    pub fn from_motor_f64(m: &Motor) -> Self {
        let (rot_wxyz, trans) = m.to_rotation_translation();
        Self {
            translation: [trans[0] as f32, trans[1] as f32, trans[2] as f32],
            rotation: [
                rot_wxyz[1] as f32,
                rot_wxyz[2] as f32,
                rot_wxyz[3] as f32,
                rot_wxyz[0] as f32,
            ],
            scale: [1.0, 1.0, 1.0],
        }
    }

    /// Converts this `Transform3D` directly into an `f32` `crate::spatial_math::pga::Motor`.
    #[inline]
    pub fn to_pga_motor_f32(&self) -> crate::spatial_math::pga::Motor {
        self.to_motor_f64().to_pga_f32()
    }

    /// Reconstructs a unit-scale `Transform3D` from an `f32` `crate::spatial_math::pga::Motor`.
    #[inline]
    pub fn from_pga_motor_f32(m: crate::spatial_math::pga::Motor) -> Self {
        Self::from_motor_f64(&Motor::from_pga_f32(m))
    }
}

/// Fixed-capacity skeletal rig (`MAX_BONES = 64`).
#[derive(Debug, Clone, Copy)]
pub struct ZeroHeapSkeleton {
    /// Number of active bones in this skeleton (`<= 64`).
    pub bone_count: usize,
    /// Parent bone indices. Root bones have `parent_index == 255`.
    pub parent_indices: [u8; MAX_BONES],
    /// Local rest/bind or animated transforms.
    pub local_transforms: [Transform3D; MAX_BONES],
    /// Computed global model-space $4 \times 4$ matrices for GPU upload.
    pub global_matrices: [[f32; 16]; MAX_BONES],
}

impl Default for ZeroHeapSkeleton {
    fn default() -> Self {
        Self {
            bone_count: 0,
            parent_indices: [255; MAX_BONES],
            local_transforms: [Transform3D::default(); MAX_BONES],
            global_matrices: [[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ]; MAX_BONES],
        }
    }
}

impl ZeroHeapSkeleton {
    /// Evaluates forward kinematics hierarchy from root to leaves, updating `global_matrices`.
    pub fn evaluate_global_poses(&mut self) {
        let count = self.bone_count.min(MAX_BONES);
        for i in 0..count {
            let local_m = self.local_transforms[i].to_mat4();
            let parent = self.parent_indices[i];
            if parent == 255 || parent as usize >= count {
                self.global_matrices[i] = local_m;
            } else {
                let parent_m = &self.global_matrices[parent as usize];
                self.global_matrices[i] = mat4_mul(parent_m, &local_m);
            }
        }
    }

    /// Evaluates model-space bone poses as PGA dual quaternions and exports a hemisphere-aligned
    /// [`GpuJointMotorPair`] palette into `out` for volume-preserving GPU dual-quaternion skinning
    /// (`HotZeroHeap`). Returns the number of bones exported.
    pub fn export_gpu_dual_quaternion_palette(&self, out: &mut [GpuJointMotorPair]) -> usize {
        let count = self.bone_count.min(MAX_BONES).min(out.len());
        if count == 0 {
            return 0;
        }
        let mut model_motors = [Motor::identity(); MAX_BONES];
        for i in 0..count {
            let local_motor = self.local_transforms[i].to_motor_f64();
            let parent = self.parent_indices[i];
            let raw_global = if parent == 255 || (parent as usize) >= count {
                local_motor
            } else {
                model_motors[parent as usize].mul(&local_motor).normalize()
            };
            let ref_motor = if parent != 255 && (parent as usize) < i {
                model_motors[parent as usize]
            } else {
                model_motors[0]
            };
            let aligned = if i == 0 {
                raw_global
            } else {
                hemisphere_align_motor(&ref_motor, &raw_global)
            };
            model_motors[i] = aligned;
            out[i] = GpuJointMotorPair::from_motors(&aligned, &aligned);
        }
        count
    }
}

/// Single keyframe sample for a [`BoneTrack`].
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Keyframe3D {
    pub time: f32,
    pub transform: Transform3D,
}

/// Fixed-capacity keyframe track for a single bone (`HotZeroHeap`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoneTrack<const MAX_KEYS_PER_BONE: usize = 16> {
    pub bone_index: u8,
    pub keyframes: [Keyframe3D; MAX_KEYS_PER_BONE],
    pub keyframe_count: usize,
}

impl<const MAX_KEYS_PER_BONE: usize> Default for BoneTrack<MAX_KEYS_PER_BONE> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<const MAX_KEYS_PER_BONE: usize> BoneTrack<MAX_KEYS_PER_BONE> {
    pub const fn new(bone_index: u8) -> Self {
        Self {
            bone_index,
            keyframes: [Keyframe3D {
                time: 0.0,
                transform: Transform3D {
                    translation: [0.0, 0.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
            }; MAX_KEYS_PER_BONE],
            keyframe_count: 0,
        }
    }

    pub fn add_keyframe(&mut self, time: f32, transform: Transform3D) -> bool {
        if self.keyframe_count >= MAX_KEYS_PER_BONE || !time.is_finite() {
            return false;
        }
        self.keyframes[self.keyframe_count] = Keyframe3D { time, transform };
        self.keyframe_count += 1;
        true
    }

    pub fn sample(&self, time: f32) -> Transform3D {
        if self.keyframe_count == 0 {
            return Transform3D::default();
        }
        if self.keyframe_count == 1 || time <= self.keyframes[0].time {
            return self.keyframes[0].transform;
        }
        let last = self.keyframe_count - 1;
        if time >= self.keyframes[last].time {
            return self.keyframes[last].transform;
        }
        for i in 0..last {
            let a = &self.keyframes[i];
            let b = &self.keyframes[i + 1];
            if time >= a.time && time <= b.time {
                let span = (b.time - a.time).max(1e-6);
                let t = ((time - a.time) / span).clamp(0.0, 1.0);
                return a.transform.lerp(&b.transform, t);
            }
        }
        self.keyframes[last].transform
    }
}

/// Discrete event marker embedded in an [`AnimationClip`] timeline.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AnimationEvent {
    pub event_id: u32,
    pub time: f32,
    pub payload: f32,
}

/// Fixed-capacity skeletal animation clip (`HotZeroHeap`).
#[derive(Debug, Clone, Copy)]
pub struct AnimationClip<const MAX_TRACKS: usize = 16, const MAX_EVENTS: usize = 8> {
    pub duration: f32,
    pub looping: bool,
    pub tracks: [BoneTrack; MAX_TRACKS],
    pub track_count: usize,
    pub events: [AnimationEvent; MAX_EVENTS],
    pub event_count: usize,
}

impl<const MAX_TRACKS: usize, const MAX_EVENTS: usize> Default
    for AnimationClip<MAX_TRACKS, MAX_EVENTS>
{
    fn default() -> Self {
        Self::new(1.0, true)
    }
}

impl<const MAX_TRACKS: usize, const MAX_EVENTS: usize> AnimationClip<MAX_TRACKS, MAX_EVENTS> {
    pub const fn new(duration: f32, looping: bool) -> Self {
        Self {
            duration,
            looping,
            tracks: [BoneTrack::new(0); MAX_TRACKS],
            track_count: 0,
            events: [AnimationEvent {
                event_id: 0,
                time: 0.0,
                payload: 0.0,
            }; MAX_EVENTS],
            event_count: 0,
        }
    }

    pub fn sample_into_pose(&self, time: f32, out_pose: &mut [Transform3D]) {
        let t = if self.looping && self.duration > 0.0 {
            time.rem_euclid(self.duration)
        } else {
            time.clamp(0.0, self.duration.max(0.0))
        };
        for i in 0..self.track_count.min(MAX_TRACKS) {
            let track = &self.tracks[i];
            let b = track.bone_index as usize;
            if b < out_pose.len() {
                out_pose[b] = track.sample(t);
            }
        }
    }
}

/// Slerp quaternion interpolation (`[x, y, z, w]` layout).
pub(crate) fn quat_slerp(q1: [f32; 4], mut q2: [f32; 4], t: f32) -> [f32; 4] {
    let mut dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3];

    if dot < 0.0 {
        dot = -dot;
        q2[0] = -q2[0];
        q2[1] = -q2[1];
        q2[2] = -q2[2];
        q2[3] = -q2[3];
    }

    if dot > 0.9995 {
        let res = [
            q1[0] + (q2[0] - q1[0]) * t,
            q1[1] + (q2[1] - q1[1]) * t,
            q1[2] + (q2[2] - q1[2]) * t,
            q1[3] + (q2[3] - q1[3]) * t,
        ];
        let len = (res[0] * res[0] + res[1] * res[1] + res[2] * res[2] + res[3] * res[3]).sqrt();
        return [res[0] / len, res[1] / len, res[2] / len, res[3] / len];
    }

    let theta_0 = dot.clamp(-1.0, 1.0).acos();
    let theta = theta_0 * t;
    let sin_theta = theta.sin();
    let sin_theta_0 = theta_0.sin();

    let s0 = (theta_0 - theta).cos() - dot * sin_theta / sin_theta_0;
    let s1 = sin_theta / sin_theta_0;

    [
        (s0 * q1[0]) + (s1 * q2[0]),
        (s0 * q1[1]) + (s1 * q2[1]),
        (s0 * q1[2]) + (s1 * q2[2]),
        (s0 * q1[3]) + (s1 * q2[3]),
    ]
}

#[inline]
pub(crate) fn quat_mul_xyzw(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

#[inline]
pub(crate) fn quat_conjugate_xyzw(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// $4 \times 4$ column-major matrix multiplication.
pub(crate) fn mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = a[row] * b[col * 4]
                + a[4 + row] * b[col * 4 + 1]
                + a[8 + row] * b[col * 4 + 2]
                + a[12 + row] * b[col * 4 + 3];
        }
    }
    out
}
