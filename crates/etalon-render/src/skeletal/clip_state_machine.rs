//! Zero-Heap Clip State Machine, Layer Blending & Sim Blend Space (`clip_state_machine.rs`).

use super::transform_hierarchy::{
    quat_conjugate_xyzw, quat_mul_xyzw, quat_slerp, Transform3D, ZeroHeapSkeleton, MAX_BONES,
};

/// Animation state machine transition blend mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransitionMode {
    /// Instant cut to new animation.
    Instant,
    /// Smooth linear cross-fade over `duration` seconds.
    CrossFade { duration: f32, elapsed: f32 },
}

/// Zero-heap animation state machine controller.
#[derive(Debug, Clone, Copy)]
pub struct AnimationStateMachine {
    /// Current active animation index.
    pub current_state: u16,
    /// Target transition state index (if transitioning).
    pub target_state: Option<u16>,
    /// Playback time in seconds within current state.
    pub playback_time: f32,
    /// Playback speed multiplier (`1.0` = normal, `0.0` = paused).
    pub playback_speed: f32,
    /// Loop playback when reaching duration.
    pub is_looping: bool,
    /// Total duration of current clip in seconds.
    pub clip_duration: f32,
    /// Active transition mode.
    pub transition: TransitionMode,
}

impl AnimationStateMachine {
    /// Creates a new animation state machine starting in `initial_state`.
    pub fn new(initial_state: u16, clip_duration: f32) -> Self {
        Self {
            current_state: initial_state,
            target_state: None,
            playback_time: 0.0,
            playback_speed: 1.0,
            is_looping: true,
            clip_duration,
            transition: TransitionMode::Instant,
        }
    }

    /// Triggers a cross-fade transition to `next_state` over `duration` seconds.
    pub fn transition_to(&mut self, next_state: u16, duration: f32) {
        if self.current_state == next_state {
            return;
        }
        if duration <= 1e-4 {
            self.current_state = next_state;
            self.target_state = None;
            self.playback_time = 0.0;
            self.transition = TransitionMode::Instant;
        } else {
            self.target_state = Some(next_state);
            self.transition = TransitionMode::CrossFade {
                duration,
                elapsed: 0.0,
            };
        }
    }

    /// Advances the animation playback timer by `dt` seconds.
    pub fn step(&mut self, dt: f32) -> f32 {
        self.playback_time += dt * self.playback_speed;
        if self.is_looping && self.clip_duration > 0.0 {
            self.playback_time %= self.clip_duration;
        }

        match self.transition {
            TransitionMode::Instant => 0.0,
            TransitionMode::CrossFade {
                duration,
                ref mut elapsed,
            } => {
                *elapsed += dt;
                let alpha = (*elapsed / duration).clamp(0.0, 1.0);
                if alpha >= 1.0 {
                    if let Some(target) = self.target_state.take() {
                        self.current_state = target;
                    }
                    self.transition = TransitionMode::Instant;
                    0.0
                } else {
                    alpha
                }
            }
        }
    }
}

/// 64-bit bone mask (1 bit per bone in [`ZeroHeapSkeleton`]).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoneMask64(pub u64);

impl Default for BoneMask64 {
    fn default() -> Self {
        Self::ALL
    }
}

impl BoneMask64 {
    pub const ALL: Self = Self(u64::MAX);
    pub const NONE: Self = Self(0);

    /// Builds a subtree mask rooted at `root_bone` from a [`ZeroHeapSkeleton`].
    pub fn from_subtree(skel: &ZeroHeapSkeleton, root_bone: usize) -> Self {
        let n = skel.bone_count.min(MAX_BONES);
        if root_bone >= n {
            return Self::NONE;
        }
        let mut mask = 1u64 << root_bone;
        for _ in 0..n {
            let mut changed = false;
            for i in 0..n {
                if (mask & (1u64 << i)) != 0 {
                    continue;
                }
                let p = skel.parent_indices[i] as usize;
                if p < n && (mask & (1u64 << p)) != 0 {
                    mask |= 1u64 << i;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Self(mask)
    }

    #[inline]
    pub const fn contains(self, bone_idx: usize) -> bool {
        bone_idx < MAX_BONES && (self.0 & (1u64 << bone_idx)) != 0
    }
}

/// Multi-layer skeletal blend mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerBlendOp {
    Override,
    Additive,
}

/// Blends `layer_pose` into `base_pose` using `mask`, `op`, and `weight` in `[0.0, 1.0]` (`HotZeroHeap`).
pub fn blend_skeleton_layer_zero_heap(
    base_pose: &mut [Transform3D],
    rest_pose: &[Transform3D],
    layer_pose: &[Transform3D],
    mask: BoneMask64,
    op: LayerBlendOp,
    weight: f32,
) {
    let w = weight.clamp(0.0, 1.0);
    if w <= 1e-6 {
        return;
    }
    let count = base_pose
        .len()
        .min(rest_pose.len())
        .min(layer_pose.len())
        .min(MAX_BONES);
    for i in 0..count {
        if !mask.contains(i) {
            continue;
        }
        match op {
            LayerBlendOp::Override => {
                base_pose[i] = base_pose[i].lerp(&layer_pose[i], w);
            }
            LayerBlendOp::Additive => {
                let dt = [
                    (layer_pose[i].translation[0] - rest_pose[i].translation[0]) * w,
                    (layer_pose[i].translation[1] - rest_pose[i].translation[1]) * w,
                    (layer_pose[i].translation[2] - rest_pose[i].translation[2]) * w,
                ];
                let delta_q = quat_mul_xyzw(
                    quat_conjugate_xyzw(rest_pose[i].rotation),
                    layer_pose[i].rotation,
                );
                let weighted_q = quat_slerp([0.0, 0.0, 0.0, 1.0], delta_q, w);
                base_pose[i].translation = [
                    base_pose[i].translation[0] + dt[0],
                    base_pose[i].translation[1] + dt[1],
                    base_pose[i].translation[2] + dt[2],
                ];
                base_pose[i].rotation = quat_mul_xyzw(base_pose[i].rotation, weighted_q);
            }
        }
    }
}

/// 2D blend space sample point for `Transform3D` poses.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SimBlendPoint2D {
    pub pos: [f32; 2],
    pub clip_index: u16,
}

/// Zero-heap 2D triangle blend space for `Transform3D` skeletal poses.
#[derive(Debug, Clone, Copy)]
pub struct SimBlendSpace2D<const MAX_PTS: usize = 16, const MAX_TRIS: usize = 32> {
    pub points: [SimBlendPoint2D; MAX_PTS],
    pub point_count: usize,
    pub triangles: [[u8; 3]; MAX_TRIS],
    pub tri_count: usize,
}

impl<const MAX_PTS: usize, const MAX_TRIS: usize> Default for SimBlendSpace2D<MAX_PTS, MAX_TRIS> {
    fn default() -> Self {
        Self {
            points: [SimBlendPoint2D {
                pos: [0.0, 0.0],
                clip_index: 0,
            }; MAX_PTS],
            point_count: 0,
            triangles: [[0; 3]; MAX_TRIS],
            tri_count: 0,
        }
    }
}

impl<const MAX_PTS: usize, const MAX_TRIS: usize> SimBlendSpace2D<MAX_PTS, MAX_TRIS> {
    pub fn add_point(&mut self, x: f32, y: f32, clip_index: u16) -> bool {
        if self.point_count >= MAX_PTS {
            return false;
        }
        self.points[self.point_count] = SimBlendPoint2D {
            pos: [x, y],
            clip_index,
        };
        self.point_count += 1;
        true
    }

    pub fn add_triangle(&mut self, i0: u8, i1: u8, i2: u8) -> bool {
        if self.tri_count >= MAX_TRIS {
            return false;
        }
        self.triangles[self.tri_count] = [i0, i1, i2];
        self.tri_count += 1;
        true
    }

    /// Evaluates barycentric weights `([clip0, clip1, clip2], [w0, w1, w2])` for `query` (`HotZeroHeap`).
    pub fn sample_weights(&self, query: [f32; 2]) -> ([u16; 3], [f32; 3]) {
        if self.point_count == 0 {
            return ([0, 0, 0], [1.0, 0.0, 0.0]);
        }
        for t in 0..self.tri_count {
            let [i0, i1, i2] = self.triangles[t];
            let p0 = self.points[i0 as usize];
            let p1 = self.points[i1 as usize];
            let p2 = self.points[i2 as usize];
            let area = (p1.pos[0] - p0.pos[0]) * (p2.pos[1] - p0.pos[1])
                - (p1.pos[1] - p0.pos[1]) * (p2.pos[0] - p0.pos[0]);
            if area.abs() <= 1e-7 {
                continue;
            }
            let inv = 1.0 / area;
            let w0 = ((p1.pos[0] - query[0]) * (p2.pos[1] - query[1])
                - (p1.pos[1] - query[1]) * (p2.pos[0] - query[0]))
                * inv;
            let w1 = ((query[0] - p0.pos[0]) * (p2.pos[1] - p0.pos[1])
                - (query[1] - p0.pos[1]) * (p2.pos[0] - p0.pos[0]))
                * inv;
            let w2 = 1.0 - w0 - w1;
            if w0 >= -1e-4 && w1 >= -1e-4 && w2 >= -1e-4 {
                let cw0 = w0.max(0.0);
                let cw1 = w1.max(0.0);
                let cw2 = w2.max(0.0);
                let s = (cw0 + cw1 + cw2).max(1e-6);
                return (
                    [p0.clip_index, p1.clip_index, p2.clip_index],
                    [cw0 / s, cw1 / s, cw2 / s],
                );
            }
        }
        let c = self.points[0].clip_index;
        ([c, c, c], [1.0, 0.0, 0.0])
    }
}

/// Samples a [`SimBlendSpace2D`] at `query` and blends the 3 enclosing clip `Transform3D` poses (`HotZeroHeap`).
pub fn sample_pose_2d_zero_heap<const MAX_PTS: usize, const MAX_TRIS: usize>(
    space: &SimBlendSpace2D<MAX_PTS, MAX_TRIS>,
    query: [f32; 2],
    clip_poses: &[&[Transform3D]],
    out_pose: &mut [Transform3D],
) -> usize {
    let (clips, [w0, w1, w2]) = space.sample_weights(query);
    let p0 = clip_poses[clips[0] as usize];
    let p1 = clip_poses[clips[1] as usize];
    let p2 = clip_poses[clips[2] as usize];
    let n = p0.len().min(p1.len()).min(p2.len()).min(out_pose.len());
    let w01 = w0 + w1;
    let t01 = if w01 > 1e-6 { w1 / w01 } else { 0.0 };
    for i in 0..n {
        let a = p0[i].lerp(&p1[i], t01);
        out_pose[i] = a.lerp(&p2[i], w2);
    }
    n
}

/// Root motion displacement extracted from bone 0.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SimRootMotionDelta {
    pub delta_translation: [f32; 3],
    pub delta_rotation: [f32; 4],
}

/// Loop-safe root motion extractor for `Transform3D` skeletons.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimRootMotionExtractor {
    pub prev_translation: [f32; 3],
    pub loop_end_translation: [f32; 3],
    pub initialized: bool,
}

impl SimRootMotionExtractor {
    pub fn extract_and_lock(
        &mut self,
        root_transform: &mut Transform3D,
        looped: bool,
    ) -> SimRootMotionDelta {
        let curr = root_transform.translation;
        if !self.initialized {
            self.prev_translation = curr;
            self.initialized = true;
            root_transform.translation[0] = 0.0;
            root_transform.translation[2] = 0.0;
            return SimRootMotionDelta {
                delta_translation: [0.0; 3],
                delta_rotation: [0.0, 0.0, 0.0, 1.0],
            };
        }
        let dt = if !looped {
            [
                curr[0] - self.prev_translation[0],
                curr[1] - self.prev_translation[1],
                curr[2] - self.prev_translation[2],
            ]
        } else {
            [
                (self.loop_end_translation[0] - self.prev_translation[0]) + curr[0],
                (self.loop_end_translation[1] - self.prev_translation[1]) + curr[1],
                (self.loop_end_translation[2] - self.prev_translation[2]) + curr[2],
            ]
        };
        self.prev_translation = curr;
        root_transform.translation[0] = 0.0;
        root_transform.translation[2] = 0.0;
        SimRootMotionDelta {
            delta_translation: dt,
            delta_rotation: root_transform.rotation,
        }
    }
}

/// Fixed-capacity animation signal event and ring buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SimAnimationSignal {
    pub signal_id: u32,
    pub normalized_time: f32,
    pub payload: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct SimSignalQueue<const CAP: usize = 16> {
    buf: [SimAnimationSignal; CAP],
    head: usize,
    tail: usize,
    len: usize,
}

impl<const CAP: usize> Default for SimSignalQueue<CAP> {
    fn default() -> Self {
        Self {
            buf: [SimAnimationSignal::default(); CAP],
            head: 0,
            tail: 0,
            len: 0,
        }
    }
}

impl<const CAP: usize> SimSignalQueue<CAP> {
    pub fn push(&mut self, sig: SimAnimationSignal) -> bool {
        if self.len >= CAP || CAP == 0 {
            return false;
        }
        self.buf[self.tail] = sig;
        self.tail = (self.tail + 1) % CAP;
        self.len += 1;
        true
    }

    pub fn pop(&mut self) -> Option<SimAnimationSignal> {
        if self.len == 0 || CAP == 0 {
            return None;
        }
        let s = self.buf[self.head];
        self.head = (self.head + 1) % CAP;
        self.len -= 1;
        Some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeletal::gpu_skin::{gpu_shader_skin_vertex_reference, GpuJointMotorPair, GpuSkinnedVertex};

    #[test]
    fn test_skeleton_forward_kinematics_and_dual_quaternion_palette_uplift() {
        let mut skel = ZeroHeapSkeleton::default();
        skel.bone_count = 2;
        skel.parent_indices[0] = 255;
        skel.local_transforms[0].translation = [0.0, 1.0, 0.0];
        skel.parent_indices[1] = 0;
        skel.local_transforms[1].translation = [0.0, 0.5, 0.0];

        skel.evaluate_global_poses();
        let m1 = &skel.global_matrices[1];
        assert!((m1[13] - 1.5).abs() < 1e-5);

        // Verify Transform3D <-> Motor f64/f32 uplift
        let m64 = skel.local_transforms[1].to_motor_f64();
        let rt = Transform3D::from_motor_f64(&m64);
        assert!((rt.translation[1] - 0.5).abs() < 1e-5);
        let pga32 = skel.local_transforms[1].to_pga_motor_f32();
        let rt32 = Transform3D::from_pga_motor_f32(pga32);
        assert!((rt32.translation[1] - 0.5).abs() < 1e-5);

        // Verify ZeroHeapSkeleton::export_gpu_dual_quaternion_palette uplift
        let mut dq_palette = [GpuJointMotorPair::identity(); 2];
        let exported = skel.export_gpu_dual_quaternion_palette(&mut dq_palette);
        assert_eq!(exported, 2);
        let attr = GpuSkinnedVertex {
            joint_indices: [1, 0, 0, 0],
            joint_weights: [1.0, 0.0, 0.0, 0.0],
        };
        let (skinned_p, _, _, _) = gpu_shader_skin_vertex_reference(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            &attr,
            &dq_palette,
        );
        assert!((skinned_p[1] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_animation_cross_fade_and_multi_layer_blend() {
        let mut asm = AnimationStateMachine::new(0, 1.0);
        asm.transition_to(1, 0.5);
        assert!((asm.step(0.25) - 0.5).abs() < 1e-4);
        assert_eq!(asm.step(0.25), 0.0);
        assert_eq!(asm.current_state, 1);

        let mut skel = ZeroHeapSkeleton::default();
        skel.bone_count = 3;
        skel.parent_indices[0] = 255;
        skel.parent_indices[1] = 0;
        skel.parent_indices[2] = 1;
        let upper_mask = BoneMask64::from_subtree(&skel, 1);
        assert!(!upper_mask.contains(0));
        assert!(upper_mask.contains(2));
    }
}
