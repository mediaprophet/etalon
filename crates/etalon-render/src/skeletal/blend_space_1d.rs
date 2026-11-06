//! Zero-Heap 1D Animation Blend Space (`blend_space_1d.rs`).
//!
//! Provides sorted 1D interval location and geodesic `Motor::sclerp` blending
//! over skeletal `Motor` poses (`HotZeroHeap`).

use super::{Motor, SkeletalError};

/// A single sample point inside a [`BlendSpace1D`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BlendPoint1D {
    /// Scalar parameter coordinate (e.g., speed in m/s).
    pub param: f32,
    /// Clip index in the character's animation clip table.
    pub clip_index: u16,
    /// Relative playback rate multiplier (`1.0` = authored speed).
    pub playback_speed: f32,
}

/// Fixed-capacity 1D animation blend space (`MAX_POINTS` default `8`).
#[derive(Clone, Copy, Debug)]
pub struct BlendSpace1D<const MAX_POINTS: usize = 8> {
    pub points: [BlendPoint1D; MAX_POINTS],
    pub count: usize,
}

impl<const MAX_POINTS: usize> Default for BlendSpace1D<MAX_POINTS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_POINTS: usize> BlendSpace1D<MAX_POINTS> {
    /// Creates an empty 1D blend space.
    pub const fn new() -> Self {
        Self {
            points: [BlendPoint1D {
                param: 0.0,
                clip_index: 0,
                playback_speed: 1.0,
            }; MAX_POINTS],
            count: 0,
        }
    }

    /// Inserts a point sorted by `param`. Returns `false` if capacity is full or `param` is non-finite.
    pub fn add_point(&mut self, param: f32, clip_index: u16) -> bool {
        self.add_point_with_speed(param, clip_index, 1.0)
    }

    /// Inserts a point with explicit playback speed multiplier.
    pub fn add_point_with_speed(
        &mut self,
        param: f32,
        clip_index: u16,
        playback_speed: f32,
    ) -> bool {
        if self.count >= MAX_POINTS || !param.is_finite() || !playback_speed.is_finite() {
            return false;
        }
        let mut insert_at = self.count;
        for i in 0..self.count {
            if param < self.points[i].param {
                insert_at = i;
                break;
            }
        }
        let mut idx = self.count;
        while idx > insert_at {
            self.points[idx] = self.points[idx - 1];
            idx -= 1;
        }
        self.points[insert_at] = BlendPoint1D {
            param,
            clip_index,
            playback_speed,
        };
        self.count += 1;
        true
    }

    /// Computes `(clip_a, clip_b, t)` for `param` with zero heap allocation.
    #[inline]
    pub fn sample_weights(&self, param: f32) -> (u16, u16, f32) {
        if self.count == 0 {
            return (0, 0, 0.0);
        }
        if self.count == 1 || param <= self.points[0].param {
            let c = self.points[0].clip_index;
            return (c, c, 0.0);
        }
        let last = self.count - 1;
        if param >= self.points[last].param {
            let c = self.points[last].clip_index;
            return (c, c, 0.0);
        }
        for i in 0..last {
            let a = &self.points[i];
            let b = &self.points[i + 1];
            if param >= a.param && param <= b.param {
                let span = b.param - a.param;
                let t = if span > 1e-6 {
                    ((param - a.param) / span).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                return (a.clip_index, b.clip_index, t);
            }
        }
        let c = self.points[last].clip_index;
        (c, c, 0.0)
    }
}

/// Evaluates a [`BlendSpace1D`] at `param` and blends the corresponding clip `Motor` poses
/// into `out_pose` using `Motor::sclerp` (`HotZeroHeap`).
pub fn sample_blend_space_1d_zero_heap<const MAX_POINTS: usize>(
    space: &BlendSpace1D<MAX_POINTS>,
    param: f32,
    clip_poses: &[&[Motor]],
    out_pose: &mut [Motor],
) -> Result<usize, SkeletalError> {
    if space.count == 0 || clip_poses.is_empty() || out_pose.is_empty() {
        return Err(SkeletalError::InvalidInput);
    }
    let (clip_a, clip_b, t) = space.sample_weights(param);
    let idx_a = clip_a as usize;
    let idx_b = clip_b as usize;
    if idx_a >= clip_poses.len() || idx_b >= clip_poses.len() {
        return Err(SkeletalError::InvalidInput);
    }
    let pose_a = clip_poses[idx_a];
    let pose_b = clip_poses[idx_b];
    let joint_count = pose_a.len().min(pose_b.len()).min(out_pose.len());
    if joint_count == 0 {
        return Err(SkeletalError::OutputTooSmall);
    }
    let t_f64 = t as f64;
    for j in 0..joint_count {
        out_pose[j] = if idx_a == idx_b || t <= 1e-6 {
            pose_a[j]
        } else if t >= 1.0 - 1e-6 {
            pose_b[j]
        } else {
            Motor::sclerp(&pose_a[j], &pose_b[j], t_f64)
        };
    }
    Ok(joint_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_space_1d_interpolates_sorted_intervals() {
        let mut bs = BlendSpace1D::<8>::new();
        assert!(bs.add_point(4.0, 2));
        assert!(bs.add_point(0.0, 0));
        assert!(bs.add_point(2.0, 1));

        let p0 = [Motor::from_translation(0.0, 0.0, 0.0)];
        let p1 = [Motor::from_translation(2.0, 0.0, 0.0)];
        let p2 = [Motor::from_translation(6.0, 0.0, 0.0)];
        let clips: [&[Motor]; 3] = [&p0, &p1, &p2];
        let mut out = [Motor::identity(); 1];

        sample_blend_space_1d_zero_heap(&bs, 1.0, &clips, &mut out).unwrap();
        let (_, t) = out[0].to_rotation_translation();
        assert!((t[0] - 1.0).abs() < 1e-5);

        sample_blend_space_1d_zero_heap(&bs, 3.0, &clips, &mut out).unwrap();
        let (_, t2) = out[0].to_rotation_translation();
        assert!((t2[0] - 4.0).abs() < 1e-5);
    }
}
