//! Zero-Heap 2D Delaunay Animation Blend Space (`blend_space_2d.rs`).
//!
//! Bowyer–Watson 2D Delaunay triangulation over parameter space with $O(N)$
//! enclosing-triangle lookup, closest-hull-edge projection when outside the convex hull,
//! and barycentric geodesic `Motor::sclerp` pose synthesis (`HotZeroHeap`).

use super::{Motor, SkeletalError};

/// A 2D parameter sample point inside a [`BlendSpace2D`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BlendPoint2D {
    /// 2D parameter coordinates `[x, y]` (e.g. `[strafe_x, forward_z]`).
    pub pos: [f32; 2],
    /// Clip index in the character's animation clip table.
    pub clip_index: u16,
    /// Playback speed multiplier (`1.0` = authored speed).
    pub playback_speed: f32,
}

/// A Delaunay triangle indexing 3 points in [`BlendSpace2D::points`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlendTriangle {
    pub indices: [u8; 3],
}

/// Result of sampling a [`BlendSpace2D`] at a 2D parameter coordinate.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BlendSample2D {
    /// Up to 3 clip indices contributing to the blended pose.
    pub clips: [u16; 3],
    /// Normalized barycentric weights summing to `1.0`.
    pub weights: [f32; 3],
}

/// Fixed-capacity 2D Delaunay animation blend space (`MAX_POINTS = 16`, `MAX_TRIS = 32`).
#[derive(Clone, Copy, Debug)]
pub struct BlendSpace2D<const MAX_POINTS: usize = 16, const MAX_TRIS: usize = 32> {
    pub points: [BlendPoint2D; MAX_POINTS],
    pub point_count: usize,
    pub triangles: [BlendTriangle; MAX_TRIS],
    pub tri_count: usize,
}

impl<const MAX_POINTS: usize, const MAX_TRIS: usize> Default
    for BlendSpace2D<MAX_POINTS, MAX_TRIS>
{
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn cross_2d(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

#[inline]
fn point_in_circumcircle(a: [f32; 2], b: [f32; 2], c: [f32; 2], p: [f32; 2]) -> bool {
    let ax = (a[0] - p[0]) as f64;
    let ay = (a[1] - p[1]) as f64;
    let bx = (b[0] - p[0]) as f64;
    let by = (b[1] - p[1]) as f64;
    let cx = (c[0] - p[0]) as f64;
    let cy = (c[1] - p[1]) as f64;

    let ab = ax * ax + ay * ay;
    let cd = bx * bx + by * by;
    let ef = cx * cx + cy * cy;

    let det = ax * (by * ef - cy * cd) - ay * (bx * ef - cx * cd) + ab * (bx * cy - cx * by);
    let orient =
        (b[0] - a[0]) as f64 * (c[1] - a[1]) as f64 - (b[1] - a[1]) as f64 * (c[0] - a[0]) as f64;
    if orient > 0.0 {
        det > 1e-7
    } else {
        det < -1e-7
    }
}

#[inline]
fn closest_point_on_segment_2d(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> ([f32; 2], f32, f32) {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len_sq = dx * dx + dy * dy;
    let t = if len_sq > 1e-12 {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let proj = [a[0] + t * dx, a[1] + t * dy];
    let ex = p[0] - proj[0];
    let ey = p[1] - proj[1];
    (proj, t, ex * ex + ey * ey)
}

impl<const MAX_POINTS: usize, const MAX_TRIS: usize> BlendSpace2D<MAX_POINTS, MAX_TRIS> {
    /// Creates an empty 2D blend space.
    pub const fn new() -> Self {
        Self {
            points: [BlendPoint2D {
                pos: [0.0, 0.0],
                clip_index: 0,
                playback_speed: 1.0,
            }; MAX_POINTS],
            point_count: 0,
            triangles: [BlendTriangle { indices: [0, 0, 0] }; MAX_TRIS],
            tri_count: 0,
        }
    }

    /// Adds a 2D sample point `(x, y)` associated with `clip_index`.
    pub fn add_point(&mut self, x: f32, y: f32, clip_index: u16) -> bool {
        if self.point_count >= MAX_POINTS
            || self.point_count >= 60
            || !x.is_finite()
            || !y.is_finite()
        {
            return false;
        }
        self.points[self.point_count] = BlendPoint2D {
            pos: [x, y],
            clip_index,
            playback_speed: 1.0,
        };
        self.point_count += 1;
        true
    }

    /// Builds a Bowyer–Watson Delaunay triangulation over `self.points[..self.point_count]`
    /// using fixed stack scratch arrays (zero heap allocations).
    pub fn build_triangulation(&mut self) -> usize {
        self.tri_count = 0;
        let n = self.point_count;
        if n < 3 {
            return 0;
        }

        let mut min_x = self.points[0].pos[0];
        let mut max_x = min_x;
        let mut min_y = self.points[0].pos[1];
        let mut max_y = min_y;
        for i in 1..n {
            let p = self.points[i].pos;
            if p[0] < min_x {
                min_x = p[0];
            }
            if p[0] > max_x {
                max_x = p[0];
            }
            if p[1] < min_y {
                min_y = p[1];
            }
            if p[1] > max_y {
                max_y = p[1];
            }
        }
        let span = (max_x - min_x).max(max_y - min_y).max(1.0) * 10.0;
        let mid_x = 0.5 * (min_x + max_x);
        let mid_y = 0.5 * (min_y + max_y);

        let super_pts = [
            [mid_x - 2.0 * span, mid_y - span],
            [mid_x + 2.0 * span, mid_y - span],
            [mid_x, mid_y + 2.0 * span],
        ];

        let get_pos = |idx: u8, pts: &[BlendPoint2D; MAX_POINTS]| -> [f32; 2] {
            let u = idx as usize;
            if u < n {
                pts[u].pos
            } else {
                super_pts[u - n]
            }
        };

        let mut work_tris = [[0u8; 3]; 128];
        let mut work_count = 1usize;
        work_tris[0] = [n as u8, (n + 1) as u8, (n + 2) as u8];

        for i in 0..n {
            let p = self.points[i].pos;
            let mut bad = [false; 128];
            for t in 0..work_count {
                let tri = work_tris[t];
                let a = get_pos(tri[0], &self.points);
                let b = get_pos(tri[1], &self.points);
                let c = get_pos(tri[2], &self.points);
                if point_in_circumcircle(a, b, c, p) {
                    bad[t] = true;
                }
            }

            let mut poly_edges = [[0u8; 2]; 128];
            let mut edge_count = 0usize;
            for t in 0..work_count {
                if !bad[t] {
                    continue;
                }
                let tri = work_tris[t];
                let edges = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]];
                for e in edges {
                    let mut shared = false;
                    for other_t in 0..work_count {
                        if other_t == t || !bad[other_t] {
                            continue;
                        }
                        let ot = work_tris[other_t];
                        let o_edges = [[ot[0], ot[1]], [ot[1], ot[2]], [ot[2], ot[0]]];
                        if o_edges.iter().any(|oe| {
                            (oe[0] == e[1] && oe[1] == e[0]) || (oe[0] == e[0] && oe[1] == e[1])
                        }) {
                            shared = true;
                            break;
                        }
                    }
                    if !shared && edge_count < poly_edges.len() {
                        poly_edges[edge_count] = e;
                        edge_count += 1;
                    }
                }
            }

            let mut write = 0usize;
            for t in 0..work_count {
                if !bad[t] {
                    work_tris[write] = work_tris[t];
                    write += 1;
                }
            }
            work_count = write;

            for e_idx in 0..edge_count {
                if work_count >= work_tris.len() {
                    break;
                }
                let e = poly_edges[e_idx];
                let a = get_pos(e[0], &self.points);
                let b = get_pos(e[1], &self.points);
                if cross_2d(a, b, p) >= 0.0 {
                    work_tris[work_count] = [e[0], e[1], i as u8];
                } else {
                    work_tris[work_count] = [e[1], e[0], i as u8];
                }
                work_count += 1;
            }
        }

        for t in 0..work_count {
            let tri = work_tris[t];
            if (tri[0] as usize) < n && (tri[1] as usize) < n && (tri[2] as usize) < n {
                let a = self.points[tri[0] as usize].pos;
                let b = self.points[tri[1] as usize].pos;
                let c = self.points[tri[2] as usize].pos;
                if cross_2d(a, b, c).abs() > 1e-6 && self.tri_count < MAX_TRIS {
                    self.triangles[self.tri_count] = BlendTriangle { indices: tri };
                    self.tri_count += 1;
                }
            }
        }

        self.tri_count
    }

    /// Evaluates exact barycentric weights for `query` in $O(N)$ with zero heap allocation (`HotZeroHeap`).
    pub fn sample_weights(&self, query: [f32; 2]) -> BlendSample2D {
        if self.point_count == 0 {
            return BlendSample2D {
                clips: [0, 0, 0],
                weights: [1.0, 0.0, 0.0],
            };
        }
        if self.point_count == 1 {
            let c = self.points[0].clip_index;
            return BlendSample2D {
                clips: [c, c, c],
                weights: [1.0, 0.0, 0.0],
            };
        }
        if self.point_count == 2 || self.tri_count == 0 {
            let a = self.points[0];
            let b = self.points[1];
            let (_, t, _) = closest_point_on_segment_2d(a.pos, b.pos, query);
            return BlendSample2D {
                clips: [a.clip_index, b.clip_index, a.clip_index],
                weights: [1.0 - t, t, 0.0],
            };
        }

        let mut best_edge_dist_sq = f32::INFINITY;
        let mut best_sample = BlendSample2D {
            clips: [self.points[0].clip_index; 3],
            weights: [1.0, 0.0, 0.0],
        };

        for t_idx in 0..self.tri_count {
            let tri = self.triangles[t_idx].indices;
            let p0 = self.points[tri[0] as usize];
            let p1 = self.points[tri[1] as usize];
            let p2 = self.points[tri[2] as usize];

            let area = cross_2d(p0.pos, p1.pos, p2.pos);
            if area.abs() <= 1e-8 {
                continue;
            }
            let inv_area = 1.0 / area;
            let w0 = cross_2d(query, p1.pos, p2.pos) * inv_area;
            let w1 = cross_2d(p0.pos, query, p2.pos) * inv_area;
            let w2 = 1.0 - w0 - w1;

            if w0 >= -1e-4 && w1 >= -1e-4 && w2 >= -1e-4 {
                let cw0 = w0.max(0.0);
                let cw1 = w1.max(0.0);
                let cw2 = w2.max(0.0);
                let sum = (cw0 + cw1 + cw2).max(1e-8);
                return BlendSample2D {
                    clips: [p0.clip_index, p1.clip_index, p2.clip_index],
                    weights: [cw0 / sum, cw1 / sum, cw2 / sum],
                };
            }

            let edges = [
                (p0, p1, p2, 0usize, 1usize),
                (p1, p2, p0, 1usize, 2usize),
                (p2, p0, p1, 2usize, 0usize),
            ];
            for (ea, eb, ec, _, _) in edges {
                let (_, seg_t, dist_sq) = closest_point_on_segment_2d(ea.pos, eb.pos, query);
                if dist_sq < best_edge_dist_sq {
                    best_edge_dist_sq = dist_sq;
                    best_sample = BlendSample2D {
                        clips: [ea.clip_index, eb.clip_index, ec.clip_index],
                        weights: [1.0 - seg_t, seg_t, 0.0],
                    };
                }
            }
        }

        best_sample
    }
}

/// Samples a [`BlendSpace2D`] at `query` and blends the 3 enclosing triangle clip poses
/// per joint using barycentric `Motor::sclerp` with zero heap allocation (`HotZeroHeap`).
pub fn sample_blend_space_2d_zero_heap<const MAX_POINTS: usize, const MAX_TRIS: usize>(
    space: &BlendSpace2D<MAX_POINTS, MAX_TRIS>,
    query: [f32; 2],
    clip_poses: &[&[Motor]],
    out_pose: &mut [Motor],
) -> Result<(BlendSample2D, usize), SkeletalError> {
    if space.point_count == 0 || clip_poses.is_empty() || out_pose.is_empty() {
        return Err(SkeletalError::InvalidInput);
    }
    let sample = space.sample_weights(query);
    let i0 = sample.clips[0] as usize;
    let i1 = sample.clips[1] as usize;
    let i2 = sample.clips[2] as usize;
    if i0 >= clip_poses.len() || i1 >= clip_poses.len() || i2 >= clip_poses.len() {
        return Err(SkeletalError::InvalidInput);
    }
    let p0 = clip_poses[i0];
    let p1 = clip_poses[i1];
    let p2 = clip_poses[i2];
    let joint_count = p0.len().min(p1.len()).min(p2.len()).min(out_pose.len());
    if joint_count == 0 {
        return Err(SkeletalError::OutputTooSmall);
    }

    let [w0, w1, w2] = sample.weights;
    let w01 = w0 + w1;
    let t01 = if w01 > 1e-6 {
        (w1 / w01).clamp(0.0, 1.0) as f64
    } else {
        0.0
    };
    let t2 = w2.clamp(0.0, 1.0) as f64;

    for j in 0..joint_count {
        let m01 = if t01 <= 1e-6 || i0 == i1 {
            p0[j]
        } else if t01 >= 1.0 - 1e-6 {
            p1[j]
        } else {
            Motor::sclerp(&p0[j], &p1[j], t01)
        };

        out_pose[j] = if t2 <= 1e-6 {
            m01
        } else if t2 >= 1.0 - 1e-6 {
            p2[j]
        } else {
            Motor::sclerp(&m01, &p2[j], t2)
        };
    }

    Ok((sample, joint_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeletal::assert_zero_alloc;

    #[test]
    fn blend_space_2d_delaunay_barycentric_and_hull_projection_zero_heap() {
        let mut bs = BlendSpace2D::<16, 32>::new();
        assert!(bs.add_point(0.0, 0.0, 0));
        assert!(bs.add_point(1.0, 0.0, 1));
        assert!(bs.add_point(0.0, 1.0, 2));
        assert!(bs.add_point(1.0, 1.0, 3));

        let tris = bs.build_triangulation();
        assert_eq!(tris, 2);

        let c0 = [Motor::from_translation(0.0, 0.0, 0.0)];
        let c1 = [Motor::from_translation(10.0, 0.0, 0.0)];
        let c2 = [Motor::from_translation(0.0, 0.0, 10.0)];
        let c3 = [Motor::from_translation(10.0, 0.0, 10.0)];
        let clips: [&[Motor]; 4] = [&c0, &c1, &c2, &c3];
        let mut out = [Motor::identity(); 1];

        assert_zero_alloc("sample_blend_space_2d", || {
            let (sample, n) =
                sample_blend_space_2d_zero_heap(&bs, [0.25, 0.5], &clips, &mut out).unwrap();
            assert_eq!(n, 1);
            let w_sum: f32 = sample.weights.iter().sum();
            assert!((w_sum - 1.0).abs() < 1e-5);
            let (_, trans) = out[0].to_rotation_translation();
            assert!((trans[0] - 2.5).abs() < 1e-4);
            assert!((trans[2] - 5.0).abs() < 1e-4);

            let (sample_out, _) =
                sample_blend_space_2d_zero_heap(&bs, [2.0, 0.5], &clips, &mut out).unwrap();
            let w_sum_out: f32 = sample_out.weights.iter().sum();
            assert!((w_sum_out - 1.0).abs() < 1e-5);
            let (_, trans_out) = out[0].to_rotation_translation();
            assert!((trans_out[0] - 10.0).abs() < 1e-4);
            assert!((trans_out[2] - 5.0).abs() < 1e-4);
        });
    }
}
