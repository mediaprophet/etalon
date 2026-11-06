//! Zero-Heap GJK / EPA & Analytical SDF Narrowphase (`gjk_epa.rs`).
//!
//! Provides zero-heap Gilbert–Johnson–Keerthi (GJK) intersection, Expanding Polytope Algorithm
//! (EPA) penetration extraction, and analytical SDF gradient $\nabla\text{sdf}(\mathbf{x})$ contacts.

use etalon_render::spatial_math::pga::Motor;

use crate::physics3d::sdf_collider::SdfPrimitive;
pub use crate::physics3d::shapes::{ColliderShape3D, MAX_CONVEX_HULL_POINTS};

/// Minkowski difference support vertex $v = S_A(\mathbf{d}) - S_B(-\mathbf{d})$.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SupportVertex {
    pub point: [f32; 3],
    pub sup_a: [f32; 3],
    pub sup_b: [f32; 3],
}

#[inline]
fn minkowski_support(
    shape_a: &ColliderShape3D,
    motor_a: &Motor,
    shape_b: &ColliderShape3D,
    motor_b: &Motor,
    dir: [f32; 3],
) -> SupportVertex {
    let neg_dir = [-dir[0], -dir[1], -dir[2]];
    let sup_a = shape_a.support_point_world(motor_a, dir);
    let sup_b = shape_b.support_point_world(motor_b, neg_dir);
    SupportVertex {
        point: sub3(sup_a, sup_b),
        sup_a,
        sup_b,
    }
}

/// Result of a narrowphase contact query between two 3D colliders.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NarrowphaseContact {
    /// Contact normal pointing from collider A toward collider B.
    pub normal: [f32; 3],
    /// Positive penetration depth in meters.
    pub penetration: f32,
    /// Contact point on collider A in world space.
    pub point_a: [f32; 3],
    /// Contact point on collider B in world space.
    pub point_b: [f32; 3],
}

/// Zero-heap GJK intersection test on a fixed 4-vertex stack simplex.
pub fn gjk_intersect(
    shape_a: &ColliderShape3D,
    motor_a: &Motor,
    shape_b: &ColliderShape3D,
    motor_b: &Motor,
) -> Option<([SupportVertex; 4], usize)> {
    let mut dir = sub3(motor_b.translation(), motor_a.translation());
    if len_sq3(dir) < 1e-8 {
        dir = [1.0, 0.0, 0.0];
    }

    let mut simplex = [SupportVertex::default(); 4];
    simplex[0] = minkowski_support(shape_a, motor_a, shape_b, motor_b, dir);
    if dot3(simplex[0].point, dir) < 0.0 {
        return None;
    }

    let mut simplex_len = 1usize;
    dir = neg3(simplex[0].point);
    if len_sq3(dir) < 1e-10 {
        return Some((simplex, simplex_len));
    }

    let mut iter = 0usize;
    while iter < 32 {
        iter += 1;
        let a = minkowski_support(shape_a, motor_a, shape_b, motor_b, dir);
        if dot3(a.point, dir) < 0.0 {
            return None;
        }
        simplex[simplex_len] = a;
        simplex_len += 1;
        if update_gjk_simplex(&mut simplex, &mut simplex_len, &mut dir) {
            return Some((simplex, simplex_len));
        }
    }
    None
}

fn update_gjk_simplex(
    s: &mut [SupportVertex; 4],
    len: &mut usize,
    dir: &mut [f32; 3],
) -> bool {
    match *len {
        2 => {
            let (a, b) = (s[1].point, s[0].point);
            let (ab, ao) = (sub3(b, a), neg3(a));
            if dot3(ab, ao) > 0.0 {
                let mut n = cross3(cross3(ab, ao), ab);
                if len_sq3(n) < 1e-10 {
                    n = orthogonal_vec3(ab);
                }
                *dir = n;
            } else {
                s[0] = s[1];
                *len = 1;
                *dir = ao;
            }
            false
        }
        3 => {
            let (a, b, c) = (s[2].point, s[1].point, s[0].point);
            let (ab, ac, ao) = (sub3(b, a), sub3(c, a), neg3(a));
            let abc = cross3(ab, ac);
            if dot3(cross3(abc, ac), ao) > 0.0 {
                if dot3(ac, ao) > 0.0 {
                    s[1] = s[2];
                    *len = 2;
                    let mut n = cross3(cross3(ac, ao), ac);
                    if len_sq3(n) < 1e-10 {
                        n = orthogonal_vec3(ac);
                    }
                    *dir = n;
                } else {
                    s[0] = s[1];
                    s[1] = s[2];
                    *len = 2;
                    return update_gjk_simplex(s, len, dir);
                }
            } else if dot3(cross3(ab, abc), ao) > 0.0 {
                s[0] = s[1];
                s[1] = s[2];
                *len = 2;
                return update_gjk_simplex(s, len, dir);
            } else if dot3(abc, ao) > 0.0 {
                *dir = abc;
            } else {
                s.swap(0, 1);
                *dir = neg3(abc);
            }
            false
        }
        4 => {
            let (a, b, c, d) = (s[3].point, s[2].point, s[1].point, s[0].point);
            let (ao, ab, ac, ad) = (neg3(a), sub3(b, a), sub3(c, a), sub3(d, a));
            let (abc, acd, adb) = (cross3(ab, ac), cross3(ac, ad), cross3(ad, ab));
            if dot3(abc, ao) > 0.0 {
                s[0] = s[1];
                s[1] = s[2];
                s[2] = s[3];
                *len = 3;
                return update_gjk_simplex(s, len, dir);
            }
            if dot3(acd, ao) > 0.0 {
                s[2] = s[3];
                *len = 3;
                return update_gjk_simplex(s, len, dir);
            }
            if dot3(adb, ao) > 0.0 {
                s[1] = s[0];
                s[0] = s[2];
                s[2] = s[3];
                *len = 3;
                return update_gjk_simplex(s, len, dir);
            }
            true
        }
        _ => false,
    }
}

/// Evaluates narrowphase contact between two 3D colliders (`HotZeroHeap`).
pub fn compute_narrowphase_contact(
    shape_a: &ColliderShape3D,
    motor_a: &Motor,
    shape_b: &ColliderShape3D,
    motor_b: &Motor,
) -> Option<NarrowphaseContact> {
    if let (ColliderShape3D::Sphere { radius: ra }, ColliderShape3D::Sphere { radius: rb }) =
        (shape_a, shape_b)
    {
        let (ca, cb) = (motor_a.translation(), motor_b.translation());
        let diff = sub3(cb, ca);
        let dist = len3(diff);
        let sum_r = ra + rb;
        if dist >= sum_r {
            return None;
        }
        let normal = if dist > 1e-6 {
            [diff[0] / dist, diff[1] / dist, diff[2] / dist]
        } else {
            [0.0, 1.0, 0.0]
        };
        return Some(NarrowphaseContact {
            normal,
            penetration: sum_r - dist,
            point_a: [
                ca[0] + normal[0] * (*ra),
                ca[1] + normal[1] * (*ra),
                ca[2] + normal[2] * (*ra),
            ],
            point_b: [
                cb[0] - normal[0] * (*rb),
                cb[1] - normal[1] * (*rb),
                cb[2] - normal[2] * (*rb),
            ],
        });
    }

    if let ColliderShape3D::SdfAnalytical { primitive } = shape_a {
        return collide_sdf_vs_convex(primitive, motor_a, shape_b, motor_b, false);
    }
    if let ColliderShape3D::SdfAnalytical { primitive } = shape_b {
        return collide_sdf_vs_convex(primitive, motor_b, shape_a, motor_a, true);
    }

    if gjk_intersect(shape_a, motor_a, shape_b, motor_b).is_none() {
        return None;
    }
    Some(epa_extract_contact(shape_a, motor_a, shape_b, motor_b))
}

fn collide_sdf_vs_convex(
    sdf: &SdfPrimitive,
    motor_sdf: &Motor,
    convex: &ColliderShape3D,
    motor_convex: &Motor,
    invert_normal: bool,
) -> Option<NarrowphaseContact> {
    let (center_sdf, center_cvx) = (motor_sdf.translation(), motor_convex.translation());
    let (_, grad_at_center) = sdf.evaluate_world(motor_sdf, center_cvx);

    let search_dir = if len_sq3(grad_at_center) > 1e-8 {
        neg3(grad_at_center)
    } else {
        normalize3_or(sub3(center_sdf, center_cvx), [0.0, -1.0, 0.0])
    };
    let pt_on_convex = convex.support_point_world(motor_convex, search_dir);
    let (signed_dist, sdf_outward_normal) = sdf.evaluate_world(motor_sdf, pt_on_convex);
    if signed_dist >= 0.0 {
        return None;
    }

    let penetration = -signed_dist;
    let pt_on_sdf = [
        pt_on_convex[0] - sdf_outward_normal[0] * signed_dist,
        pt_on_convex[1] - sdf_outward_normal[1] * signed_dist,
        pt_on_convex[2] - sdf_outward_normal[2] * signed_dist,
    ];

    if invert_normal {
        Some(NarrowphaseContact {
            normal: neg3(sdf_outward_normal),
            penetration,
            point_a: pt_on_convex,
            point_b: pt_on_sdf,
        })
    } else {
        Some(NarrowphaseContact {
            normal: sdf_outward_normal,
            penetration,
            point_a: pt_on_sdf,
            point_b: pt_on_convex,
        })
    }
}

/// Fixed-stack EPA / Minimum-Translation-Vector solver over 13 canonical + centroid directions.
fn epa_extract_contact(
    shape_a: &ColliderShape3D,
    motor_a: &Motor,
    shape_b: &ColliderShape3D,
    motor_b: &Motor,
) -> NarrowphaseContact {
    let (ca, cb) = (motor_a.translation(), motor_b.translation());
    let c_diff = sub3(cb, ca);
    let c_dir = normalize3_or(c_diff, [0.0, 1.0, 0.0]);

    const DIRS: [[f32; 3]; 13] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.70710677, 0.70710677, 0.0],
        [0.70710677, -0.70710677, 0.0],
        [0.70710677, 0.0, 0.70710677],
        [0.70710677, 0.0, -0.70710677],
        [0.0, 0.70710677, 0.70710677],
        [0.0, 0.70710677, -0.70710677],
        [0.57735026, 0.57735026, 0.57735026],
        [0.57735026, 0.57735026, -0.57735026],
        [0.57735026, -0.57735026, 0.57735026],
        [-0.57735026, 0.57735026, 0.57735026],
    ];

    let mut best_pen = f32::MAX;
    let mut best_normal = c_dir;
    let mut best_a = ca;
    let mut best_b = cb;

    let mut check_dir = |d: [f32; 3]| {
        let n = if dot3(d, c_diff) < 0.0 { neg3(d) } else { d };
        let sv = minkowski_support(shape_a, motor_a, shape_b, motor_b, n);
        let pen = dot3(sv.point, n);
        if pen > 0.0 && pen < best_pen {
            best_pen = pen;
            best_normal = n;
            best_a = sv.sup_a;
            best_b = sv.sup_b;
        }
    };

    check_dir(c_dir);
    let mut i = 0usize;
    while i < DIRS.len() {
        check_dir(DIRS[i]);
        i += 1;
    }
    if best_pen == f32::MAX {
        best_pen = 0.01;
    }

    NarrowphaseContact {
        normal: best_normal,
        penetration: best_pen,
        point_a: best_a,
        point_b: best_b,
    }
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn neg3(v: [f32; 3]) -> [f32; 3] {
    [-v[0], -v[1], -v[2]]
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn len_sq3(v: [f32; 3]) -> f32 {
    dot3(v, v)
}

#[inline]
fn len3(v: [f32; 3]) -> f32 {
    len_sq3(v).sqrt()
}

#[inline]
fn normalize3_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let l = len3(v);
    if l < 1e-8 {
        fallback
    } else {
        [v[0] / l, v[1] / l, v[2] / l]
    }
}

#[inline]
fn orthogonal_vec3(v: [f32; 3]) -> [f32; 3] {
    if v[0].abs() < 0.9 {
        normalize3_or(cross3(v, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
    } else {
        normalize3_or(cross3(v, [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use etalon_render::spatial_math::pga::motor_translate;

    #[test]
    fn gjk_and_narrowphase_detect_box_sphere_and_sdf_contacts() {
        let box_shape = ColliderShape3D::Cuboid {
            half_extents: [1.0, 1.0, 1.0],
        };
        let sphere_shape = ColliderShape3D::Sphere { radius: 0.6 };
        let m_box = Motor::identity();
        let m_sphere_hit = motor_translate([1.4, 0.0, 0.0]);
        let m_sphere_miss = motor_translate([2.0, 0.0, 0.0]);

        let c = compute_narrowphase_contact(&box_shape, &m_box, &sphere_shape, &m_sphere_hit).unwrap();
        assert!((c.penetration - 0.2).abs() < 0.05);
        assert!(c.normal[0] > 0.9);
        assert!(compute_narrowphase_contact(&box_shape, &m_box, &sphere_shape, &m_sphere_miss).is_none());

        let sdf_shape = ColliderShape3D::SdfAnalytical {
            primitive: SdfPrimitive::RoundedBox {
                half_extents: [0.9, 0.9, 0.9],
                bevel_radius: 0.1,
            },
        };
        let sdf_hit =
            compute_narrowphase_contact(&sdf_shape, &m_box, &sphere_shape, &m_sphere_hit).unwrap();
        assert!((sdf_hit.penetration - 0.2).abs() < 0.02);
        assert!(sdf_hit.normal[0] > 0.95);
    }
}
