//! 3D Spatial Raycasting (`query_ray.rs`).
//!
//! Provides exact zero-heap 3D ray intersection queries:
//! - [`ray_triangle_moller_trumbore`]: Möller–Trumbore ray-triangle intersection,
//! - [`ray_sphere_local`] & [`ray_sdf_local`]: analytical sphere and sphere-traced SDF raycasts,
//! - [`raycast_collider`] & [`cast_ray`]: BVH-accelerated world-space raycasting against
//!   [`ColliderShape3D`] instances.

use etalon_render::spatial_math::pga::Motor;

use crate::physics3d::broadphase_bvh::{Aabb3D, BroadphaseBvh};
use crate::physics3d::sdf_collider::SdfPrimitive;
use crate::physics3d::shapes::ColliderShape3D;

/// Hit result returned by [`cast_ray`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit3D {
    /// Index of the hit collider.
    pub collider_id: u32,
    /// Distance / time-of-impact along the unit ray direction (`0.0 <= toi <= max_toi`).
    pub toi: f32,
    /// Exact world-space hit position `origin + dir * toi`.
    pub point: [f32; 3],
    /// Outward surface normal at the hit point in world space.
    pub normal: [f32; 3],
}

/// Möller–Trumbore ray-triangle intersection test (`HotZeroHeap`).
///
/// Returns `Some((toi, normal))` if the ray `origin + t * dir` hits triangle `(v0, v1, v2)`
/// within `0.0 <= t <= max_toi`.
#[inline]
pub fn ray_triangle_moller_trumbore(
    origin: [f32; 3],
    dir: [f32; 3],
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    max_toi: f32,
) -> Option<(f32, [f32; 3])> {
    const EPS: f32 = 1e-7;
    let e1 = sub3(v1, v0);
    let e2 = sub3(v2, v0);
    let h = cross3(dir, e2);
    let a = dot3(e1, h);
    if a.abs() < EPS {
        return None;
    }
    let f = 1.0 / a;
    let s = sub3(origin, v0);
    let u = f * dot3(s, h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross3(s, e1);
    let v = f * dot3(dir, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * dot3(e2, q);
    if t >= 0.0 && t <= max_toi {
        let mut n = normalize3(cross3(e1, e2));
        if dot3(n, dir) > 0.0 {
            n = [-n[0], -n[1], -n[2]];
        }
        Some((t, n))
    } else {
        None
    }
}

/// Analytical ray-sphere intersection in local space (sphere centered at origin).
#[inline]
pub fn ray_sphere_local(
    origin: [f32; 3],
    dir: [f32; 3],
    radius: f32,
    max_toi: f32,
) -> Option<(f32, [f32; 3])> {
    let b = dot3(origin, dir);
    let c = dot3(origin, origin) - radius * radius;
    if c > 0.0 && b > 0.0 {
        return None;
    }
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = (-b - disc.sqrt()).max(0.0);
    if t > max_toi {
        return None;
    }
    let hit = [
        origin[0] + dir[0] * t,
        origin[1] + dir[1] * t,
        origin[2] + dir[2] * t,
    ];
    Some((t, normalize3(hit)))
}

/// Sphere-tracing raycast against an analytical [`SdfPrimitive`] in local space.
#[inline]
pub fn ray_sdf_local(
    origin: [f32; 3],
    dir: [f32; 3],
    sdf: &SdfPrimitive,
    max_toi: f32,
) -> Option<(f32, [f32; 3])> {
    let mut t = 0.0f32;
    let mut step = 0usize;
    while step < 64 && t <= max_toi {
        let p = [
            origin[0] + dir[0] * t,
            origin[1] + dir[1] * t,
            origin[2] + dir[2] * t,
        ];
        let d = sdf.evaluate_local(p);
        if d <= 1e-4 {
            let n = sdf.gradient_normal_local(p);
            return Some((t, n));
        }
        t += d.max(1e-3);
        step += 1;
    }
    None
}

/// Raycast against a single [`ColliderShape3D`] transformed by `motor`.
pub fn raycast_collider(
    origin_world: [f32; 3],
    dir_world: [f32; 3],
    max_toi: f32,
    shape: &ColliderShape3D,
    motor: &Motor,
) -> Option<(f32, [f32; 3])> {
    let inv = motor.reverse();
    let o_local = inv.transform_point(origin_world);
    let d_local = normalize3(inv.transform_vector(dir_world));

    let local_hit = match *shape {
        ColliderShape3D::Sphere { radius } => ray_sphere_local(o_local, d_local, radius, max_toi),
        ColliderShape3D::Capsule {
            radius,
            half_height,
        } => {
            let sdf = SdfPrimitive::Capsule {
                radius,
                half_height,
            };
            ray_sdf_local(o_local, d_local, &sdf, max_toi)
        }
        ColliderShape3D::Cuboid { half_extents } => {
            let aabb = Aabb3D::from_center_half_extents([0.0; 3], half_extents);
            aabb.ray_intersect(o_local, d_local, max_toi).map(|(t, _)| {
                let hit = [
                    o_local[0] + d_local[0] * t,
                    o_local[1] + d_local[1] * t,
                    o_local[2] + d_local[2] * t,
                ];
                let dx = (hit[0].abs() - half_extents[0]).abs();
                let dy = (hit[1].abs() - half_extents[1]).abs();
                let dz = (hit[2].abs() - half_extents[2]).abs();
                let n = if dx <= dy && dx <= dz {
                    [hit[0].signum(), 0.0, 0.0]
                } else if dy <= dz {
                    [0.0, hit[1].signum(), 0.0]
                } else {
                    [0.0, 0.0, hit[2].signum()]
                };
                (t, n)
            })
        }
        ColliderShape3D::Cylinder {
            radius,
            half_height,
        } => {
            let sdf = SdfPrimitive::RoundedBox {
                half_extents: [radius, half_height, radius],
                bevel_radius: 0.0,
            };
            ray_sdf_local(o_local, d_local, &sdf, max_toi)
        }
        ColliderShape3D::ConvexHull { .. } => {
            let aabb = shape.local_aabb();
            aabb.ray_intersect(o_local, d_local, max_toi)
                .map(|(t, _)| (t, [-d_local[0], -d_local[1], -d_local[2]]))
        }
        ColliderShape3D::SdfAnalytical { ref primitive } => {
            ray_sdf_local(o_local, d_local, primitive, max_toi)
        }
    };

    local_hit.map(|(t, n_local)| (t, normalize3(motor.transform_vector(n_local))))
}

/// Casts a 3D ray through a [`BroadphaseBvh`] and collider slice, returning the closest [`RayHit3D`].
pub fn cast_ray<const MAX_LEAVES: usize>(
    bvh: &BroadphaseBvh<MAX_LEAVES>,
    shapes: &[ColliderShape3D],
    motors: &[Motor],
    origin: [f32; 3],
    dir: [f32; 3],
    max_toi: f32,
    ignore_collider: Option<u32>,
) -> Option<RayHit3D> {
    let unit_dir = normalize3(dir);
    let mut candidates = [0u32; 128];
    let cand_count = bvh.query_ray_candidates(origin, unit_dir, max_toi, &mut candidates);

    let mut best_hit: Option<RayHit3D> = None;
    let mut best_toi = max_toi;

    let mut i = 0usize;
    while i < cand_count {
        let cid = candidates[i];
        i += 1;
        if Some(cid) == ignore_collider {
            continue;
        }
        let idx = cid as usize;
        if idx >= shapes.len() || idx >= motors.len() {
            continue;
        }
        if let Some((toi, normal)) =
            raycast_collider(origin, unit_dir, best_toi, &shapes[idx], &motors[idx])
        {
            if toi <= best_toi {
                best_toi = toi;
                best_hit = Some(RayHit3D {
                    collider_id: cid,
                    toi,
                    point: [
                        origin[0] + unit_dir[0] * toi,
                        origin[1] + unit_dir[1] * toi,
                        origin[2] + unit_dir[2] * toi,
                    ],
                    normal,
                });
            }
        }
    }

    best_hit
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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
pub(crate) fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = dot3(v, v).sqrt();
    if l < 1e-8 {
        [0.0, 1.0, 0.0]
    } else {
        [v[0] / l, v[1] / l, v[2] / l]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use etalon_render::spatial_math::pga::motor_translate;

    #[test]
    fn ray_triangle_and_bvh_raycast_hit_analytical_sdf() {
        let tri_hit = ray_triangle_moller_trumbore(
            [0.2, 5.0, 0.2],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            10.0,
        );
        assert!(tri_hit.is_some());
        assert!((tri_hit.unwrap().0 - 5.0).abs() < 1e-5);

        let shapes = [ColliderShape3D::SdfAnalytical {
            primitive: SdfPrimitive::Sphere { radius: 1.0 },
        }];
        let motors = [motor_translate([0.0, 0.0, 5.0])];
        let mut bvh = BroadphaseBvh::<16>::new(0.1);
        bvh.insert(0, shapes[0].world_aabb(&motors[0])).unwrap();

        let r_hit = cast_ray(
            &bvh,
            &shapes,
            &motors,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            10.0,
            None,
        )
        .unwrap();
        assert_eq!(r_hit.collider_id, 0);
        assert!((r_hit.toi - 4.0).abs() < 1e-2);
    }
}
