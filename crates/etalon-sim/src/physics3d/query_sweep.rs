//! 3D Swept Shape-Casting & Volume Overlap Queries (`query_sweep.rs`).
//!
//! Provides exact zero-heap 3D swept shape and overlap queries:
//! - [`cast_sphere`]: swept sphere time-of-impact query via Minkowski radius inflation,
//! - [`cast_capsule`]: swept vertical capsule time-of-impact query (used by `step_character_3d`),
//! - [`overlap_aabb`]: broadphase + narrowphase AABB overlap query.

use etalon_render::spatial_math::pga::Motor;

use crate::physics3d::broadphase_bvh::{Aabb3D, BroadphaseBvh};
use crate::physics3d::query_ray::{normalize3, raycast_collider};
use crate::physics3d::sdf_collider::SdfPrimitive;
use crate::physics3d::shapes::ColliderShape3D;

/// Hit result returned by [`cast_sphere`] and [`cast_capsule`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeCastHit3D {
    /// Index of the hit collider.
    pub collider_id: u32,
    /// Travel distance along `dir` at initial contact (`0.0 <= toi <= max_dist`).
    pub toi: f32,
    /// Contact point in world space.
    pub point: [f32; 3],
    /// Outward surface normal in world space.
    pub normal: [f32; 3],
}

/// Casts a sphere of `radius` along `dir` up to `max_dist` by inflating target primitives by `radius`.
#[allow(clippy::too_many_arguments)]
pub fn cast_sphere<const MAX_LEAVES: usize>(
    bvh: &BroadphaseBvh<MAX_LEAVES>,
    shapes: &[ColliderShape3D],
    motors: &[Motor],
    center: [f32; 3],
    radius: f32,
    dir: [f32; 3],
    max_dist: f32,
    ignore_collider: Option<u32>,
) -> Option<ShapeCastHit3D> {
    let unit_dir = normalize3(dir);
    let sweep_aabb = Aabb3D::from_center_half_extents(center, [radius; 3]).union(
        &Aabb3D::from_center_half_extents(
            [
                center[0] + unit_dir[0] * max_dist,
                center[1] + unit_dir[1] * max_dist,
                center[2] + unit_dir[2] * max_dist,
            ],
            [radius; 3],
        ),
    );

    let mut candidates = [0u32; 128];
    let cand_count = bvh.overlap_aabb(&sweep_aabb, &mut candidates);

    let mut best_hit: Option<ShapeCastHit3D> = None;
    let mut best_toi = max_dist;

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

        // Inflate target shape by `radius` (Minkowski sum with sphere) and raycast!
        let inflated = match shapes[idx] {
            ColliderShape3D::Sphere { radius: r } => ColliderShape3D::Sphere {
                radius: r + radius,
            },
            ColliderShape3D::Capsule {
                radius: r,
                half_height,
            } => ColliderShape3D::Capsule {
                radius: r + radius,
                half_height,
            },
            ColliderShape3D::Cuboid { half_extents } => ColliderShape3D::SdfAnalytical {
                primitive: SdfPrimitive::RoundedBox {
                    half_extents,
                    bevel_radius: radius,
                },
            },
            ColliderShape3D::Cylinder {
                radius: r,
                half_height,
            } => ColliderShape3D::Cylinder {
                radius: r + radius,
                half_height: half_height + radius,
            },
            ColliderShape3D::ConvexHull { .. } => {
                let he = shapes[idx].local_aabb().half_extents();
                ColliderShape3D::SdfAnalytical {
                    primitive: SdfPrimitive::RoundedBox {
                        half_extents: he,
                        bevel_radius: radius,
                    },
                }
            }
            ColliderShape3D::SdfAnalytical { primitive } => match primitive {
                SdfPrimitive::Sphere { radius: r } => ColliderShape3D::SdfAnalytical {
                    primitive: SdfPrimitive::Sphere {
                        radius: r + radius,
                    },
                },
                SdfPrimitive::RoundedBox {
                    half_extents,
                    bevel_radius,
                } => ColliderShape3D::SdfAnalytical {
                    primitive: SdfPrimitive::RoundedBox {
                        half_extents,
                        bevel_radius: bevel_radius + radius,
                    },
                },
                SdfPrimitive::Capsule {
                    radius: r,
                    half_height,
                } => ColliderShape3D::SdfAnalytical {
                    primitive: SdfPrimitive::Capsule {
                        radius: r + radius,
                        half_height,
                    },
                },
                SdfPrimitive::Torus {
                    major_radius,
                    minor_radius,
                } => ColliderShape3D::SdfAnalytical {
                    primitive: SdfPrimitive::Torus {
                        major_radius,
                        minor_radius: minor_radius + radius,
                    },
                },
                SdfPrimitive::SmoothUnionPair {
                    center_a,
                    radius_a,
                    center_b,
                    radius_b,
                    blend_radius,
                } => ColliderShape3D::SdfAnalytical {
                    primitive: SdfPrimitive::SmoothUnionPair {
                        center_a,
                        radius_a: radius_a + radius,
                        center_b,
                        radius_b: radius_b + radius,
                        blend_radius,
                    },
                },
            },
        };

        if let Some((toi, normal)) =
            raycast_collider(center, unit_dir, best_toi, &inflated, &motors[idx])
        {
            if toi <= best_toi {
                best_toi = toi;
                let sphere_center_at_hit = [
                    center[0] + unit_dir[0] * toi,
                    center[1] + unit_dir[1] * toi,
                    center[2] + unit_dir[2] * toi,
                ];
                best_hit = Some(ShapeCastHit3D {
                    collider_id: cid,
                    toi,
                    point: [
                        sphere_center_at_hit[0] - normal[0] * radius,
                        sphere_center_at_hit[1] - normal[1] * radius,
                        sphere_center_at_hit[2] - normal[2] * radius,
                    ],
                    normal,
                });
            }
        }
    }

    best_hit
}

/// Casts a vertical 3D capsule (`radius`, `half_height`) along `dir` up to `max_dist`
/// by sweeping 3 axial proxy spheres (bottom hemisphere, center, top hemisphere).
#[allow(clippy::too_many_arguments)]
pub fn cast_capsule<const MAX_LEAVES: usize>(
    bvh: &BroadphaseBvh<MAX_LEAVES>,
    shapes: &[ColliderShape3D],
    motors: &[Motor],
    center: [f32; 3],
    radius: f32,
    half_height: f32,
    dir: [f32; 3],
    max_dist: f32,
    ignore_collider: Option<u32>,
) -> Option<ShapeCastHit3D> {
    let offsets = [-half_height, 0.0, half_height];
    let mut best_hit: Option<ShapeCastHit3D> = None;
    let mut best_toi = max_dist;

    let mut i = 0usize;
    while i < 3 {
        let sample_center = [center[0], center[1] + offsets[i], center[2]];
        if let Some(hit) = cast_sphere(
            bvh,
            shapes,
            motors,
            sample_center,
            radius,
            dir,
            best_toi,
            ignore_collider,
        ) {
            if hit.toi <= best_toi {
                best_toi = hit.toi;
                best_hit = Some(hit);
            }
        }
        i += 1;
    }

    best_hit
}

/// Convenience wrapper returning all collider IDs overlapping `query_aabb`.
#[inline]
pub fn overlap_aabb<const MAX_LEAVES: usize>(
    bvh: &BroadphaseBvh<MAX_LEAVES>,
    query_aabb: &Aabb3D,
    out_collider_ids: &mut [u32],
) -> usize {
    bvh.overlap_aabb(query_aabb, out_collider_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use etalon_render::spatial_math::pga::motor_translate;

    #[test]
    fn capsule_and_sphere_casts_and_aabb_overlap_succeed() {
        let shapes = [ColliderShape3D::SdfAnalytical {
            primitive: SdfPrimitive::Sphere { radius: 1.0 },
        }];
        let motors = [motor_translate([0.0, 0.0, 5.0])];
        let mut bvh = BroadphaseBvh::<16>::new(0.1);
        bvh.insert(0, shapes[0].world_aabb(&motors[0])).unwrap();

        let c_hit = cast_capsule(
            &bvh,
            &shapes,
            &motors,
            [0.0, 0.0, 0.0],
            0.5,
            0.5,
            [0.0, 0.0, 1.0],
            10.0,
            None,
        )
        .unwrap();
        assert!((c_hit.toi - 3.5).abs() < 1e-2);

        let mut overlaps = [u32::MAX; 4];
        let count = overlap_aabb(
            &bvh,
            &Aabb3D::from_center_half_extents([0.0, 0.0, 5.0], [0.5, 0.5, 0.5]),
            &mut overlaps,
        );
        assert_eq!(count, 1);
        assert_eq!(overlaps[0], 0);
    }
}
