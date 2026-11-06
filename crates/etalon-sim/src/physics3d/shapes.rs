//! 3D Collider Shapes, Support Mappings, AABBs & Inertia Tensors (`shapes.rs`).
//!
//! Defines [`ColliderShape3D`] (`Sphere`, `Capsule`, `Cuboid`, `Cylinder`, `ConvexHull`,
//! `SdfAnalytical`) with PGA [`Motor`] support mappings, local/world [`Aabb3D`] evaluation,
//! volume calculation, and principal inverse inertia tensor estimation.

use etalon_render::spatial_math::pga::Motor;

use crate::physics3d::broadphase_bvh::Aabb3D;
use crate::physics3d::sdf_collider::SdfPrimitive;

/// Maximum number of vertices in an inline zero-heap `ConvexHull` collider.
pub const MAX_CONVEX_HULL_POINTS: usize = 16;

/// Unified 3D collider shape supporting analytical primitives, convex hulls, and analytical SDFs.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColliderShape3D {
    /// Sphere of given radius.
    Sphere { radius: f32 },
    /// Vertical Y-axis capsule with hemisphere `radius` and cylinder `half_height`.
    Capsule { radius: f32, half_height: f32 },
    /// Oriented box with local `half_extents` `[hx, hy, hz]`.
    Cuboid { half_extents: [f32; 3] },
    /// Vertical Y-axis cylinder with `radius` and `half_height`.
    Cylinder { radius: f32, half_height: f32 },
    /// Convex hull point cloud in local coordinates (`point_count <= 16`).
    ConvexHull {
        points: [[f32; 3]; MAX_CONVEX_HULL_POINTS],
        point_count: u8,
    },
    /// First-class analytical SDF collider (`HolographicNodeGpu` / `PROC1`).
    SdfAnalytical { primitive: SdfPrimitive },
}

impl Default for ColliderShape3D {
    #[inline]
    fn default() -> Self {
        Self::Sphere { radius: 0.5 }
    }
}

impl ColliderShape3D {
    /// Evaluates the farthest point on the shape along `dir_local` in local coordinates.
    #[inline]
    pub fn support_point_local(&self, dir_local: [f32; 3]) -> [f32; 3] {
        let d = normalize3_or(dir_local, [0.0, 1.0, 0.0]);
        match *self {
            Self::Sphere { radius } => [d[0] * radius, d[1] * radius, d[2] * radius],
            Self::Capsule {
                radius,
                half_height,
            } => {
                let sy = if d[1] >= 0.0 {
                    half_height
                } else {
                    -half_height
                };
                [d[0] * radius, sy + d[1] * radius, d[2] * radius]
            }
            Self::Cuboid { half_extents } => [
                if dir_local[0] >= 0.0 {
                    half_extents[0]
                } else {
                    -half_extents[0]
                },
                if dir_local[1] >= 0.0 {
                    half_extents[1]
                } else {
                    -half_extents[1]
                },
                if dir_local[2] >= 0.0 {
                    half_extents[2]
                } else {
                    -half_extents[2]
                },
            ],
            Self::Cylinder {
                radius,
                half_height,
            } => {
                let radial_len =
                    (dir_local[0] * dir_local[0] + dir_local[2] * dir_local[2]).sqrt();
                let (sx, sz) = if radial_len > 1e-8 {
                    (
                        dir_local[0] / radial_len * radius,
                        dir_local[2] / radial_len * radius,
                    )
                } else {
                    (0.0, 0.0)
                };
                let sy = if dir_local[1] >= 0.0 {
                    half_height
                } else {
                    -half_height
                };
                [sx, sy, sz]
            }
            Self::ConvexHull {
                ref points,
                point_count,
            } => {
                let count = (point_count as usize).min(MAX_CONVEX_HULL_POINTS);
                if count == 0 {
                    return [0.0; 3];
                }
                let mut best_pt = points[0];
                let mut best_dot = dot3(best_pt, dir_local);
                let mut i = 1usize;
                while i < count {
                    let s = dot3(points[i], dir_local);
                    if s > best_dot {
                        best_dot = s;
                        best_pt = points[i];
                    }
                    i += 1;
                }
                best_pt
            }
            Self::SdfAnalytical { ref primitive } => {
                let he = primitive.local_half_extents();
                let max_r = len3(he);
                let probe = [d[0] * max_r, d[1] * max_r, d[2] * max_r];
                let dist = primitive.evaluate_local(probe);
                [
                    probe[0] - d[0] * dist,
                    probe[1] - d[1] * dist,
                    probe[2] - d[2] * dist,
                ]
            }
        }
    }

    /// Evaluates the support point in world coordinates using the shape's PGA [`Motor`].
    #[inline]
    pub fn support_point_world(&self, motor: &Motor, dir_world: [f32; 3]) -> [f32; 3] {
        let inv = motor.reverse();
        let dir_local = inv.transform_vector(dir_world);
        let local_pt = self.support_point_local(dir_local);
        motor.transform_point(local_pt)
    }

    /// Computes the local-space [`Aabb3D`] of this collider shape.
    #[inline]
    pub fn local_aabb(&self) -> Aabb3D {
        self.world_aabb(&Motor::identity())
    }

    /// Computes a world-space [`Aabb3D`] enclosing the transformed shape.
    #[inline]
    pub fn world_aabb(&self, motor: &Motor) -> Aabb3D {
        let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut min = [0.0f32; 3];
        let mut max = [0.0f32; 3];
        let mut a = 0usize;
        while a < 3 {
            let pos_dir = axes[a];
            let neg_dir = [-axes[a][0], -axes[a][1], -axes[a][2]];
            max[a] = self.support_point_world(motor, pos_dir)[a];
            min[a] = self.support_point_world(motor, neg_dir)[a];
            a += 1;
        }
        Aabb3D::new(min, max)
    }

    /// Approximate volume ($m^3$) of the collider in local space (used for mass & Archimedes buoyancy).
    #[inline]
    pub fn volume(&self) -> f32 {
        const PI: f32 = core::f32::consts::PI;
        match *self {
            Self::Sphere { radius } => (4.0 / 3.0) * PI * radius * radius * radius,
            Self::Capsule {
                radius,
                half_height,
            } => (4.0 / 3.0) * PI * radius.powi(3) + PI * radius * radius * (2.0 * half_height),
            Self::Cuboid { half_extents } => {
                8.0 * half_extents[0] * half_extents[1] * half_extents[2]
            }
            Self::Cylinder {
                radius,
                half_height,
            } => PI * radius * radius * (2.0 * half_height),
            Self::ConvexHull { .. } => {
                let aabb = self.local_aabb();
                let he = aabb.half_extents();
                (8.0 * he[0] * he[1] * he[2] * 0.6).max(1e-4)
            }
            Self::SdfAnalytical { ref primitive } => {
                let he = primitive.local_half_extents();
                (4.0 * he[0] * he[1] * he[2]).max(1e-4)
            }
        }
    }

    /// Estimates the principal local inverse inertia diagonal $[I_{xx}^{-1}, I_{yy}^{-1}, I_{zz}^{-1}]$
    /// for a given `mass_kg`.
    #[inline]
    pub fn estimate_inv_inertia_local(&self, mass_kg: f32) -> [f32; 3] {
        if mass_kg <= 1e-6 {
            return [0.0; 3];
        }
        let aabb = self.local_aabb();
        let he = aabb.half_extents();
        let lx = (2.0 * he[0]).max(0.1);
        let ly = (2.0 * he[1]).max(0.1);
        let lz = (2.0 * he[2]).max(0.1);
        let ix = (mass_kg / 12.0) * (ly * ly + lz * lz);
        let iy = (mass_kg / 12.0) * (lx * lx + lz * lz);
        let iz = (mass_kg / 12.0) * (lx * lx + ly * ly);
        [
            1.0 / ix.max(1e-4),
            1.0 / iy.max(1e-4),
            1.0 / iz.max(1e-4),
        ]
    }
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn len3(v: [f32; 3]) -> f32 {
    dot3(v, v).sqrt()
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
