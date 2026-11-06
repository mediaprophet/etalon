//! Unified Analytical SDF + B-Rep 3D Physics & ODE Response Engine (`physics3d`).
//!
//! Provides `etalon-sim`'s Tier-1 (`HotZeroHeap`) 3D rigid-body, SDF, joint, fracture,
//! continuum-coupling, and RK4 ODE response physics engine:
//! - [`sdf_collider`]: Exact analytical SDF evaluation (`Sphere`, `RoundedBox`, `Capsule`, `Torus`,
//!   `SmoothUnionPair`) with tetrahedron gradient normals $\nabla\text{sdf}(\mathbf{x})$.
//! - [`broadphase_bvh`], [`query_ray`], & [`query_sweep`]: Dynamic fat-AABB BVH (`BroadphaseBvh`),
//!   `cast_ray` (sphere-tracing + Möller–Trumbore), `cast_sphere`, `cast_capsule`, and `overlap_aabb`.
//! - [`shapes`], [`gjk_epa`], & [`manifold`]: PGA [`Motor`] support mappings, zero-heap GJK/EPA
//!   narrowphase, and persistent 4-point [`ContactManifold3D`] with warm-starting.
//! - [`fracture_csg`]: Principled tensile stress-cleavage dynamic fracture
//!   (`evaluate_and_fracture_body`) conserving linear and angular momentum.
//! - [`xpbd_solver`], [`joints`], & [`ragdoll`]: 64 Hz deterministic XPBD rigid-body solver,
//!   6-DOF PGA [`Motor`] joints (`BallMotorJoint`, `RevoluteMotorJoint`, `FixedMotorJoint`,
//!   `SpringSuspensionJoint`), and `ArticulatedRagdoll`.
//! - [`continuum_bridge`]: Two-way force coupling with geomechanics, hydrodynamics, and
//!   botanical FSPM metamers.
//! - [`response_ode`]: Zero-heap 4D RK4 driven spring (`step_spring`) and thermal relaxation
//!   (`step_thermal`) ODE kernels.

pub mod broadphase_bvh;
pub mod continuum_bridge;
pub mod fracture_csg;
pub mod gjk_epa;
pub mod joints;
pub mod manifold;
pub mod query_ray;
pub mod query_sweep;
pub mod ragdoll;
pub mod response_ode;
pub mod sdf_collider;
pub mod shapes;
pub mod xpbd_solver;

/// Backwards-compatible `query` module alias re-exporting both ray and swept shape queries.
pub mod query {
    pub use super::query_ray::*;
    pub use super::query_sweep::*;
}

pub use broadphase_bvh::{Aabb3D, BroadphaseBvh, BvhLeafEntry};
pub use continuum_bridge::{
    couple_rigid_body_with_botanical_stem, couple_rigid_body_with_hydro,
    couple_rigid_body_with_soil, BotanicalImpactOutcome, HydroCouplingOutcome,
    SoilCouplingOutcome,
};
pub use fracture_csg::{
    evaluate_and_fracture_body, evaluate_stress_cleavage_partition, FractureOutcome3D,
};
pub use gjk_epa::{compute_narrowphase_contact, gjk_intersect, NarrowphaseContact, SupportVertex};
pub use joints::{
    solve_motor_joint_xpbd, BallMotorJoint, FixedMotorJoint, MotorJoint3D, RevoluteMotorJoint,
    SpringSuspensionJoint,
};
pub use manifold::{
    ContactManifold3D, ContactPoint3D, MAX_MANIFOLD_CONTACTS, MAX_MANIFOLD_POINTS,
    PERSISTENT_CONTACT_THRESHOLD_SQ,
};
pub use query_ray::{
    cast_ray, ray_sdf_local, ray_sphere_local, ray_triangle_moller_trumbore, raycast_collider,
    RayHit3D,
};
pub use query_sweep::{cast_capsule, cast_sphere, overlap_aabb, ShapeCastHit3D};
pub use ragdoll::{ArticulatedRagdoll, RagdollLimbBinding, RagdollMode};
pub use response_ode::{
    step_spring, step_thermal, DrivenSpring, ODEFunction4D, ODEState4D, ResponseOdeError,
    ResponseOdeResult, Rk4Integrator4D, SpringState, ThermalTargets,
};
pub use sdf_collider::{evaluate_smooth_minimum_sdf, SdfPrimitive};
pub use shapes::{ColliderShape3D, MAX_CONVEX_HULL_POINTS};
pub use xpbd_solver::{step_xpbd_world, RigidBody3D, RigidBodyType3D};
