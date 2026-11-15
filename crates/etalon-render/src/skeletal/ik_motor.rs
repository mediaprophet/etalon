//! Zero-Heap Analytical Two-Bone Motor IK & Constrained Motor FABRIK (`ik_motor.rs`).
//!
//! Operates directly on 3D Projective Geometric Algebra (PGA $\mathcal{G}_{3,0,1}$) `Motor`
//! chains with zero heap allocations (`HotZeroHeap`):
//! - [`solve_two_bone_motor_ik_zero_heap`]: closed-form Law of Cosines + pole-target plane
//!   alignment for biped/quadruped limbs and hand grasping.
//! - [`solve_constrained_fabrik_motor_ik_zero_heap`]: stack-resident (`MAX_IK_CHAIN = 16`)
//!   Forward-And-Backward Reaching Inverse Kinematics with spherical cone and hinge joint limits.

use super::{Motor, SkeletalError};

/// Maximum joints in a single constrained FABRIK chain (`HotZeroHeap` stack bound).
pub const MAX_IK_CHAIN: usize = 16;

/// Maximum vertices in a polygonal swing cone on $S^2$ (`HotZeroHeap` stack bound).
pub const MAX_SWING_POLYGON_VERTICES: usize = 16;

/// Maximum joints spanned by a single [`BiarticularLengthConstraint`] (`HotZeroHeap` stack bound).
pub const MAX_BIARTICULAR_JOINTS: usize = 8;

/// Coupled quaternion swing-twist reach cone with optional polygonal swing boundary,
/// swing-dependent twist narrowing $[\tau_{\min}(\mathbf{s}), \tau_{\max}(\mathbf{s})]$,
/// and synovial translation bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwingTwistConeConstraint {
    /// Unit twist axis along the child bone in parent/neutral space.
    pub twist_axis: [f32; 3],
    /// Maximum swing half-angle about local Y (radians, `>= 0`).
    pub max_swing_y_rad: f32,
    /// Maximum swing half-angle about local Z (radians, `>= 0`).
    pub max_swing_z_rad: f32,
    /// Minimum axial twist angle at neutral swing (radians, `<= 0`).
    pub min_twist_rad: f32,
    /// Maximum axial twist angle at neutral swing (radians, `>= 0`).
    pub max_twist_rad: f32,
    /// Fraction in `[0.0, 0.95]` by which twist bounds narrow at normalized swing eccentricity `e = 1`.
    pub twist_coupling_narrowing: f32,
    /// Optional convex polygon vertices `(swing_y_rad, swing_z_rad)` in tangent space when `polygon_vertex_count >= 3`.
    pub polygon_swing_rad: [[f32; 2]; MAX_SWING_POLYGON_VERTICES],
    /// Number of active vertices in `polygon_swing_rad` (`0` uses elliptical `(max_swing_y_rad, max_swing_z_rad)`).
    pub polygon_vertex_count: u8,
    /// Enables clamping of synovial joint translation inside `[translation_min_m, translation_max_m]`.
    pub has_translation_bounds: bool,
    /// Minimum local translation in meters.
    pub translation_min_m: [f32; 3],
    /// Maximum local translation in meters.
    pub translation_max_m: [f32; 3],
}

impl Default for SwingTwistConeConstraint {
    fn default() -> Self {
        Self::elliptical(
            [1.0, 0.0, 0.0],
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::FRAC_PI_2,
            -std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_4,
        )
    }
}

/// Angular constraint applied to a joint during [`solve_constrained_fabrik_motor_ik_zero_heap`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum JointConstraint {
    /// Full $SO(3)$ rotational freedom.
    #[default]
    Unconstrained,
    /// Conical swing limit relative to the parent bone direction.
    ConeLimit {
        /// Maximum half-angle of the cone in radians (`(0, \pi]`).
        max_half_angle_rad: f32,
    },
    /// Single-degree-of-freedom hinge rotation around `axis` (in parent/local space).
    HingeLimit {
        /// Unit hinge axis `[ax, ay, az]`.
        axis: [f32; 3],
        /// Minimum hinge angle in radians.
        min_angle_rad: f32,
        /// Maximum hinge angle in radians.
        max_angle_rad: f32,
    },
    /// Coupled elliptical/polygonal swing-twist cone constraint.
    SwingTwistCone(SwingTwistConeConstraint),
}

/// Output receipt for [`solve_two_bone_motor_ik_zero_heap`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoBoneIkResult {
    /// Updated world-space `Motor`s for `[root, mid, end]`.
    pub world_motors: [Motor; 3],
    /// Updated local-space `Motor`s (`[root_world, root^-1 * mid, mid^-1 * end]`).
    pub local_motors: [Motor; 3],
    /// Interior knee/elbow angle in radians (`\pi` = straight limb).
    pub knee_angle_rad: f32,
    /// Euclidean distance from solved end-effector to `target_pos`.
    pub target_error: f32,
}

#[inline]
pub(crate) fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub(crate) fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

#[inline]
pub(crate) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub(crate) fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub(crate) fn len3(v: [f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}

#[inline]
pub(crate) fn normalize3_or(v: [f64; 3], fallback: [f64; 3]) -> [f64; 3] {
    let l = len3(v);
    if l > 1e-9 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        fallback
    }
}

#[inline]
fn orthogonal_unit(u: [f64; 3]) -> [f64; 3] {
    let ref_axis = if u[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    normalize3_or(cross3(u, ref_axis), [0.0, 0.0, 1.0])
}

/// Shortest-arc unit quaternion `[w, x, y, z]` rotating unit vector `from` to unit vector `to`.
#[inline]
fn quat_from_unit_vecs(from: [f64; 3], to: [f64; 3]) -> [f64; 4] {
    let d = dot3(from, to).clamp(-1.0, 1.0);
    if d > 1.0 - 1e-9 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    if d < -1.0 + 1e-9 {
        let axis = orthogonal_unit(from);
        return [0.0, axis[0], axis[1], axis[2]];
    }
    let c = cross3(from, to);
    let w = ((1.0 + d) * 0.5).sqrt();
    let inv = 1.0 / (2.0 * w);
    let q = [w, c[0] * inv, c[1] * inv, c[2] * inv];
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

#[inline]
fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

#[inline]
fn rotate_vec_by_axis_angle(v: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let (s, c) = angle.sin_cos();
    let k_dot_v = dot3(axis, v);
    let k_cross_v = cross3(axis, v);
    [
        v[0] * c + k_cross_v[0] * s + axis[0] * k_dot_v * (1.0 - c),
        v[1] * c + k_cross_v[1] * s + axis[1] * k_dot_v * (1.0 - c),
        v[2] * c + k_cross_v[2] * s + axis[2] * k_dot_v * (1.0 - c),
    ]
}

/// Analytically solves two-bone inverse kinematics (e.g., hip–knee–ankle or shoulder–elbow–wrist)
/// directly on PGA `Motor`s using the Law of Cosines and a pole-vector bend plane (`HotZeroHeap`).
pub fn solve_two_bone_motor_ik_zero_heap(
    root_world: Motor,
    mid_world: Motor,
    end_world: Motor,
    l1: f32,
    l2: f32,
    target_pos: [f32; 3],
    pole_target: [f32; 3],
    min_knee_angle_rad: f32,
    max_knee_angle_rad: f32,
) -> Result<TwoBoneIkResult, SkeletalError> {
    if target_pos
        .iter()
        .chain(pole_target.iter())
        .any(|v| !v.is_finite())
    {
        return Err(SkeletalError::InvalidInput);
    }

    let (root_r0, p_root) = root_world.to_rotation_translation();
    let (mid_r0, p_mid0) = mid_world.to_rotation_translation();
    let (end_r0, p_end0) = end_world.to_rotation_translation();

    let len1 = if l1 > 1e-5 && l1.is_finite() {
        l1 as f64
    } else {
        len3(sub3(p_mid0, p_root)).max(1e-4)
    };
    let len2 = if l2 > 1e-5 && l2.is_finite() {
        l2 as f64
    } else {
        len3(sub3(p_end0, p_mid0)).max(1e-4)
    };

    let p_target = [
        target_pos[0] as f64,
        target_pos[1] as f64,
        target_pos[2] as f64,
    ];
    let p_pole = [
        pole_target[0] as f64,
        pole_target[1] as f64,
        pole_target[2] as f64,
    ];

    let min_angle = (min_knee_angle_rad as f64).clamp(0.01, std::f64::consts::PI - 1e-4);
    let max_angle = (max_knee_angle_rad as f64).clamp(min_angle, std::f64::consts::PI);

    // Distance bounds implied by interior knee angle [min_angle, max_angle]:
    // D^2 = L1^2 + L2^2 - 2 L1 L2 cos(theta_interior)
    let min_reach = (len1 * len1 + len2 * len2 - 2.0 * len1 * len2 * min_angle.cos())
        .max(1e-8)
        .sqrt();
    let max_reach = (len1 * len1 + len2 * len2 - 2.0 * len1 * len2 * max_angle.cos())
        .max(min_reach * min_reach)
        .sqrt();

    let to_target = sub3(p_target, p_root);
    let raw_dist = len3(to_target);
    let u_dir = normalize3_or(to_target, [0.0, -1.0, 0.0]);
    let clamped_dist = raw_dist.clamp(min_reach, max_reach);

    // Law of cosines for root elevation angle alpha
    let cos_alpha = ((len1 * len1 + clamped_dist * clamped_dist - len2 * len2)
        / (2.0 * len1 * clamped_dist))
        .clamp(-1.0, 1.0);
    let sin_alpha = (1.0 - cos_alpha * cos_alpha).max(0.0).sqrt();

    // Interior knee angle theta
    let cos_theta = ((len1 * len1 + len2 * len2 - clamped_dist * clamped_dist)
        / (2.0 * len1 * len2))
        .clamp(-1.0, 1.0);
    let knee_angle = cos_theta.acos() as f32;

    // Pole plane direction perpendicular to u_dir
    let to_pole = sub3(p_pole, p_root);
    let pole_perp = sub3(to_pole, scale3(u_dir, dot3(to_pole, u_dir)));
    let n_pole = normalize3_or(pole_perp, orthogonal_unit(u_dir));

    // Solved mid (knee/elbow) and end (ankle/wrist) positions
    let p_mid_new = add3(
        p_root,
        add3(
            scale3(u_dir, len1 * cos_alpha),
            scale3(n_pole, len1 * sin_alpha),
        ),
    );
    let lower_dir = normalize3_or(sub3(p_target, p_mid_new), u_dir);
    let p_end_new = add3(p_mid_new, scale3(lower_dir, len2));

    // Update world orientations using shortest-arc rotors from initial bone directions
    let init_upper_dir = normalize3_or(sub3(p_mid0, p_root), [0.0, -1.0, 0.0]);
    let new_upper_dir = normalize3_or(sub3(p_mid_new, p_root), init_upper_dir);
    let q_upper = quat_from_unit_vecs(init_upper_dir, new_upper_dir);
    let new_root_r = quat_mul(q_upper, root_r0);

    let init_lower_dir = normalize3_or(sub3(p_end0, p_mid0), init_upper_dir);
    let q_lower = quat_from_unit_vecs(init_lower_dir, lower_dir);
    let new_mid_r = quat_mul(q_lower, mid_r0);

    let m_root_new = Motor::from_rotation_translation(new_root_r, p_root);
    let m_mid_new = Motor::from_rotation_translation(new_mid_r, p_mid_new);
    let m_end_new = Motor::from_rotation_translation(end_r0, p_end_new);

    let local_root = m_root_new;
    let local_mid = m_root_new.conjugate().mul(&m_mid_new).normalize();
    let local_end = m_mid_new.conjugate().mul(&m_end_new).normalize();

    let target_error = len3(sub3(p_end_new, p_target)) as f32;

    Ok(TwoBoneIkResult {
        world_motors: [m_root_new, m_mid_new, m_end_new],
        local_motors: [local_root, local_mid, local_end],
        knee_angle_rad: knee_angle,
        target_error,
    })
}

/// Applies `constraint` to a child bone unit direction `dir` relative to `parent_dir`.
#[inline]
fn apply_joint_constraint_dir(
    dir: [f64; 3],
    parent_dir: [f64; 3],
    constraint: JointConstraint,
) -> [f64; 3] {
    match constraint {
        JointConstraint::Unconstrained => dir,
        JointConstraint::ConeLimit { max_half_angle_rad } => {
            let max_angle = (max_half_angle_rad as f64).clamp(1e-4, std::f64::consts::PI);
            let cos_angle = dot3(parent_dir, dir).clamp(-1.0, 1.0);
            let angle = cos_angle.acos();
            if angle <= max_angle {
                return dir;
            }
            let axis = normalize3_or(cross3(parent_dir, dir), orthogonal_unit(parent_dir));
            normalize3_or(
                rotate_vec_by_axis_angle(parent_dir, axis, max_angle),
                parent_dir,
            )
        }
        JointConstraint::HingeLimit {
            axis,
            min_angle_rad,
            max_angle_rad,
        } => {
            let h_axis = normalize3_or(
                [axis[0] as f64, axis[1] as f64, axis[2] as f64],
                [1.0, 0.0, 0.0],
            );
            let ref_proj = sub3(parent_dir, scale3(h_axis, dot3(parent_dir, h_axis)));
            let ref_dir = normalize3_or(ref_proj, orthogonal_unit(h_axis));

            let cand_proj = sub3(dir, scale3(h_axis, dot3(dir, h_axis)));
            let cand_dir = normalize3_or(cand_proj, ref_dir);

            let sin_a = dot3(cross3(ref_dir, cand_dir), h_axis);
            let cos_a = dot3(ref_dir, cand_dir);
            let angle = sin_a.atan2(cos_a);
            let lo = (min_angle_rad as f64).min(max_angle_rad as f64);
            let hi = (min_angle_rad as f64).max(max_angle_rad as f64);
            let clamped = angle.clamp(lo, hi);
            normalize3_or(rotate_vec_by_axis_angle(ref_dir, h_axis, clamped), ref_dir)
        }
        JointConstraint::SwingTwistCone(cone) => {
            let t_axis = normalize3_or(
                [
                    cone.twist_axis[0] as f64,
                    cone.twist_axis[1] as f64,
                    cone.twist_axis[2] as f64,
                ],
                [1.0, 0.0, 0.0],
            );
            let q_align = quat_from_unit_vecs(parent_dir, t_axis);
            let local_dir = rotate_vec_by_quat(dir, q_align);
            let q_swing = quat_from_unit_vecs(t_axis, local_dir);
            let q_proj = cone.project_quaternion(
                [
                    q_swing[0] as f32,
                    q_swing[1] as f32,
                    q_swing[2] as f32,
                    q_swing[3] as f32,
                ],
                1.0,
            );
            let q_proj_f64 = [
                q_proj[0] as f64,
                q_proj[1] as f64,
                q_proj[2] as f64,
                q_proj[3] as f64,
            ];
            let clamped_local = rotate_vec_by_quat(t_axis, q_proj_f64);
            normalize3_or(
                rotate_vec_by_quat(clamped_local, quat_conj_f64(q_align)),
                parent_dir,
            )
        }
    }
}

/// Solves multi-joint Constrained Forward-And-Backward Reaching Inverse Kinematics (FABRIK)
/// directly on a chain of up to `MAX_IK_CHAIN = 16` PGA `Motor`s with zero heap allocations (`HotZeroHeap`).
pub fn solve_constrained_fabrik_motor_ik_zero_heap(
    chain_world_motors: &[Motor],
    constraints: &[JointConstraint],
    target_pos: [f32; 3],
    tolerance: f32,
    max_iterations: usize,
    out_world_motors: &mut [Motor],
    out_local_motors: &mut [Motor],
) -> Result<f32, SkeletalError> {
    let n = chain_world_motors.len();
    if n < 2 || n > MAX_IK_CHAIN || target_pos.iter().any(|v| !v.is_finite()) {
        return Err(SkeletalError::InvalidInput);
    }
    if out_world_motors.len() < n || out_local_motors.len() < n {
        return Err(SkeletalError::OutputTooSmall);
    }

    let mut pos = [[0.0f64; 3]; MAX_IK_CHAIN];
    let mut init_pos = [[0.0f64; 3]; MAX_IK_CHAIN];
    let mut init_rot = [[1.0f64, 0.0, 0.0, 0.0]; MAX_IK_CHAIN];
    let mut lengths = [0.0f64; MAX_IK_CHAIN];

    for i in 0..n {
        let (r, t) = chain_world_motors[i].to_rotation_translation();
        pos[i] = t;
        init_pos[i] = t;
        init_rot[i] = r;
    }
    for i in 0..(n - 1) {
        lengths[i] = len3(sub3(pos[i + 1], pos[i])).max(1e-5);
    }

    let root_anchor = pos[0];
    let root_parent_dir = normalize3_or(sub3(init_pos[1], init_pos[0]), [0.0, 1.0, 0.0]);
    let target = [
        target_pos[0] as f64,
        target_pos[1] as f64,
        target_pos[2] as f64,
    ];
    let tol = (tolerance.max(1e-5)) as f64;
    let iters = max_iterations.clamp(1, 64);

    for _ in 0..iters {
        let err = len3(sub3(pos[n - 1], target));
        if err <= tol {
            break;
        }

        // 1. Backward reaching pass: pull from end-effector to root
        pos[n - 1] = target;
        for i in (0..(n - 1)).rev() {
            let dir = normalize3_or(sub3(pos[i], pos[i + 1]), [0.0, -1.0, 0.0]);
            pos[i] = add3(pos[i + 1], scale3(dir, lengths[i]));
        }

        // 2. Forward reaching pass: anchor root and enforce JointConstraint per segment
        pos[0] = root_anchor;
        let mut parent_dir = root_parent_dir;
        for i in 0..(n - 1) {
            let raw_dir = normalize3_or(sub3(pos[i + 1], pos[i]), parent_dir);
            let constraint = constraints
                .get(i)
                .copied()
                .unwrap_or(JointConstraint::Unconstrained);
            let constrained_dir = apply_joint_constraint_dir(raw_dir, parent_dir, constraint);
            pos[i + 1] = add3(pos[i], scale3(constrained_dir, lengths[i]));
            parent_dir = constrained_dir;
        }
    }

    // Reconstruct world and local Motors from solved joint positions
    for i in 0..(n - 1) {
        let old_dir = normalize3_or(sub3(init_pos[i + 1], init_pos[i]), [0.0, 1.0, 0.0]);
        let new_dir = normalize3_or(sub3(pos[i + 1], pos[i]), old_dir);
        let dq = quat_from_unit_vecs(old_dir, new_dir);
        let new_r = quat_mul(dq, init_rot[i]);
        out_world_motors[i] = Motor::from_rotation_translation(new_r, pos[i]);
    }
    out_world_motors[n - 1] = Motor::from_rotation_translation(init_rot[n - 1], pos[n - 1]);

    out_local_motors[0] = out_world_motors[0];
    for i in 1..n {
        out_local_motors[i] = out_world_motors[i - 1]
            .conjugate()
            .mul(&out_world_motors[i])
            .normalize();
    }

    Ok(len3(sub3(pos[n - 1], target)) as f32)
}

#[inline]
fn quat_conj_f64(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

#[inline]
fn quat_normalize_f64(q: [f64; 4]) -> [f64; 4] {
    if q.iter().any(|v| !v.is_finite()) {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if norm_sq < 1e-12 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let inv = 1.0 / norm_sq.sqrt();
    [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
}

#[inline]
fn quat_to_f32_norm(q: [f64; 4]) -> [f32; 4] {
    let n = quat_normalize_f64(q);
    [n[0] as f32, n[1] as f32, n[2] as f32, n[3] as f32]
}

#[inline]
fn quat_from_axis_angle_f64(axis: [f64; 3], angle_rad: f64) -> [f64; 4] {
    let (s, c) = (0.5 * angle_rad).sin_cos();
    [c, axis[0] * s, axis[1] * s, axis[2] * s]
}

#[inline]
fn rotate_vec_by_quat(v: [f64; 3], q: [f64; 4]) -> [f64; 3] {
    let u = [q[1], q[2], q[3]];
    let t = scale3(cross3(u, v), 2.0);
    add3(add3(v, scale3(t, q[0])), cross3(u, t))
}

#[inline]
fn orthonormal_swing_basis(axis: [f64; 3]) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let ref_v = if axis[1].abs() < 0.99 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let c = normalize3_or(cross3(axis, ref_v), [0.0, 0.0, 1.0]);
    let b = normalize3_or(cross3(c, axis), [0.0, 1.0, 0.0]);
    (axis, b, c)
}

/// Decomposes unit quaternion `q` (`[w, x, y, z]`) into `(swing, twist)` around `twist_axis`
/// such that `quat_mul(swing, twist) == normalize(q)` and `twist` is a pure rotation around
/// `twist_axis` in the shortest-arc hemisphere (`twist[0] >= 0`).
pub fn decompose_swing_twist(q: [f32; 4], twist_axis: [f32; 3]) -> ([f32; 4], [f32; 4]) {
    let qn = quat_normalize_f64([q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64]);
    let axis_raw = [
        twist_axis[0] as f64,
        twist_axis[1] as f64,
        twist_axis[2] as f64,
    ];
    let axis = if axis_raw.iter().all(|v| v.is_finite()) {
        normalize3_or(axis_raw, [1.0, 0.0, 0.0])
    } else {
        [1.0, 0.0, 0.0]
    };

    let proj = qn[1] * axis[0] + qn[2] * axis[1] + qn[3] * axis[2];
    let twist_sq = qn[0] * qn[0] + proj * proj;
    if twist_sq < 1e-12 {
        return (quat_to_f32_norm(qn), [1.0, 0.0, 0.0, 0.0]);
    }

    let inv = 1.0 / twist_sq.sqrt();
    let mut twist = [
        qn[0] * inv,
        axis[0] * proj * inv,
        axis[1] * proj * inv,
        axis[2] * proj * inv,
    ];
    if twist[0] < 0.0 {
        twist = [-twist[0], -twist[1], -twist[2], -twist[3]];
    }
    let swing = quat_mul(qn, quat_conj_f64(twist));
    (quat_to_f32_norm(swing), quat_to_f32_norm(twist))
}

#[inline]
fn project_point_onto_ellipse_2d(px: f64, py: f64, a: f64, b: f64) -> (f64, f64) {
    if a <= 1e-12 && b <= 1e-12 {
        return (0.0, 0.0);
    }
    if a <= 1e-12 {
        return (0.0, py.clamp(-b, b));
    }
    if b <= 1e-12 {
        return (px.clamp(-a, a), 0.0);
    }
    let norm_sq = (px / a) * (px / a) + (py / b) * (py / b);
    if norm_sq <= 1.0 {
        return (px, py);
    }

    // Closest point on ellipse in first quadrant via strictly decreasing F(t).
    let qx = px.abs();
    let qy = py.abs();
    let a2 = a * a;
    let b2 = b * b;
    let mut lo = 0.0_f64;
    let mut hi = (a * qx).max(b * qy) * std::f64::consts::SQRT_2 + 1.0;

    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        let rx = (a * qx) / (mid + a2);
        let ry = (b * qy) / (mid + b2);
        let f = rx * rx + ry * ry - 1.0;
        if f.abs() < 1e-13 {
            lo = mid;
            break;
        }
        if f > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = 0.5 * (lo + hi);
    let x_out = (a2 * qx) / (t + a2);
    let y_out = (b2 * qy) / (t + b2);
    (x_out.copysign(px), y_out.copysign(py))
}

#[inline]
fn point_in_convex_polygon_2d(px: f64, py: f64, verts: &[[f32; 2]]) -> bool {
    let n = verts.len();
    if n < 3 {
        return false;
    }
    let mut pos = false;
    let mut neg = false;
    for i in 0..n {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        let ex = (b[0] - a[0]) as f64;
        let ey = (b[1] - a[1]) as f64;
        let dx = px - a[0] as f64;
        let dy = py - a[1] as f64;
        let cross = ex * dy - ey * dx;
        if cross > 1e-12 {
            pos = true;
        } else if cross < -1e-12 {
            neg = true;
        }
        if pos && neg {
            return false;
        }
    }
    true
}

#[inline]
fn project_point_onto_convex_polygon_2d(px: f64, py: f64, verts: &[[f32; 2]]) -> (f64, f64) {
    if point_in_convex_polygon_2d(px, py, verts) {
        return (px, py);
    }
    let n = verts.len();
    let mut best_dist_sq = f64::INFINITY;
    let mut best = (0.0_f64, 0.0_f64);
    for i in 0..n {
        let a = [verts[i][0] as f64, verts[i][1] as f64];
        let b = [verts[(i + 1) % n][0] as f64, verts[(i + 1) % n][1] as f64];
        let abx = b[0] - a[0];
        let aby = b[1] - a[1];
        let len_sq = abx * abx + aby * aby;
        let t = if len_sq > 1e-12 {
            (((px - a[0]) * abx + (py - a[1]) * aby) / len_sq).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let cx = a[0] + t * abx;
        let cy = a[1] + t * aby;
        let d_sq = (px - cx) * (px - cx) + (py - cy) * (py - cy);
        if d_sq < best_dist_sq {
            best_dist_sq = d_sq;
            best = (cx, cy);
        }
    }
    best
}

/// Projects quaternion `q` (`[w, x, y, z]`) onto an elliptical swing-twist reach cone
/// around `twist_axis`, scaled by `laxity_scale`.
pub fn project_swing_twist_cone(
    q: [f32; 4],
    twist_axis: [f32; 3],
    max_swing_y_rad: f32,
    max_swing_z_rad: f32,
    min_twist_rad: f32,
    max_twist_rad: f32,
    laxity_scale: f32,
) -> [f32; 4] {
    SwingTwistConeConstraint::elliptical(
        twist_axis,
        max_swing_y_rad,
        max_swing_z_rad,
        min_twist_rad,
        max_twist_rad,
    )
    .project_quaternion(q, laxity_scale)
}

impl SwingTwistConeConstraint {
    /// Creates an elliptical swing-twist reach cone constraint.
    pub fn elliptical(
        twist_axis: [f32; 3],
        max_swing_y_rad: f32,
        max_swing_z_rad: f32,
        min_twist_rad: f32,
        max_twist_rad: f32,
    ) -> Self {
        let lo = min_twist_rad.min(max_twist_rad);
        let hi = min_twist_rad.max(max_twist_rad);
        Self {
            twist_axis,
            max_swing_y_rad: max_swing_y_rad.abs(),
            max_swing_z_rad: max_swing_z_rad.abs(),
            min_twist_rad: lo,
            max_twist_rad: hi,
            twist_coupling_narrowing: 0.0,
            polygon_swing_rad: [[0.0; 2]; MAX_SWING_POLYGON_VERTICES],
            polygon_vertex_count: 0,
            has_translation_bounds: false,
            translation_min_m: [0.0; 3],
            translation_max_m: [0.0; 3],
        }
    }

    /// Sets the swing-coupled twist narrowing fraction in `[0.0, 0.95]`.
    pub fn with_twist_coupling(mut self, narrowing: f32) -> Self {
        self.twist_coupling_narrowing = if narrowing.is_finite() {
            narrowing.clamp(0.0, 0.95)
        } else {
            0.0
        };
        self
    }

    /// Replaces the elliptical swing boundary with a convex polygon in `(swing_y, swing_z)` radians.
    pub fn with_polygon_swing(mut self, vertices: &[[f32; 2]]) -> Result<Self, SkeletalError> {
        if vertices.len() < 3
            || vertices.len() > MAX_SWING_POLYGON_VERTICES
            || vertices
                .iter()
                .any(|v| !v[0].is_finite() || !v[1].is_finite())
        {
            return Err(SkeletalError::InvalidInput);
        }
        self.polygon_swing_rad = [[0.0; 2]; MAX_SWING_POLYGON_VERTICES];
        for (i, v) in vertices.iter().enumerate() {
            self.polygon_swing_rad[i] = *v;
        }
        self.polygon_vertex_count = vertices.len() as u8;
        Ok(self)
    }

    /// Enables synovial joint translation bounds in meters.
    pub fn with_translation_bounds(mut self, min_m: [f32; 3], max_m: [f32; 3]) -> Self {
        self.has_translation_bounds = true;
        for i in 0..3 {
            self.translation_min_m[i] = min_m[i].min(max_m[i]);
            self.translation_max_m[i] = min_m[i].max(max_m[i]);
        }
        self
    }

    /// Computes effective `(min_twist_rad, max_twist_rad)` at normalized swing eccentricity `e in [0, 1]`.
    pub fn swing_coupled_twist_bounds(
        &self,
        swing_eccentricity_01: f32,
        laxity_scale: f32,
    ) -> (f32, f32) {
        let lax = if laxity_scale.is_finite() {
            laxity_scale.max(0.0)
        } else {
            1.0
        };
        let e = if swing_eccentricity_01.is_finite() {
            swing_eccentricity_01.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let k = if self.twist_coupling_narrowing.is_finite() {
            self.twist_coupling_narrowing.clamp(0.0, 0.95)
        } else {
            0.0
        };
        let scale = lax * (1.0 - k * e);
        let lo = (self.min_twist_rad.min(self.max_twist_rad) * scale)
            .clamp(-std::f32::consts::PI, std::f32::consts::PI);
        let hi = (self.min_twist_rad.max(self.max_twist_rad) * scale)
            .clamp(-std::f32::consts::PI, std::f32::consts::PI);
        (lo, hi)
    }

    /// Projects quaternion `q` (`[w, x, y, z]`) onto this swing-twist cone with zero heap allocation.
    pub fn project_quaternion(&self, q: [f32; 4], laxity_scale: f32) -> [f32; 4] {
        let lax = if laxity_scale.is_finite() {
            laxity_scale.max(0.0) as f64
        } else {
            1.0
        };
        let axis_raw = [
            self.twist_axis[0] as f64,
            self.twist_axis[1] as f64,
            self.twist_axis[2] as f64,
        ];
        let axis = if axis_raw.iter().all(|v| v.is_finite()) {
            normalize3_or(axis_raw, [1.0, 0.0, 0.0])
        } else {
            [1.0, 0.0, 0.0]
        };
        let (_, b, c) = orthonormal_swing_basis(axis);

        let (swing_f32, twist_f32) = decompose_swing_twist(q, self.twist_axis);
        let mut swing = [
            swing_f32[0] as f64,
            swing_f32[1] as f64,
            swing_f32[2] as f64,
            swing_f32[3] as f64,
        ];
        if swing[0] < 0.0 {
            swing = [-swing[0], -swing[1], -swing[2], -swing[3]];
        }
        let twist = [
            twist_f32[0] as f64,
            twist_f32[1] as f64,
            twist_f32[2] as f64,
            twist_f32[3] as f64,
        ];

        let vs = [swing[1], swing[2], swing[3]];
        let vb = dot3(vs, b);
        let vc = dot3(vs, c);
        let v_norm = (vb * vb + vc * vc).sqrt();
        let (phi_y, phi_z) = if v_norm > 1e-12 {
            let phi = 2.0 * v_norm.atan2(swing[0]);
            (phi * (vb / v_norm), phi * (vc / v_norm))
        } else {
            (0.0, 0.0)
        };

        let eff_y = ((self.max_swing_y_rad.abs() as f64) * lax).clamp(0.0, std::f64::consts::PI);
        let eff_z = ((self.max_swing_z_rad.abs() as f64) * lax).clamp(0.0, std::f64::consts::PI);

        let (clamped_y, clamped_z, ecc_01) = if self.polygon_vertex_count >= 3 {
            let n = (self.polygon_vertex_count as usize).min(MAX_SWING_POLYGON_VERTICES);
            let mut scaled_verts = [[0.0f32; 2]; MAX_SWING_POLYGON_VERTICES];
            let mut max_r = 1e-6_f64;
            for i in 0..n {
                let vx = (self.polygon_swing_rad[i][0] as f64 * lax)
                    .clamp(-std::f64::consts::PI, std::f64::consts::PI);
                let vy = (self.polygon_swing_rad[i][1] as f64 * lax)
                    .clamp(-std::f64::consts::PI, std::f64::consts::PI);
                max_r = max_r.max((vx * vx + vy * vy).sqrt());
                scaled_verts[i] = [vx as f32, vy as f32];
            }
            let (py, pz) = project_point_onto_convex_polygon_2d(phi_y, phi_z, &scaled_verts[..n]);
            let r = (py * py + pz * pz).sqrt();
            (py, pz, (r / max_r).clamp(0.0, 1.0))
        } else {
            let (py, pz) = project_point_onto_ellipse_2d(phi_y, phi_z, eff_y, eff_z);
            let e = match (eff_y > 1e-9, eff_z > 1e-9) {
                (true, true) => ((py / eff_y).powi(2) + (pz / eff_z).powi(2)).sqrt(),
                (true, false) => (py / eff_y).abs(),
                (false, true) => (pz / eff_z).abs(),
                (false, false) => 0.0,
            }
            .clamp(0.0, 1.0);
            (py, pz, e)
        };

        let swing_angle_clamped = (clamped_y * clamped_y + clamped_z * clamped_z).sqrt();
        let swing_clamped = if swing_angle_clamped > 1e-12 {
            let swing_axis = add3(
                scale3(b, clamped_y / swing_angle_clamped),
                scale3(c, clamped_z / swing_angle_clamped),
            );
            quat_from_axis_angle_f64(swing_axis, swing_angle_clamped)
        } else {
            [1.0, 0.0, 0.0, 0.0]
        };

        let vt = [twist[1], twist[2], twist[3]];
        let twist_angle = 2.0 * dot3(vt, axis).atan2(twist[0]);
        let (eff_min_twist, eff_max_twist) =
            self.swing_coupled_twist_bounds(ecc_01 as f32, laxity_scale);
        let twist_clamped_angle = twist_angle.clamp(eff_min_twist as f64, eff_max_twist as f64);
        let twist_clamped = quat_from_axis_angle_f64(axis, twist_clamped_angle);

        quat_to_f32_norm(quat_mul(swing_clamped, twist_clamped))
    }

    /// Returns `true` if `q` lies within `tol_rad` geodesic distance of its projection.
    pub fn is_quaternion_within(&self, q: [f32; 4], laxity_scale: f32, tol_rad: f32) -> bool {
        let qn = quat_normalize_f64([q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64]);
        let qp_f32 = self.project_quaternion(q, laxity_scale);
        let qp = quat_normalize_f64([
            qp_f32[0] as f64,
            qp_f32[1] as f64,
            qp_f32[2] as f64,
            qp_f32[3] as f64,
        ]);
        let dot = qn[0] * qp[0] + qn[1] * qp[1] + qn[2] * qp[2] + qn[3] * qp[3];
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        let mut diff_sq = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for i in 0..4 {
            let b = sign * qp[i];
            let d = qn[i] - b;
            let s = qn[i] + b;
            diff_sq += d * d;
            sum_sq += s * s;
        }
        let geodesic = 4.0 * diff_sq.sqrt().atan2(sum_sq.sqrt());
        geodesic <= (tol_rad.max(0.0) as f64)
    }

    /// Projects a local PGA `Motor` onto this swing-twist cone and optional synovial translation bounds.
    pub fn project_motor(&self, local_motor: Motor, laxity_scale: f32) -> Motor {
        let (rot, mut trans) = local_motor.to_rotation_translation();
        let proj = self.project_quaternion(
            [rot[0] as f32, rot[1] as f32, rot[2] as f32, rot[3] as f32],
            laxity_scale,
        );
        if self.has_translation_bounds {
            let lax = if laxity_scale.is_finite() {
                laxity_scale.max(0.0) as f64
            } else {
                1.0
            };
            for i in 0..3 {
                let lo = (self.translation_min_m[i] as f64) * lax;
                let hi = (self.translation_max_m[i] as f64) * lax;
                trans[i] = trans[i].clamp(lo.min(hi), lo.max(hi));
            }
        }
        Motor::from_rotation_translation(
            [
                proj[0] as f64,
                proj[1] as f64,
                proj[2] as f64,
                proj[3] as f64,
            ],
            trans,
        )
    }
}

/// Coupled glenohumeral-to-scapulothoracic rhythm (2:1 GH:ST above 30 deg elevation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScapuloHumeralCoupling {
    /// Setting-phase elevation threshold below which scapular rotation is zero (default: 30 deg).
    pub setting_threshold_rad: f32,
    /// Glenohumeral-to-scapulothoracic incremental ratio above threshold (default: 2.0 for 2:1).
    pub gh_to_st_ratio: f32,
    /// Unit axis of scapulothoracic upward rotation in thorax frame.
    pub scapular_upward_axis: [f32; 3],
    /// Unit axis of humeral elevation in scapular frame.
    pub humeral_elevation_axis: [f32; 3],
    /// Unit shaft axis of the resting humerus (orthogonal to `humeral_elevation_axis`).
    pub humeral_shaft_axis: [f32; 3],
    /// Maximum scapulothoracic upward rotation in radians (default: 60 deg).
    pub max_scapular_elevation_rad: f32,
}

impl Default for ScapuloHumeralCoupling {
    fn default() -> Self {
        Self {
            setting_threshold_rad: std::f32::consts::PI / 6.0,
            gh_to_st_ratio: 2.0,
            scapular_upward_axis: [0.0, 0.0, 1.0],
            humeral_elevation_axis: [0.0, 0.0, 1.0],
            humeral_shaft_axis: [0.0, -1.0, 0.0],
            max_scapular_elevation_rad: std::f32::consts::PI / 3.0,
        }
    }
}

impl ScapuloHumeralCoupling {
    /// Constructs a scapulohumeral rhythm coupler with normalized axes and orthogonal shaft frame.
    pub fn new(
        setting_threshold_rad: f32,
        gh_to_st_ratio: f32,
        scapular_upward_axis: [f32; 3],
        humeral_elevation_axis: [f32; 3],
    ) -> Self {
        let elev = normalize3_or(
            [
                humeral_elevation_axis[0] as f64,
                humeral_elevation_axis[1] as f64,
                humeral_elevation_axis[2] as f64,
            ],
            [0.0, 0.0, 1.0],
        );
        let scap = normalize3_or(
            [
                scapular_upward_axis[0] as f64,
                scapular_upward_axis[1] as f64,
                scapular_upward_axis[2] as f64,
            ],
            elev,
        );
        let shaft_ref = if elev[1].abs() < 0.9 {
            [0.0, -1.0, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        let shaft = normalize3_or(
            sub3(shaft_ref, scale3(elev, dot3(shaft_ref, elev))),
            orthogonal_unit(elev),
        );
        Self {
            setting_threshold_rad: if setting_threshold_rad.is_finite() {
                setting_threshold_rad.max(0.0)
            } else {
                std::f32::consts::PI / 6.0
            },
            gh_to_st_ratio: if gh_to_st_ratio.is_finite() && gh_to_st_ratio > 0.0 {
                gh_to_st_ratio
            } else {
                2.0
            },
            scapular_upward_axis: [scap[0] as f32, scap[1] as f32, scap[2] as f32],
            humeral_elevation_axis: [elev[0] as f32, elev[1] as f32, elev[2] as f32],
            humeral_shaft_axis: [shaft[0] as f32, shaft[1] as f32, shaft[2] as f32],
            max_scapular_elevation_rad: std::f32::consts::PI / 3.0,
        }
    }

    /// Partitions `total_elevation_rad` into `(glenohumeral_rad, scapulothoracic_rad)`.
    pub fn partition_elevation_rad(&self, total_elevation_rad: f32) -> (f32, f32) {
        if !total_elevation_rad.is_finite() {
            return (0.0, 0.0);
        }
        let thresh = self.setting_threshold_rad.max(0.0);
        if total_elevation_rad <= thresh {
            return (total_elevation_rad, 0.0);
        }
        let excess = total_elevation_rad - thresh;
        let ratio = if self.gh_to_st_ratio.is_finite() && self.gh_to_st_ratio > 0.0 {
            self.gh_to_st_ratio
        } else {
            2.0
        };
        let raw_st = excess / (1.0 + ratio);
        let st = raw_st.min(self.max_scapular_elevation_rad.max(0.0));
        let gh = total_elevation_rad - st;
        (gh, st)
    }

    /// Decomposes desired thorax-relative humeral orientation `total_humeral_q` (`[w, x, y, z]`)
    /// into `(scapulothoracic_q, glenohumeral_q)` satisfying `st_q * gh_q == total_humeral_q`.
    pub fn solve_coupled_quaternions(&self, total_humeral_q: [f32; 4]) -> ([f32; 4], [f32; 4]) {
        let qn = quat_normalize_f64([
            total_humeral_q[0] as f64,
            total_humeral_q[1] as f64,
            total_humeral_q[2] as f64,
            total_humeral_q[3] as f64,
        ]);
        let (swing_f32, _) = decompose_swing_twist(
            [qn[0] as f32, qn[1] as f32, qn[2] as f32, qn[3] as f32],
            self.humeral_shaft_axis,
        );
        let mut swing = [
            swing_f32[0] as f64,
            swing_f32[1] as f64,
            swing_f32[2] as f64,
            swing_f32[3] as f64,
        ];
        if swing[0] < 0.0 {
            swing = [-swing[0], -swing[1], -swing[2], -swing[3]];
        }
        let vs_norm = (swing[1] * swing[1] + swing[2] * swing[2] + swing[3] * swing[3]).sqrt();
        let elev_angle = if vs_norm > 1e-12 {
            (2.0 * vs_norm.atan2(swing[0])) as f32
        } else {
            0.0
        };
        let (_, st_angle) = self.partition_elevation_rad(elev_angle);
        let scap_axis = normalize3_or(
            [
                self.scapular_upward_axis[0] as f64,
                self.scapular_upward_axis[1] as f64,
                self.scapular_upward_axis[2] as f64,
            ],
            [0.0, 0.0, 1.0],
        );
        let st_q64 = quat_from_axis_angle_f64(scap_axis, st_angle as f64);
        let gh_q64 = quat_mul(quat_conj_f64(st_q64), qn);
        (quat_to_f32_norm(st_q64), quat_to_f32_norm(gh_q64))
    }

    /// Decomposes a thorax-relative humeral local `Motor` into `(scapulothoracic_motor, glenohumeral_motor)`.
    pub fn solve_coupled_motors(&self, total_humeral_local: Motor) -> (Motor, Motor) {
        let (rot, trans) = total_humeral_local.to_rotation_translation();
        let (st_q, gh_q) = self.solve_coupled_quaternions([
            rot[0] as f32,
            rot[1] as f32,
            rot[2] as f32,
            rot[3] as f32,
        ]);
        let st_m = Motor::from_rotation_translation(
            [
                st_q[0] as f64,
                st_q[1] as f64,
                st_q[2] as f64,
                st_q[3] as f64,
            ],
            [0.0, 0.0, 0.0],
        );
        let gh_m = Motor::from_rotation_translation(
            [
                gh_q[0] as f64,
                gh_q[1] as f64,
                gh_q[2] as f64,
                gh_q[3] as f64,
            ],
            trans,
        );
        (st_m, gh_m)
    }
}

/// One joint-coupling term inside a [`BiarticularLengthConstraint`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BiarticularTerm {
    /// Index of the spanned joint in the caller's joint angle or `Motor` slice.
    pub joint_index: usize,
    /// Signed moment arm $r_{m,k}$ in meters (positive when positive rotation lengthens the tendon).
    pub moment_arm_m: f32,
    /// Unit joint axis in local joint coordinates used when reading/writing PGA `Motor`s.
    pub axis: [f32; 3],
}

/// Passive multi-joint tendon excursion invariant $\sum_k r_{m,k}\,\theta_k \le L_{m,\max}$
/// (e.g. rectus femoris, hamstrings, gastrocnemius, extrinsic finger flexors).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiarticularLengthConstraint {
    /// Fixed-capacity joint terms (`HotZeroHeap`).
    pub terms: [BiarticularTerm; MAX_BIARTICULAR_JOINTS],
    /// Active term count in `0..=MAX_BIARTICULAR_JOINTS`.
    pub term_count: u8,
    /// Maximum allowable passive tendon excursion $L_{m,\max}$ in meters.
    pub max_excursion_m: f32,
}

impl BiarticularLengthConstraint {
    /// Constructs a passive multi-joint tendon excursion constraint.
    pub fn new(terms: &[BiarticularTerm], max_excursion_m: f32) -> Result<Self, SkeletalError> {
        if terms.is_empty()
            || terms.len() > MAX_BIARTICULAR_JOINTS
            || !max_excursion_m.is_finite()
            || terms.iter().any(|t| !t.moment_arm_m.is_finite())
        {
            return Err(SkeletalError::InvalidInput);
        }
        let mut buf = [BiarticularTerm::default(); MAX_BIARTICULAR_JOINTS];
        for (i, t) in terms.iter().enumerate() {
            buf[i] = *t;
        }
        Ok(Self {
            terms: buf,
            term_count: terms.len() as u8,
            max_excursion_m,
        })
    }

    /// Evaluates $\sum_k r_{m,k}\,\theta_k$ in meters from `joint_angles_rad`.
    pub fn excursion(&self, joint_angles_rad: &[f32]) -> Result<f32, SkeletalError> {
        let n = self.term_count as usize;
        if n == 0 || n > MAX_BIARTICULAR_JOINTS {
            return Err(SkeletalError::InvalidInput);
        }
        let mut sum = 0.0_f64;
        for t in &self.terms[..n] {
            let theta = *joint_angles_rad
                .get(t.joint_index)
                .ok_or(SkeletalError::InvalidInput)?;
            if !theta.is_finite() {
                return Err(SkeletalError::InvalidInput);
            }
            sum += (t.moment_arm_m as f64) * (theta as f64);
        }
        Ok(sum as f32)
    }

    /// Returns excess tendon excursion above `max_excursion_m * laxity_scale` (`0.0` when satisfied).
    pub fn violation(
        &self,
        joint_angles_rad: &[f32],
        laxity_scale: f32,
    ) -> Result<f32, SkeletalError> {
        let lax = if laxity_scale.is_finite() {
            laxity_scale.max(0.0)
        } else {
            1.0
        };
        let exc = self.excursion(joint_angles_rad)?;
        Ok((exc - self.max_excursion_m * lax).max(0.0))
    }

    /// Returns `true` when $\sum_k r_{m,k}\,\theta_k \le L_{m,\max} \cdot \text{laxity\_scale}$.
    pub fn is_satisfied(
        &self,
        joint_angles_rad: &[f32],
        laxity_scale: f32,
    ) -> Result<bool, SkeletalError> {
        Ok(self.violation(joint_angles_rad, laxity_scale)? <= 1e-6)
    }

    /// Projects `joint_angles_rad` in-place onto the half-space $\sum_k r_{m,k}\theta_k \le L_{\max}$
    /// using compliance-weighted minimum-norm KKT projection (`weights` defaults to uniform `1.0`).
    /// Returns the resolved violation magnitude before projection.
    pub fn project_angles_in_place(
        &self,
        joint_angles_rad: &mut [f32],
        weights: Option<&[f32]>,
        laxity_scale: f32,
    ) -> Result<f32, SkeletalError> {
        let violation = self.violation(joint_angles_rad, laxity_scale)?;
        if violation <= 1e-7 {
            return Ok(0.0);
        }
        let n = self.term_count as usize;
        let mut denom = 0.0_f64;
        for t in &self.terms[..n] {
            let w = weights
                .and_then(|ws| ws.get(t.joint_index).copied())
                .filter(|w| w.is_finite() && *w > 0.0)
                .unwrap_or(1.0) as f64;
            let r = t.moment_arm_m as f64;
            denom += w * r * r;
        }
        if denom <= 1e-12 {
            return Ok(violation);
        }
        let lambda = (violation as f64) / denom;
        for t in &self.terms[..n] {
            let w = weights
                .and_then(|ws| ws.get(t.joint_index).copied())
                .filter(|w| w.is_finite() && *w > 0.0)
                .unwrap_or(1.0) as f64;
            let r = t.moment_arm_m as f64;
            joint_angles_rad[t.joint_index] -= (lambda * w * r) as f32;
        }
        Ok(violation)
    }

    /// Enforces the passive biarticular excursion invariant directly on local PGA `Motor`s.
    pub fn project_local_motors_in_place(
        &self,
        local_motors: &mut [Motor],
        laxity_scale: f32,
    ) -> Result<f32, SkeletalError> {
        let n = self.term_count as usize;
        if n == 0 || n > MAX_BIARTICULAR_JOINTS {
            return Err(SkeletalError::InvalidInput);
        }
        let mut angles = [0.0_f32; MAX_BIARTICULAR_JOINTS];
        let mut axes = [[1.0_f64, 0.0, 0.0]; MAX_BIARTICULAR_JOINTS];
        for (idx, t) in self.terms[..n].iter().enumerate() {
            let m = local_motors
                .get(t.joint_index)
                .ok_or(SkeletalError::InvalidInput)?;
            let (rot, _) = m.to_rotation_translation();
            let axis = normalize3_or(
                [t.axis[0] as f64, t.axis[1] as f64, t.axis[2] as f64],
                [1.0, 0.0, 0.0],
            );
            axes[idx] = axis;
            let (_, twist) = decompose_swing_twist(
                [rot[0] as f32, rot[1] as f32, rot[2] as f32, rot[3] as f32],
                [axis[0] as f32, axis[1] as f32, axis[2] as f32],
            );
            let dot_ax = (twist[1] as f64) * axis[0]
                + (twist[2] as f64) * axis[1]
                + (twist[3] as f64) * axis[2];
            angles[idx] = (2.0 * dot_ax.atan2(twist[0] as f64)) as f32;
        }

        let lax = if laxity_scale.is_finite() {
            laxity_scale.max(0.0)
        } else {
            1.0
        };
        let mut sum = 0.0_f64;
        let mut denom = 0.0_f64;
        for (idx, t) in self.terms[..n].iter().enumerate() {
            let r = t.moment_arm_m as f64;
            sum += r * (angles[idx] as f64);
            denom += r * r;
        }
        let violation = (sum - (self.max_excursion_m as f64) * (lax as f64)).max(0.0);
        if violation <= 1e-7 || denom <= 1e-12 {
            return Ok(0.0);
        }
        let lambda = violation / denom;
        for (idx, t) in self.terms[..n].iter().enumerate() {
            let delta = -lambda * (t.moment_arm_m as f64);
            let (rot, trans) = local_motors[t.joint_index].to_rotation_translation();
            let dq = quat_from_axis_angle_f64(axes[idx], delta);
            let new_rot = quat_normalize_f64(quat_mul(rot, dq));
            local_motors[t.joint_index] = Motor::from_rotation_translation(new_rot, trans);
        }
        Ok(violation as f32)
    }
}

#[cfg(test)]
mod swing_twist_tests {
    use super::super::assert_zero_alloc;
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_6, PI};

    fn quat_mul_f32(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]
    }

    fn axis_angle_f32(axis: [f32; 3], angle: f32) -> [f32; 4] {
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let (s, c) = (0.5 * angle).sin_cos();
        [c, axis[0] / len * s, axis[1] / len * s, axis[2] / len * s]
    }

    #[test]
    fn swing_twist_decomposition_reconstructs_and_orthogonally_separates() {
        let twist_axis = [1.0, 0.0, 0.0];
        let q_swing_true = axis_angle_f32([0.0, 0.6, 0.8], 0.9);
        let q_twist_true = axis_angle_f32(twist_axis, -0.55);
        let q = quat_mul_f32(q_swing_true, q_twist_true);

        let (swing, twist) = decompose_swing_twist(q, twist_axis);
        let recombined = quat_mul_f32(swing, twist);
        for i in 0..4 {
            assert!((recombined[i] - q[i]).abs() < 1e-5, "recombine[{i}]");
            assert!((swing[i] - q_swing_true[i]).abs() < 1e-5, "swing[{i}]");
            assert!((twist[i] - q_twist_true[i]).abs() < 1e-5, "twist[{i}]");
        }
        assert!(
            swing[1].abs() < 1e-5,
            "swing must have zero twist-axis component"
        );

        // 180-degree swing singularity must stay finite and reconstruct q.
        let q_180_swing = [0.0, 0.0, 1.0, 0.0];
        let (s180, t180) = decompose_swing_twist(q_180_swing, twist_axis);
        let r180 = quat_mul_f32(s180, t180);
        for i in 0..4 {
            assert!((r180[i] - q_180_swing[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn swing_twist_elliptical_and_polygonal_projection_with_laxity() {
        let axis = [1.0, 0.0, 0.0];
        let q_out = quat_mul_f32(
            axis_angle_f32([0.0, 1.0, 0.0], 1.1),
            axis_angle_f32(axis, 0.9),
        );
        let clamped = project_swing_twist_cone(q_out, axis, 0.6, 0.3, -0.4, 0.5, 1.0);
        let (s, t) = decompose_swing_twist(clamped, axis);
        let swing_angle = 2.0
            * (s[1] * s[1] + s[2] * s[2] + s[3] * s[3])
                .sqrt()
                .atan2(s[0].abs());
        let twist_angle = 2.0 * t[1].atan2(t[0]);
        assert!(
            (swing_angle - 0.6).abs() < 1e-4,
            "swing_angle={swing_angle}"
        );
        assert!(
            (twist_angle - 0.5).abs() < 1e-4,
            "twist_angle={twist_angle}"
        );

        // Relaxin-style laxity_scale = 1.5 widens cone margins proportionally.
        let lax = project_swing_twist_cone(q_out, axis, 0.6, 0.3, -0.4, 0.5, 1.5);
        let (sl, tl) = decompose_swing_twist(lax, axis);
        let swing_lax = 2.0
            * (sl[1] * sl[1] + sl[2] * sl[2] + sl[3] * sl[3])
                .sqrt()
                .atan2(sl[0].abs());
        let twist_lax = 2.0 * tl[1].atan2(tl[0]);
        assert!((swing_lax - 0.9).abs() < 1e-4, "swing_lax={swing_lax}");
        assert!((twist_lax - 0.75).abs() < 1e-4, "twist_lax={twist_lax}");

        // Swing-coupled twist narrowing and polygonal cone.
        let poly = SwingTwistConeConstraint::elliptical(axis, 0.8, 0.8, -0.6, 0.6)
            .with_twist_coupling(0.5)
            .with_polygon_swing(&[[0.5, 0.0], [0.0, 0.3], [-0.5, 0.0], [0.0, -0.3]])
            .unwrap();
        let q_coupled = quat_mul_f32(
            axis_angle_f32([0.0, 1.0, 0.0], 0.5),
            axis_angle_f32(axis, 0.5),
        );
        let proj_c = poly.project_quaternion(q_coupled, 1.0);
        let (_, tc) = decompose_swing_twist(proj_c, axis);
        let tc_angle = 2.0 * tc[1].atan2(tc[0]);
        assert!(
            tc_angle <= 0.31,
            "coupled twist should narrow near boundary: {tc_angle}"
        );
        assert!(poly.is_quaternion_within(proj_c, 1.0, 1e-4));
    }

    #[test]
    fn scapulohumeral_rhythm_follows_two_to_one_above_thirty_degrees() {
        let coupling = ScapuloHumeralCoupling::default();

        let (gh_low, st_low) = coupling.partition_elevation_rad(FRAC_PI_6 * 0.8);
        assert!((st_low - 0.0).abs() < 1e-6);
        assert!((gh_low - FRAC_PI_6 * 0.8).abs() < 1e-6);

        // At 180 deg: 30 deg setting + 100 deg GH + 50 deg ST (2:1 above 30 deg).
        let (gh_180, st_180) = coupling.partition_elevation_rad(PI);
        assert!((gh_180 - (FRAC_PI_6 + 2.0 * (PI - FRAC_PI_6) / 3.0)).abs() < 1e-5);
        assert!((st_180 - ((PI - FRAC_PI_6) / 3.0)).abs() < 1e-5);

        // Quaternion split at 90 deg elevation: ST = 20 deg, GH = 70 deg, st * gh == total.
        let q_90 = axis_angle_f32([0.0, 0.0, 1.0], FRAC_PI_2);
        let (st_q, gh_q) = coupling.solve_coupled_quaternions(q_90);
        let recombined = quat_mul_f32(st_q, gh_q);
        for i in 0..4 {
            assert!((recombined[i] - q_90[i]).abs() < 1e-5);
        }
        let st_ang = 2.0 * st_q[3].atan2(st_q[0]);
        assert!((st_ang - (FRAC_PI_2 - FRAC_PI_6) / 3.0).abs() < 1e-4);

        // Pure axial humeral rotation around -Y must produce zero scapular upward rotation.
        let q_axial = axis_angle_f32([0.0, -1.0, 0.0], FRAC_PI_3);
        let (st_axial, _) = coupling.solve_coupled_quaternions(q_axial);
        assert!((st_axial[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn biarticular_tendon_excursion_projects_angles_and_motors() {
        // Gastrocnemius-like constraint: knee extension (+theta_0) and ankle dorsiflexion (+theta_1)
        // both lengthen the tendon with moment arms 0.03m and 0.04m, max excursion 0.035m.
        let constraint = BiarticularLengthConstraint::new(
            &[
                BiarticularTerm {
                    joint_index: 0,
                    moment_arm_m: 0.03,
                    axis: [1.0, 0.0, 0.0],
                },
                BiarticularTerm {
                    joint_index: 1,
                    moment_arm_m: 0.04,
                    axis: [1.0, 0.0, 0.0],
                },
            ],
            0.035,
        )
        .unwrap();

        let mut angles = [0.8_f32, 0.7_f32]; // 0.024 + 0.028 = 0.052 > 0.035
        assert!(!constraint.is_satisfied(&angles, 1.0).unwrap());
        let viol = constraint
            .project_angles_in_place(&mut angles, None, 1.0)
            .unwrap();
        assert!((viol - 0.017).abs() < 1e-5);
        assert!(constraint.is_satisfied(&angles, 1.0).unwrap());
        assert!((constraint.excursion(&angles).unwrap() - 0.035).abs() < 1e-5);

        let q0 = axis_angle_f32([1.0, 0.0, 0.0], 0.8);
        let q1 = axis_angle_f32([1.0, 0.0, 0.0], 0.7);
        let mut motors = [
            Motor::from_rotation_translation(
                [q0[0] as f64, q0[1] as f64, q0[2] as f64, q0[3] as f64],
                [0.0; 3],
            ),
            Motor::from_rotation_translation(
                [q1[0] as f64, q1[1] as f64, q1[2] as f64, q1[3] as f64],
                [0.0, -0.4, 0.0],
            ),
        ];
        let m_viol = constraint
            .project_local_motors_in_place(&mut motors, 1.0)
            .unwrap();
        assert!((m_viol - 0.017).abs() < 1e-4);
        let m_viol_after = constraint
            .project_local_motors_in_place(&mut motors, 1.0)
            .unwrap();
        assert!(m_viol_after <= 1e-5);
    }

    #[test]
    fn swing_twist_cone_motor_and_fabrik_zero_alloc() {
        let cone = SwingTwistConeConstraint::elliptical([0.0, 1.0, 0.0], 0.5, 0.25, -0.4, 0.4)
            .with_translation_bounds([-0.01; 3], [0.01; 3]);
        let q = quat_mul_f32(
            axis_angle_f32([1.0, 0.0, 0.0], 0.9),
            axis_angle_f32([0.0, 1.0, 0.0], 0.8),
        );
        let m = Motor::from_rotation_translation(
            [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64],
            [0.05, 0.0, -0.05],
        );

        assert_zero_alloc("swing_twist_motor_projection", || {
            let proj = cone.project_motor(m, 1.0);
            let (r, t) = proj.to_rotation_translation();
            assert!(cone.is_quaternion_within(
                [r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32],
                1.0,
                1e-4,
            ));
            assert!((t[0] - 0.01).abs() < 1e-6);
            assert!((t[2] + 0.01).abs() < 1e-6);
        });

        let chain = [
            Motor::from_translation(0.0, 0.0, 0.0),
            Motor::from_translation(0.0, 1.0, 0.0),
            Motor::from_translation(0.0, 2.0, 0.0),
        ];
        let constraints = [
            JointConstraint::SwingTwistCone(cone),
            JointConstraint::Unconstrained,
        ];
        let mut world_out = [Motor::identity(); 3];
        let mut local_out = [Motor::identity(); 3];
        let err = solve_constrained_fabrik_motor_ik_zero_heap(
            &chain,
            &constraints,
            [1.5, 0.5, 0.0],
            1e-3,
            16,
            &mut world_out,
            &mut local_out,
        )
        .unwrap();
        assert!(err.is_finite());
        let (_, p1) = world_out[1].to_rotation_translation();
        let dir0 = normalize3_or(p1, [0.0, 1.0, 0.0]);
        let angle0 = dot3(dir0, [0.0, 1.0, 0.0]).clamp(-1.0, 1.0).acos();
        assert!(
            angle0 <= 0.5 + 1e-3,
            "FABRIK SwingTwistCone violated: {angle0}"
        );
    }
}
