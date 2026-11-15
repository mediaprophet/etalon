//! Regime I: Articulated Multibody Endoskeleton Solver (`dca_multibody.rs`).
//!
//! Implements 6x6 Featherstone spatial inertia scaled by cortical/trabecular BMD and
//! muscle/adipose mass, $O(N)$ Articulated-Body Algorithm (ABA), $O(\log N)$ Divide-and-Conquer
//! Articulated-Body Algorithm (DCA) over a balanced binary assembly tree, closed-loop Jacobian
//! projection, and a WGSL compute kernel for parallel GPU assembly/disassembly (`HotZeroHeap`).

use super::{Motor, SkeletalError};

pub const MAX_DCA_BODIES: usize = 128;
pub const MAX_DCA_LOOPS: usize = 32;

/// 6D spatial motion or force vector `[ang_0, ang_1, ang_2, lin_0, lin_1, lin_2]`.
pub type SpatialVec6 = [f64; 6];
/// 6x6 spatial matrix in row-major `[row][col]` layout.
pub type SpatialMat6 = [[f64; 6]; 6];

/// Tissue densitometry and geometry parameters used to derive BMD-scaled 6x6 spatial inertia.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyDensitometryInertia {
    pub bone_volume_m3: f64,
    pub cortical_bmd_g_cm3: f64,
    pub trabecular_bv_tv: f64,
    pub cortical_thickness_mm: f64,
    pub muscle_volume_m3: f64,
    pub muscle_density_kg_m3: f64,
    pub adipose_volume_m3: f64,
    pub adipose_density_kg_m3: f64,
    pub com_local_m: [f64; 3],
    pub unit_gyration_radii_sq_m2: [f64; 3],
}

impl Default for BodyDensitometryInertia {
    fn default() -> Self {
        Self {
            bone_volume_m3: 0.0004,
            cortical_bmd_g_cm3: 1.85,
            trabecular_bv_tv: 0.30,
            cortical_thickness_mm: 4.0,
            muscle_volume_m3: 0.0025,
            muscle_density_kg_m3: 1060.0,
            adipose_volume_m3: 0.0010,
            adipose_density_kg_m3: 900.0,
            com_local_m: [0.0, -0.2, 0.0],
            unit_gyration_radii_sq_m2: [0.02, 0.004, 0.02],
        }
    }
}

impl BodyDensitometryInertia {
    /// Computes effective mass (kg) and 6x6 spatial inertia matrix at the body origin.
    pub fn compute_spatial_inertia(&self) -> Result<(f64, SpatialMat6), SkeletalError> {
        let vals = [
            self.bone_volume_m3,
            self.cortical_bmd_g_cm3,
            self.trabecular_bv_tv,
            self.cortical_thickness_mm,
            self.muscle_volume_m3,
            self.muscle_density_kg_m3,
            self.adipose_volume_m3,
            self.adipose_density_kg_m3,
            self.com_local_m[0],
            self.com_local_m[1],
            self.com_local_m[2],
            self.unit_gyration_radii_sq_m2[0],
            self.unit_gyration_radii_sq_m2[1],
            self.unit_gyration_radii_sq_m2[2],
        ];
        if vals.iter().any(|v| !v.is_finite())
            || self.bone_volume_m3 < 0.0
            || self.cortical_bmd_g_cm3 < 0.0
            || self.trabecular_bv_tv <= 0.0
            || self.cortical_thickness_mm <= 0.0
            || self.muscle_volume_m3 < 0.0
            || self.adipose_volume_m3 < 0.0
            || self.unit_gyration_radii_sq_m2.iter().any(|&r| r <= 0.0)
        {
            return Err(SkeletalError::InvalidInput);
        }

        let bmd_kg_m3 = self.cortical_bmd_g_cm3 * 1000.0;
        let microarch_scale = 0.4 + 0.6 * self.trabecular_bv_tv.clamp(0.05, 1.0);
        let cort_scale = 0.5 + 0.5 * (self.cortical_thickness_mm / 4.0).clamp(0.2, 2.5);
        let bone_mass = self.bone_volume_m3 * bmd_kg_m3 * microarch_scale * cort_scale;
        let muscle_mass = self.muscle_volume_m3 * self.muscle_density_kg_m3;
        let adipose_mass = self.adipose_volume_m3 * self.adipose_density_kg_m3;
        let mass = bone_mass + muscle_mass + adipose_mass;
        if mass <= 1e-9 {
            return Err(SkeletalError::InvalidInput);
        }

        let ic = [
            mass * self.unit_gyration_radii_sq_m2[0],
            mass * self.unit_gyration_radii_sq_m2[1],
            mass * self.unit_gyration_radii_sq_m2[2],
        ];
        let [cx, cy, cz] = self.com_local_m;
        let c_sq = cx * cx + cy * cy + cz * cz;

        let mut i6 = [[0.0f64; 6]; 6];
        // Upper-left 3x3: I_c + m * (c.c * I - c c^T)
        i6[0][0] = ic[0] + mass * (c_sq - cx * cx);
        i6[1][1] = ic[1] + mass * (c_sq - cy * cy);
        i6[2][2] = ic[2] + mass * (c_sq - cz * cz);
        i6[0][1] = -mass * cx * cy;
        i6[1][0] = i6[0][1];
        i6[0][2] = -mass * cx * cz;
        i6[2][0] = i6[0][2];
        i6[1][2] = -mass * cy * cz;
        i6[2][1] = i6[1][2];

        // Off-diagonal 3x3: m * [c]_x and m * [c]_x^T
        i6[0][4] = -mass * cz;
        i6[0][5] = mass * cy;
        i6[1][3] = mass * cz;
        i6[1][5] = -mass * cx;
        i6[2][3] = -mass * cy;
        i6[2][4] = mass * cx;

        i6[3][1] = mass * cz;
        i6[3][2] = -mass * cy;
        i6[4][0] = -mass * cz;
        i6[4][2] = mass * cx;
        i6[5][0] = mass * cy;
        i6[5][1] = -mass * cx;

        // Lower-right 3x3: m * I_3
        i6[3][3] = mass;
        i6[4][4] = mass;
        i6[5][5] = mass;

        Ok((mass, i6))
    }
}

/// Rigid-body articulated node for Regime I ABA/DCA dynamics.
#[derive(Debug, Clone, Copy)]
pub struct DcaBodyNode {
    pub parent: Option<u16>,
    pub joint_axis: SpatialVec6,
    pub parent_to_joint_origin_m: [f64; 3],
    pub parent_to_joint_rot_wxyz: [f64; 4],
    pub q: f64,
    pub qd: f64,
    pub tau: f64,
    pub external_wrench: SpatialVec6,
    pub synovial_stiffness: f64,
    pub synovial_damping: f64,
    pub inertia: BodyDensitometryInertia,
}

impl Default for DcaBodyNode {
    fn default() -> Self {
        Self {
            parent: None,
            joint_axis: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            parent_to_joint_origin_m: [0.0, 0.0, 0.0],
            parent_to_joint_rot_wxyz: [1.0, 0.0, 0.0, 0.0],
            q: 0.0,
            qd: 0.0,
            tau: 0.0,
            external_wrench: [0.0; 6],
            synovial_stiffness: 0.0,
            synovial_damping: 0.0,
            inertia: BodyDensitometryInertia::default(),
        }
    }
}

/// Closed kinematic loop constraint (ribcage, radius-ulna, pelvic ring).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopClosureConstraint {
    pub body_a: u16,
    pub body_b: u16,
    pub anchor_a_local_m: [f64; 3],
    pub anchor_b_local_m: [f64; 3],
    pub compliance: f64,
}

/// GPU-aligned 64-byte per-body state record matching `DCA_MULTIBODY_WGSL`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DcaBodyGpu {
    pub q_qd_tau_mass: [f32; 4],
    pub axis_ang_and_pad: [f32; 4],
    pub axis_lin_and_pad: [f32; 4],
    pub origin_and_parent: [f32; 4],
}

const _: [(); 64] = [(); core::mem::size_of::<DcaBodyGpu>()];

/// GPU-aligned 128-byte binary-tree DCA assembly node matching `DCA_MULTIBODY_WGSL`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DcaAssemblyNodeGpu {
    pub phi11_diag: [f32; 4],
    pub phi22_diag: [f32; 4],
    pub bias1_ang_lin: [f32; 4],
    pub bias2_ang_lin: [f32; 4],
    pub bias_tail_and_qdd: [f32; 4],
    pub left_right_joint_level: [u32; 4],
    pub _pad0: [u32; 4],
    pub _pad1: [u32; 4],
}

const _: [(); 128] = [(); core::mem::size_of::<DcaAssemblyNodeGpu>()];

pub const DCA_MULTIBODY_WGSL: &str = r#"
struct DcaParams {
    body_count: u32,
    level_stride: u32,
    dt: f32,
    gravity_y: f32,
};

struct DcaBodyGpu {
    q_qd_tau_mass: vec4<f32>,
    axis_ang_and_pad: vec4<f32>,
    axis_lin_and_pad: vec4<f32>,
    origin_and_parent: vec4<f32>,
};

struct DcaAssemblyNodeGpu {
    phi11_diag: vec4<f32>,
    phi22_diag: vec4<f32>,
    bias1_ang_lin: vec4<f32>,
    bias2_ang_lin: vec4<f32>,
    bias_tail_and_qdd: vec4<f32>,
    left_right_joint_level: vec4<u32>,
    _pad0: vec4<u32>,
    _pad1: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: DcaParams;
@group(0) @binding(1) var<storage, read> bodies: array<DcaBodyGpu>;
@group(0) @binding(2) var<storage, read_write> tree_nodes: array<DcaAssemblyNodeGpu>;

@compute @workgroup_size(64)
fn dca_assemble_level(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.body_count) {
        return;
    }
    let node = tree_nodes[idx];
    let left_idx = node.left_right_joint_level.x;
    let right_idx = node.left_right_joint_level.y;
    if (left_idx == right_idx) {
        let body = bodies[idx];
        let inv_m = 1.0 / max(body.q_qd_tau_mass.w, 1e-4);
        var leaf = node;
        leaf.phi11_diag = vec4<f32>(inv_m, inv_m, inv_m, inv_m);
        leaf.phi22_diag = vec4<f32>(inv_m, inv_m, inv_m, inv_m);
        leaf.bias1_ang_lin = vec4<f32>(0.0, params.gravity_y, 0.0, 0.0);
        leaf.bias2_ang_lin = vec4<f32>(0.0, params.gravity_y, 0.0, 0.0);
        tree_nodes[idx] = leaf;
        return;
    }
    let a = tree_nodes[left_idx];
    let b = tree_nodes[right_idx];
    let w_diag = max(a.phi22_diag + b.phi11_diag, vec4<f32>(1e-6));
    let v_diag = vec4<f32>(1.0) / w_diag;
    var merged = node;
    merged.phi11_diag = max(a.phi11_diag - a.phi11_diag * v_diag * a.phi11_diag, vec4<f32>(1e-6));
    merged.phi22_diag = max(b.phi22_diag - b.phi22_diag * v_diag * b.phi22_diag, vec4<f32>(1e-6));
    let beta = v_diag * (b.bias1_ang_lin - a.bias2_ang_lin);
    merged.bias1_ang_lin = a.bias1_ang_lin - a.phi11_diag * beta;
    merged.bias2_ang_lin = b.bias2_ang_lin + b.phi22_diag * beta;
    tree_nodes[idx] = merged;
}

@compute @workgroup_size(64)
fn dca_disassemble_level(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.body_count) {
        return;
    }
    var node = tree_nodes[idx];
    let body = bodies[idx];
    let tau = body.q_qd_tau_mass.z;
    let d_eff = max(dot(node.phi11_diag.xyz, abs(body.axis_ang_and_pad.xyz)) + 1e-4, 1e-4);
    node.bias_tail_and_qdd.w = (tau - dot(body.axis_ang_and_pad.xyz, node.bias1_ang_lin.xyz)) / d_eff;
    tree_nodes[idx] = node;
}
"#;

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot6(a: SpatialVec6, b: SpatialVec6) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + a[4] * b[4] + a[5] * b[5]
}

#[inline]
fn add6(a: SpatialVec6, b: SpatialVec6) -> SpatialVec6 {
    [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
        a[4] + b[4],
        a[5] + b[5],
    ]
}

#[inline]
fn sub6(a: SpatialVec6, b: SpatialVec6) -> SpatialVec6 {
    [
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
        a[3] - b[3],
        a[4] - b[4],
        a[5] - b[5],
    ]
}

#[inline]
fn scale6(a: SpatialVec6, s: f64) -> SpatialVec6 {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s, a[4] * s, a[5] * s]
}

#[inline]
fn mat6_vec(m: &SpatialMat6, v: SpatialVec6) -> SpatialVec6 {
    let mut out = [0.0; 6];
    for r in 0..6 {
        out[r] = dot6(m[r], v);
    }
    out
}

#[inline]
fn mat6_mul(a: &SpatialMat6, b: &SpatialMat6) -> SpatialMat6 {
    let mut out = [[0.0; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            let mut s = 0.0;
            for k in 0..6 {
                s += a[r][k] * b[k][c];
            }
            out[r][c] = s;
        }
    }
    out
}

#[inline]
fn mat6_transpose(a: &SpatialMat6) -> SpatialMat6 {
    let mut out = [[0.0; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            out[r][c] = a[c][r];
        }
    }
    out
}

#[inline]
fn mat6_add(a: &SpatialMat6, b: &SpatialMat6) -> SpatialMat6 {
    let mut out = [[0.0; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            out[r][c] = a[r][c] + b[r][c];
        }
    }
    out
}

#[inline]
fn mat6_sub(a: &SpatialMat6, b: &SpatialMat6) -> SpatialMat6 {
    let mut out = [[0.0; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            out[r][c] = a[r][c] - b[r][c];
        }
    }
    out
}

fn mat6_invert(a: &SpatialMat6) -> Result<SpatialMat6, SkeletalError> {
    let mut aug = [[0.0f64; 12]; 6];
    for r in 0..6 {
        for c in 0..6 {
            aug[r][c] = a[r][c];
            aug[r][c + 6] = if r == c { 1.0 } else { 0.0 };
        }
    }
    for col in 0..6 {
        let mut pivot = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..6 {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                pivot = row;
            }
        }
        if max_val < 1e-12 || !max_val.is_finite() {
            return Err(SkeletalError::InvalidInput);
        }
        if pivot != col {
            aug.swap(col, pivot);
        }
        let inv_diag = 1.0 / aug[col][col];
        for k in col..12 {
            aug[col][k] *= inv_diag;
        }
        for row in 0..6 {
            if row != col {
                let factor = aug[row][col];
                if factor != 0.0 {
                    for k in col..12 {
                        aug[row][k] -= factor * aug[col][k];
                    }
                }
            }
        }
    }
    let mut inv = [[0.0f64; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            inv[r][c] = aug[r][c + 6];
        }
    }
    Ok(inv)
}

#[inline]
fn quat_to_rot_mat(q: [f64; 4]) -> [[f64; 3]; 3] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        .sqrt()
        .max(1e-12);
    let (w, x, y, z) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

#[inline]
fn rot_mat_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

#[inline]
fn axis_angle_to_rot_mat(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if len < 1e-9 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let (ux, uy, uz) = (axis[0] / len, axis[1] / len, axis[2] / len);
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    [
        [t * ux * ux + c, t * ux * uy - s * uz, t * ux * uz + s * uy],
        [t * ux * uy + s * uz, t * uy * uy + c, t * uy * uz - s * ux],
        [t * ux * uz - s * uy, t * uy * uz + s * ux, t * uz * uz + c],
    ]
}

#[inline]
fn spatial_motion_cross(v1: SpatialVec6, v2: SpatialVec6) -> SpatialVec6 {
    let w1 = [v1[0], v1[1], v1[2]];
    let l1 = [v1[3], v1[4], v1[5]];
    let w2 = [v2[0], v2[1], v2[2]];
    let l2 = [v2[3], v2[4], v2[5]];
    let aw = cross3(w1, w2);
    let al1 = cross3(w1, l2);
    let al2 = cross3(l1, w2);
    [
        aw[0],
        aw[1],
        aw[2],
        al1[0] + al2[0],
        al1[1] + al2[1],
        al1[2] + al2[2],
    ]
}

#[inline]
fn spatial_force_cross(v: SpatialVec6, f: SpatialVec6) -> SpatialVec6 {
    let w = [v[0], v[1], v[2]];
    let lv = [v[3], v[4], v[5]];
    let n = [f[0], f[1], f[2]];
    let lf = [f[3], f[4], f[5]];
    let wn = cross3(w, n);
    let vf = cross3(lv, lf);
    let wf = cross3(w, lf);
    [
        wn[0] + vf[0],
        wn[1] + vf[1],
        wn[2] + vf[2],
        wf[0],
        wf[1],
        wf[2],
    ]
}

/// Builds 6x6 motion transform X_{cp} mapping parent velocity to child velocity.
fn build_joint_spatial_xform(node: &DcaBodyNode) -> SpatialMat6 {
    let r_static = quat_to_rot_mat(node.parent_to_joint_rot_wxyz);
    // Featherstone X_J(q) maps from joint-predecessor to joint-successor:
    // E_J = R(-axis * q) for parent->child convention.
    let e_j = axis_angle_to_rot_mat(
        [node.joint_axis[0], node.joint_axis[1], node.joint_axis[2]],
        -node.q,
    );
    // Prismatic displacement along joint linear axis if applicable
    let r = [
        node.parent_to_joint_origin_m[0] + node.joint_axis[3] * node.q,
        node.parent_to_joint_origin_m[1] + node.joint_axis[4] * node.q,
        node.parent_to_joint_origin_m[2] + node.joint_axis[5] * node.q,
    ];
    let e = rot_mat_mul(e_j, r_static);
    // -E * [r]_x
    let rx = [[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]];
    let mut neg_e_rx = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            neg_e_rx[row][col] =
                -(e[row][0] * rx[0][col] + e[row][1] * rx[1][col] + e[row][2] * rx[2][col]);
        }
    }
    let mut x = [[0.0; 6]; 6];
    for row in 0..3 {
        for col in 0..3 {
            x[row][col] = e[row][col];
            x[row + 3][col + 3] = e[row][col];
            x[row + 3][col] = neg_e_rx[row][col];
        }
    }
    x
}

/// Solves forward dynamics in $O(N)$ using Featherstone's Articulated-Body Algorithm (`HotZeroHeap`).
pub fn solve_aba_zero_heap(
    nodes: &[DcaBodyNode],
    gravity: [f64; 3],
    out_qdd: &mut [f64],
) -> Result<(), SkeletalError> {
    let n = nodes.len();
    if n == 0 || n > MAX_DCA_BODIES || gravity.iter().any(|g| !g.is_finite()) {
        return Err(SkeletalError::InvalidInput);
    }
    if out_qdd.len() < n {
        return Err(SkeletalError::OutputTooSmall);
    }
    for (i, node) in nodes.iter().enumerate() {
        if node.parent.is_some_and(|p| p as usize >= i)
            || !node.q.is_finite()
            || !node.qd.is_finite()
            || !node.tau.is_finite()
        {
            return Err(SkeletalError::InvalidInput);
        }
    }

    let mut x_up = [[[0.0f64; 6]; 6]; MAX_DCA_BODIES];
    let mut v = [[0.0f64; 6]; MAX_DCA_BODIES];
    let mut c = [[0.0f64; 6]; MAX_DCA_BODIES];
    let mut ia = [[[0.0f64; 6]; 6]; MAX_DCA_BODIES];
    let mut pa = [[0.0f64; 6]; MAX_DCA_BODIES];
    let mut u_vec = [[0.0f64; 6]; MAX_DCA_BODIES];
    let mut d_scalar = [0.0f64; MAX_DCA_BODIES];
    let mut u_scalar = [0.0f64; MAX_DCA_BODIES];
    let mut a = [[0.0f64; 6]; MAX_DCA_BODIES];

    for i in 0..n {
        let (_, i_body) = nodes[i].inertia.compute_spatial_inertia()?;
        ia[i] = i_body;
        x_up[i] = build_joint_spatial_xform(&nodes[i]);
        let vj = scale6(nodes[i].joint_axis, nodes[i].qd);
        if let Some(p) = nodes[i].parent {
            let vp = mat6_vec(&x_up[i], v[p as usize]);
            v[i] = add6(vp, vj);
            c[i] = spatial_motion_cross(v[i], vj);
        } else {
            v[i] = vj;
            c[i] = [0.0; 6];
        }
        let iv = mat6_vec(&i_body, v[i]);
        let coriolis = spatial_force_cross(v[i], iv);
        pa[i] = sub6(coriolis, nodes[i].external_wrench);
    }

    for i in (0..n).rev() {
        let s = nodes[i].joint_axis;
        let u = mat6_vec(&ia[i], s);
        let d = dot6(s, u);
        if d.abs() < 1e-12 || !d.is_finite() {
            return Err(SkeletalError::InvalidInput);
        }
        let tau_eff = nodes[i].tau
            - nodes[i].synovial_stiffness * nodes[i].q
            - nodes[i].synovial_damping * nodes[i].qd;
        let u_val = tau_eff - dot6(s, pa[i]);
        u_vec[i] = u;
        d_scalar[i] = d;
        u_scalar[i] = u_val;

        if let Some(p) = nodes[i].parent {
            let p_idx = p as usize;
            let mut ia_sub = ia[i];
            for r in 0..6 {
                for col in 0..6 {
                    ia_sub[r][col] -= (u[r] * u[col]) / d;
                }
            }
            let x_t = mat6_transpose(&x_up[i]);
            let ia_p_contrib = mat6_mul(&x_t, &mat6_mul(&ia_sub, &x_up[i]));
            ia[p_idx] = mat6_add(&ia[p_idx], &ia_p_contrib);

            let pa_term = add6(pa[i], add6(mat6_vec(&ia_sub, c[i]), scale6(u, u_val / d)));
            pa[p_idx] = add6(pa[p_idx], mat6_vec(&x_t, pa_term));
        }
    }

    let a_grav: SpatialVec6 = [0.0, 0.0, 0.0, -gravity[0], -gravity[1], -gravity[2]];
    for i in 0..n {
        let a_parent = if let Some(p) = nodes[i].parent {
            a[p as usize]
        } else {
            a_grav
        };
        let a_prime = add6(mat6_vec(&x_up[i], a_parent), c[i]);
        let qdd = (u_scalar[i] - dot6(u_vec[i], a_prime)) / d_scalar[i];
        out_qdd[i] = qdd;
        a[i] = add6(a_prime, scale6(nodes[i].joint_axis, qdd));
    }

    Ok(())
}

#[derive(Clone, Copy)]
struct DcaHandle {
    phi11: SpatialMat6,
    phi12: SpatialMat6,
    phi21: SpatialMat6,
    phi22: SpatialMat6,
    b1: SpatialVec6,
    b2: SpatialVec6,
}

#[derive(Clone, Copy)]
struct DcaMergeNode {
    handle: DcaHandle,
    left: usize,
    right: usize,
    joint_idx: usize,
    is_leaf: bool,
}

/// Solves forward dynamics for a serial chain in $O(\log N)$ balanced binary-tree passes
/// via Featherstone's Divide-and-Conquer Articulated-Body Algorithm (`HotZeroHeap`).
pub fn solve_dca_parallel_chain_zero_heap(
    nodes: &[DcaBodyNode],
    gravity: [f64; 3],
    out_qdd: &mut [f64],
) -> Result<(), SkeletalError> {
    let n = nodes.len();
    if n == 0 || n > MAX_DCA_BODIES || gravity.iter().any(|g| !g.is_finite()) {
        return Err(SkeletalError::InvalidInput);
    }
    if out_qdd.len() < n {
        return Err(SkeletalError::OutputTooSmall);
    }
    for (i, node) in nodes.iter().enumerate() {
        let expected_parent = if i == 0 { None } else { Some((i - 1) as u16) };
        if node.parent != expected_parent
            || !node.q.is_finite()
            || !node.qd.is_finite()
            || !node.tau.is_finite()
        {
            return Err(SkeletalError::InvalidInput);
        }
    }

    if n == 1 {
        return solve_aba_zero_heap(nodes, gravity, out_qdd);
    }

    let mut x_up = [[[0.0f64; 6]; 6]; MAX_DCA_BODIES];
    let mut v = [[0.0f64; 6]; MAX_DCA_BODIES];
    let mut c_joint = [[0.0f64; 6]; MAX_DCA_BODIES];

    let mut tree = [DcaMergeNode {
        handle: DcaHandle {
            phi11: [[0.0; 6]; 6],
            phi12: [[0.0; 6]; 6],
            phi21: [[0.0; 6]; 6],
            phi22: [[0.0; 6]; 6],
            b1: [0.0; 6],
            b2: [0.0; 6],
        },
        left: 0,
        right: 0,
        joint_idx: 0,
        is_leaf: true,
    }; MAX_DCA_BODIES * 2];

    // World->body transforms for expressing all port quantities in body-i frames at the joint boundary.
    let mut x_world_to_body = [[[0.0f64; 6]; 6]; MAX_DCA_BODIES];
    for i in 0..n {
        x_up[i] = build_joint_spatial_xform(&nodes[i]);
        let vj = scale6(nodes[i].joint_axis, nodes[i].qd);
        if i == 0 {
            x_world_to_body[0] = x_up[0];
            v[0] = vj;
            c_joint[0] = [0.0; 6];
        } else {
            x_world_to_body[i] = mat6_mul(&x_up[i], &x_world_to_body[i - 1]);
            v[i] = add6(mat6_vec(&x_up[i], v[i - 1]), vj);
            c_joint[i] = spatial_motion_cross(v[i], vj);
        }

        let (_, i_body) = nodes[i].inertia.compute_spatial_inertia()?;
        let i_inv = mat6_invert(&i_body)?;
        let iv = mat6_vec(&i_body, v[i]);
        let coriolis = spatial_force_cross(v[i], iv);
        let bias_force = sub6(nodes[i].external_wrench, coriolis);
        let b_origin = mat6_vec(&i_inv, bias_force);

        // Port 1 is proximal (in body i frame); Port 2 is distal (at joint i+1 frame, or body i frame if tip).
        let (phi12, phi21, phi22, b2) = if i + 1 < n {
            let x_next = build_joint_spatial_xform(&nodes[i + 1]);
            let x_next_t = mat6_transpose(&x_next);
            let phi22_next = mat6_mul(&x_next, &mat6_mul(&i_inv, &x_next_t));
            let phi12_next = mat6_mul(&i_inv, &x_next_t);
            let phi21_next = mat6_mul(&x_next, &i_inv);
            let b2_next = mat6_vec(&x_next, b_origin);
            (phi12_next, phi21_next, phi22_next, b2_next)
        } else {
            (i_inv, i_inv, i_inv, b_origin)
        };

        tree[i] = DcaMergeNode {
            handle: DcaHandle {
                phi11: i_inv,
                phi12,
                phi21,
                phi22,
                b1: b_origin,
                b2,
            },
            left: i,
            right: i,
            joint_idx: i,
            is_leaf: true,
        };
    }

    // Build balanced binary assembly tree in O(log N) parallel levels.
    let mut level_nodes = [0usize; MAX_DCA_BODIES];
    let mut level_len = n;
    for i in 0..n {
        level_nodes[i] = i;
    }
    let mut next_alloc = n;

    // Track per-internal-node cached assembly data for O(log N) top-down disassembly.
    let mut v_mat = [[[0.0f64; 6]; 6]; MAX_DCA_BODIES * 2];
    let mut beta_vec = [[0.0f64; 6]; MAX_DCA_BODIES * 2];
    let mut z_mat = [[[0.0f64; 6]; 6]; MAX_DCA_BODIES * 2];
    let mut d_inv_vec = [0.0f64; MAX_DCA_BODIES * 2];
    let mut level_rightmost_body = [0usize; MAX_DCA_BODIES];
    for i in 0..n {
        level_rightmost_body[i] = i;
    }

    while level_len > 1 {
        let mut next_len = 0usize;
        let mut idx = 0usize;
        while idx < level_len {
            if idx + 1 < level_len {
                let left_id = level_nodes[idx];
                let right_id = level_nodes[idx + 1];
                let j_idx = level_rightmost_body[left_id] + 1;
                let a_h = &tree[left_id].handle;
                let b_h = &tree[right_id].handle;

                let w = mat6_add(&a_h.phi22, &b_h.phi11);
                let z = mat6_invert(&w)?;
                let s = nodes[j_idx].joint_axis;
                let zs = mat6_vec(&z, s);
                let d = dot6(s, zs);
                if d.abs() < 1e-12 {
                    return Err(SkeletalError::InvalidInput);
                }
                let d_inv = 1.0 / d;
                let mut v_proj = z;
                for r in 0..6 {
                    for col in 0..6 {
                        v_proj[r][col] -= zs[r] * zs[col] * d_inv;
                    }
                }

                let tau_eff = nodes[j_idx].tau
                    - nodes[j_idx].synovial_stiffness * nodes[j_idx].q
                    - nodes[j_idx].synovial_damping * nodes[j_idx].qd;
                let rhs = add6(c_joint[j_idx], sub6(a_h.b2, b_h.b1));
                let beta = add6(mat6_vec(&v_proj, rhs), scale6(zs, tau_eff * d_inv));

                let phi11_c = mat6_sub(
                    &a_h.phi11,
                    &mat6_mul(&a_h.phi12, &mat6_mul(&v_proj, &a_h.phi21)),
                );
                let phi22_c = mat6_sub(
                    &b_h.phi22,
                    &mat6_mul(&b_h.phi21, &mat6_mul(&v_proj, &b_h.phi12)),
                );
                let phi12_c = mat6_mul(&a_h.phi12, &mat6_mul(&v_proj, &b_h.phi12));
                let phi21_c = mat6_transpose(&phi12_c);
                let b1_c = sub6(a_h.b1, mat6_vec(&a_h.phi12, beta));
                let b2_c = add6(b_h.b2, mat6_vec(&b_h.phi21, beta));

                let merged_id = next_alloc;
                next_alloc += 1;
                tree[merged_id] = DcaMergeNode {
                    handle: DcaHandle {
                        phi11: phi11_c,
                        phi12: phi12_c,
                        phi21: phi21_c,
                        phi22: phi22_c,
                        b1: b1_c,
                        b2: b2_c,
                    },
                    left: left_id,
                    right: right_id,
                    joint_idx: j_idx,
                    is_leaf: false,
                };
                v_mat[merged_id] = v_proj;
                beta_vec[merged_id] = beta;
                z_mat[merged_id] = z;
                d_inv_vec[merged_id] = d_inv;
                level_rightmost_body[merged_id] = level_rightmost_body[right_id];
                level_nodes[next_len] = merged_id;
                next_len += 1;
                idx += 2;
            } else {
                level_nodes[next_len] = level_nodes[idx];
                next_len += 1;
                idx += 1;
            }
        }
        level_len = next_len;
    }

    let root_id = level_nodes[0];
    let root_h = &tree[root_id].handle;
    let a_grav_world: SpatialVec6 = [0.0, 0.0, 0.0, -gravity[0], -gravity[1], -gravity[2]];
    let a_grav_base = mat6_vec(&x_up[0], a_grav_world);
    let s0 = nodes[0].joint_axis;
    let tau0_eff = nodes[0].tau
        - nodes[0].synovial_stiffness * nodes[0].q
        - nodes[0].synovial_damping * nodes[0].qd;

    // Solve joint 0 constraint force f0 and qdd[0] against base acceleration a_grav_base:
    // a_1^root = Phi11 * f0 + b1 = a_grav_base + s0 * qdd[0], s0^T f0 = tau0_eff.
    let k0 = mat6_invert(&root_h.phi11)?;
    let k0_s0 = mat6_vec(&k0, s0);
    let d0 = dot6(s0, k0_s0);
    if d0.abs() < 1e-12 {
        return Err(SkeletalError::InvalidInput);
    }
    let delta_a0 = sub6(a_grav_base, root_h.b1);
    let qdd0 = (tau0_eff - dot6(s0, mat6_vec(&k0, delta_a0))) / d0;
    out_qdd[0] = qdd0;
    let f_root = mat6_vec(&k0, add6(delta_a0, scale6(s0, qdd0)));

    // Top-down disassembly stack (O(log N) depth).
    let mut stack = [(root_id, f_root, [0.0f64; 6]); MAX_DCA_BODIES * 2];
    let mut sp = 1usize;
    while sp > 0 {
        sp -= 1;
        let (nid, f1, f2) = stack[sp];
        let node = &tree[nid];
        if node.is_leaf {
            continue;
        }
        let a_h = &tree[node.left].handle;
        let b_h = &tree[node.right].handle;
        let j_idx = node.joint_idx;
        let fj = add6(
            beta_vec[nid],
            mat6_vec(
                &v_mat[nid],
                sub6(mat6_vec(&a_h.phi21, f1), mat6_vec(&b_h.phi12, f2)),
            ),
        );
        let a2_a = add6(
            sub6(mat6_vec(&a_h.phi21, f1), mat6_vec(&a_h.phi22, fj)),
            a_h.b2,
        );
        let a1_b = add6(
            add6(mat6_vec(&b_h.phi11, fj), mat6_vec(&b_h.phi12, f2)),
            b_h.b1,
        );
        let s = nodes[j_idx].joint_axis;
        let zs = mat6_vec(&z_mat[nid], s);
        let rel_a = sub6(sub6(a1_b, a2_a), c_joint[j_idx]);
        out_qdd[j_idx] = dot6(zs, rel_a) * d_inv_vec[nid];

        stack[sp] = (node.left, f1, scale6(fj, -1.0));
        sp += 1;
        stack[sp] = (node.right, fj, f2);
        sp += 1;
    }

    Ok(())
}

/// Resolves closed kinematic loop residuals (e.g. ribcage, radius-ulna) via compliant
/// constraint projection over world body `Motor` poses (`HotZeroHeap`), returning max residual (m).
pub fn resolve_loop_closures_zero_heap(
    loops: &[LoopClosureConstraint],
    body_motors: &mut [Motor],
    iterations: usize,
) -> Result<f64, SkeletalError> {
    if loops.len() > MAX_DCA_LOOPS {
        return Err(SkeletalError::BudgetExceeded);
    }
    let n_bodies = body_motors.len();
    for lc in loops {
        if lc.body_a as usize >= n_bodies
            || lc.body_b as usize >= n_bodies
            || lc.body_a == lc.body_b
            || !lc.compliance.is_finite()
            || lc.compliance < 0.0
        {
            return Err(SkeletalError::InvalidInput);
        }
    }

    let iters = iterations.clamp(1, 32);
    let mut max_residual = 0.0f64;
    for _ in 0..iters {
        max_residual = 0.0;
        for lc in loops {
            let (ra, ta) = body_motors[lc.body_a as usize].to_rotation_translation();
            let (rb, tb) = body_motors[lc.body_b as usize].to_rotation_translation();
            let pa = body_motors[lc.body_a as usize].transform_point(lc.anchor_a_local_m);
            let pb = body_motors[lc.body_b as usize].transform_point(lc.anchor_b_local_m);
            let err = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let dist = (err[0] * err[0] + err[1] * err[1] + err[2] * err[2]).sqrt();
            max_residual = max_residual.max(dist);
            if dist > 1e-9 {
                let w = 0.5 / (1.0 + lc.compliance);
                let corr = [err[0] * w, err[1] * w, err[2] * w];
                body_motors[lc.body_a as usize] = Motor::from_rotation_translation(
                    ra,
                    [ta[0] + corr[0], ta[1] + corr[1], ta[2] + corr[2]],
                );
                body_motors[lc.body_b as usize] = Motor::from_rotation_translation(
                    rb,
                    [tb[0] - corr[0], tb[1] - corr[1], tb[2] - corr[2]],
                );
            }
        }
    }
    Ok(max_residual)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bmd_scaling_increases_spatial_inertia_and_mass() {
        let low_bmd = BodyDensitometryInertia {
            cortical_bmd_g_cm3: 1.0,
            trabecular_bv_tv: 0.12,
            cortical_thickness_mm: 2.0,
            ..Default::default()
        };
        let high_bmd = BodyDensitometryInertia {
            cortical_bmd_g_cm3: 1.9,
            trabecular_bv_tv: 0.42,
            cortical_thickness_mm: 5.5,
            ..Default::default()
        };
        let (m_lo, i_lo) = low_bmd.compute_spatial_inertia().unwrap();
        let (m_hi, i_hi) = high_bmd.compute_spatial_inertia().unwrap();
        assert!(m_hi > m_lo);
        assert!(i_hi[0][0] > i_lo[0][0]);
        assert!((i_lo[0][1] - i_lo[1][0]).abs() < 1e-12);
    }

    #[test]
    fn dca_binary_tree_matches_aba_on_multilink_chain() {
        let mut chain = [DcaBodyNode::default(); 6];
        for i in 0..6 {
            chain[i] = DcaBodyNode {
                parent: if i == 0 { None } else { Some((i - 1) as u16) },
                joint_axis: if i % 2 == 0 {
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                } else {
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                },
                parent_to_joint_origin_m: if i == 0 {
                    [0.0, 0.0, 0.0]
                } else {
                    [0.0, -0.35, 0.0]
                },
                parent_to_joint_rot_wxyz: [1.0, 0.0, 0.0, 0.0],
                q: 0.15 * (i as f64 + 1.0),
                qd: 0.25 * ((i % 3) as f64 - 1.0),
                tau: 12.0 - 1.8 * (i as f64),
                external_wrench: [0.0, 0.0, 0.0, 0.5, -2.0, 0.2],
                synovial_stiffness: 4.0,
                synovial_damping: 0.6,
                inertia: BodyDensitometryInertia::default(),
            };
        }

        let mut qdd_aba = [0.0f64; 6];
        let mut qdd_dca = [0.0f64; 6];
        let g = [0.0, -9.80665, 0.0];
        solve_aba_zero_heap(&chain, g, &mut qdd_aba).unwrap();
        solve_dca_parallel_chain_zero_heap(&chain, g, &mut qdd_dca).unwrap();

        for i in 0..6 {
            assert!(
                (qdd_aba[i] - qdd_dca[i]).abs() < 1e-8,
                "joint {i}: aba={} != dca={}",
                qdd_aba[i],
                qdd_dca[i]
            );
        }
    }

    #[test]
    fn closed_loop_projection_converges_below_submillimeter_residual() {
        let loops = [LoopClosureConstraint {
            body_a: 0,
            body_b: 1,
            anchor_a_local_m: [0.1, 0.0, 0.0],
            anchor_b_local_m: [-0.1, 0.0, 0.0],
            compliance: 0.0,
        }];
        let mut motors = [
            Motor::from_translation(0.0, 0.0, 0.0),
            Motor::from_translation(0.25, 0.05, 0.0),
        ];
        let res = resolve_loop_closures_zero_heap(&loops, &mut motors, 16).unwrap();
        assert!(res < 1e-4, "residual={res}");
    }

    #[test]
    fn dca_wgsl_shader_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(DCA_MULTIBODY_WGSL).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
