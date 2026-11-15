//! Regime IV: Volumetric Soft Tissue, Fascia & Hydrostatic Skeleton Solver (`vbd_soft_tissue.rs`).
//!
//! Implements graph-colored Vertex Block Descent (VBD) over tetrahedral meshes with
//! Stable Neo-Hookean quasi-incompressibility ($\det(\mathbf{F}) \approx 1$),
//! Ogden-Holzapfel anisotropic collagen fibers, Hill-type active muscle stress modulated by
//! $\eta_{\text{myo}}$ along $\mathbf{a}_0$, Gent dermal strain-hardening, fascia SDF sliding
//! contact, and a WGSL compute kernel (`HotZeroHeap`).

use super::SkeletalError;

pub const MAX_VBD_VERTICES: usize = 512;
pub const MAX_VBD_TETS: usize = 1024;
pub const MAX_VBD_COLORS: u8 = 8;

/// Vertex state for graph-colored Vertex Block Descent soft-tissue simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VbdVertex {
    pub pos: [f64; 3],
    pub prev_pos: [f64; 3],
    pub vel: [f64; 3],
    pub mass: f64,
    pub color: u8,
    pub pinned: bool,
    /// Optional fascia sliding plane: `signed_dist = dot(pos - fascia_origin, fascia_normal)`.
    pub fascia_active: bool,
    pub fascia_origin_m: [f64; 3],
    pub fascia_normal: [f64; 3],
    pub fascia_friction_mu: f64,
}

impl Default for VbdVertex {
    fn default() -> Self {
        Self {
            pos: [0.0; 3],
            prev_pos: [0.0; 3],
            vel: [0.0; 3],
            mass: 0.05,
            color: 0,
            pinned: false,
            fascia_active: false,
            fascia_origin_m: [0.0; 3],
            fascia_normal: [0.0, 1.0, 0.0],
            fascia_friction_mu: 0.08,
        }
    }
}

/// Constitutive hyperelastic + active Hill muscle + dermal strain-hardening parameters per tet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VbdTetElement {
    pub indices: [u16; 4],
    /// Rest inverse shape matrix $\mathbf{D}_m^{-1}$ (row-major 3x3).
    pub dm_inv: [f64; 9],
    pub rest_volume_m3: f64,
    /// Neo-Hookean shear modulus $\mu$ (Pa) and bulk Lamé parameter $\lambda_{\text{vol}}$ (Pa).
    pub mu_pa: f64,
    pub lambda_vol_pa: f64,
    /// Primary fiber direction $\mathbf{a}_0 \in S^2$ in rest configuration.
    pub fiber_dir: [f64; 3],
    /// Ogden-Holzapfel fiber stiffness $k_1$ (Pa), nonlinearity $k_2$, and dispersion $\kappa \in [0, 1/3]$.
    pub holzapfel_k1_pa: f64,
    pub holzapfel_k2: f64,
    pub holzapfel_kappa: f64,
    /// Active Hill muscle excitation $a(t) \in [0, 1]$, quality $\eta_{\text{myo}} \in [0, 1]$, peak stress $\sigma_{0,\max}$ (Pa).
    pub activation: f64,
    pub muscle_quality_eta: f64,
    pub sigma0_max_pa: f64,
    /// Dermal strain-hardening Gent shear modulus $\mu_{\text{derm}}$ (Pa) and locking parameter $J_m > 0$.
    pub dermal_mu_pa: f64,
    pub dermal_jm: f64,
}

impl Default for VbdTetElement {
    fn default() -> Self {
        Self {
            indices: [0, 1, 2, 3],
            dm_inv: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            rest_volume_m3: 1.0 / 6.0,
            mu_pa: 12_000.0,
            lambda_vol_pa: 180_000.0,
            fiber_dir: [1.0, 0.0, 0.0],
            holzapfel_k1_pa: 8_000.0,
            holzapfel_k2: 4.0,
            holzapfel_kappa: 0.05,
            activation: 0.0,
            muscle_quality_eta: 1.0,
            sigma0_max_pa: 300_000.0,
            dermal_mu_pa: 4_000.0,
            dermal_jm: 2.5,
        }
    }
}

impl VbdTetElement {
    /// Initializes `dm_inv` and `rest_volume_m3` from rest vertex coordinates.
    pub fn from_rest_positions(
        indices: [u16; 4],
        p0: [f64; 3],
        p1: [f64; 3],
        p2: [f64; 3],
        p3: [f64; 3],
    ) -> Result<Self, SkeletalError> {
        let dm = [
            p1[0] - p0[0],
            p2[0] - p0[0],
            p3[0] - p0[0],
            p1[1] - p0[1],
            p2[1] - p0[1],
            p3[1] - p0[1],
            p1[2] - p0[2],
            p2[2] - p0[2],
            p3[2] - p0[2],
        ];
        let det = det3(dm);
        if !det.is_finite() || det.abs() < 1e-12 {
            return Err(SkeletalError::InvalidInput);
        }
        let inv = inv3(dm)?;
        Ok(Self {
            indices,
            dm_inv: inv,
            rest_volume_m3: det.abs() / 6.0,
            ..Default::default()
        })
    }
}

/// GPU-aligned 32-byte VBD vertex record matching `VBD_SOFT_TISSUE_WGSL`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VbdVertexGpu {
    pub pos_mass: [f32; 4],
    pub inertia_color_pin: [f32; 4],
}

const _: [(); 32] = [(); core::mem::size_of::<VbdVertexGpu>()];

pub const VBD_SOFT_TISSUE_WGSL: &str = r#"
struct VbdParams {
    vertex_count: u32,
    active_color: u32,
    dt: f32,
    gravity_y: f32,
};

struct VbdVertexGpu {
    pos_mass: vec4<f32>,
    inertia_color_pin: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: VbdParams;
@group(0) @binding(1) var<storage, read_write> vertices: array<VbdVertexGpu>;
@group(0) @binding(2) var<storage, read> forces_and_hessian_diag: array<vec4<f32>>;

@compute @workgroup_size(64)
fn vbd_relax_color(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.vertex_count) {
        return;
    }
    var v = vertices[idx];
    let color_id = u32(v.inertia_color_pin.w + 0.5);
    let pinned = v.pos_mass.w <= 0.0;
    if (pinned || color_id != params.active_color) {
        return;
    }
    let dt2 = max(params.dt * params.dt, 1e-8);
    let m_dt2 = v.pos_mass.w / dt2;
    let grad_h = forces_and_hessian_diag[idx];
    let g = m_dt2 * (v.pos_mass.xyz - v.inertia_color_pin.xyz) + grad_h.xyz;
    let h_diag = max(m_dt2 + grad_h.w, 1e-4);
    v.pos_mass = vec4<f32>(v.pos_mass.xyz - g / h_diag, v.pos_mass.w);
    vertices[idx] = v;
}
"#;

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn det3(m: [f64; 9]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

fn inv3(m: [f64; 9]) -> Result<[f64; 9], SkeletalError> {
    let d = det3(m);
    if !d.is_finite() || d.abs() < 1e-12 {
        return Err(SkeletalError::InvalidInput);
    }
    let id = 1.0 / d;
    Ok([
        (m[4] * m[8] - m[5] * m[7]) * id,
        (m[2] * m[7] - m[1] * m[8]) * id,
        (m[1] * m[5] - m[2] * m[4]) * id,
        (m[5] * m[6] - m[3] * m[8]) * id,
        (m[0] * m[8] - m[2] * m[6]) * id,
        (m[2] * m[3] - m[0] * m[5]) * id,
        (m[3] * m[7] - m[4] * m[6]) * id,
        (m[1] * m[6] - m[0] * m[7]) * id,
        (m[0] * m[4] - m[1] * m[3]) * id,
    ])
}

#[inline]
fn mat3_mul(a: [f64; 9], b: [f64; 9]) -> [f64; 9] {
    let mut c = [0.0; 9];
    for r in 0..3 {
        for col in 0..3 {
            c[r * 3 + col] =
                a[r * 3] * b[col] + a[r * 3 + 1] * b[3 + col] + a[r * 3 + 2] * b[6 + col];
        }
    }
    c
}

#[inline]
fn mat3_vec(m: [f64; 9], v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

fn solve_spd_3x3(h: [[f64; 3]; 3], g: [f64; 3]) -> [f64; 3] {
    let m = [
        h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
    ];
    let det = det3(m);
    if det.abs() < 1e-12 || !det.is_finite() {
        return [
            -g[0] / h[0][0].max(1e-6),
            -g[1] / h[1][1].max(1e-6),
            -g[2] / h[2][2].max(1e-6),
        ];
    }
    let inv = match inv3(m) {
        Ok(i) => i,
        Err(_) => return [0.0; 3],
    };
    let dx = mat3_vec(inv, g);
    [-dx[0], -dx[1], -dx[2]]
}

/// Evaluates first Piola-Kirchhoff stress $\mathbf{P}$ and effective scalar tangent moduli
/// `(p_mat, k_iso, k_vol, k_fib)` for Neo-Hookean + Ogden-Holzapfel + Hill active + Gent dermal.
fn evaluate_constitutive_pk1(el: &VbdTetElement, f: [f64; 9]) -> ([f64; 9], f64, f64, f64) {
    let f0 = [f[0], f[3], f[6]];
    let f1 = [f[1], f[4], f[7]];
    let f2 = [f[2], f[5], f[8]];

    let j = det3(f);
    let i1 = dot3(f0, f0) + dot3(f1, f1) + dot3(f2, f2);

    // Cofactor matrix dJ/dF = [f1 x f2 | f2 x f0 | f0 x f1].
    let cof0 = cross3(f1, f2);
    let cof1 = cross3(f2, f0);
    let cof2 = cross3(f0, f1);
    let dj_df = [
        cof0[0], cof1[0], cof2[0], cof0[1], cof1[1], cof2[1], cof0[2], cof1[2], cof2[2],
    ];

    // 1. Stable Neo-Hookean (Smith et al. 2018).
    let vol_coeff = -el.mu_pa + el.lambda_vol_pa * (j - 1.0);
    let mut p = [0.0f64; 9];
    for idx in 0..9 {
        p[idx] = el.mu_pa * f[idx] + vol_coeff * dj_df[idx];
    }

    // 2. Dermal strain-hardening (Gent-like stiffening above rest state I1 > 3).
    let strain_i1 = (i1 - 3.0).max(0.0);
    let jm = el.dermal_jm.max(0.25);
    let eta_lock = (strain_i1 / jm).min(0.92);
    let dermal_mult = (1.0 / (1.0 - eta_lock)) - 1.0;
    let k_dermal = el.dermal_mu_pa.max(0.0) * dermal_mult;
    if k_dermal > 0.0 {
        for idx in 0..9 {
            p[idx] += k_dermal * f[idx];
        }
    }

    // 3. Ogden-Holzapfel collagen fiber + Hill active muscle contraction along a_0.
    let a_len = (el.fiber_dir[0] * el.fiber_dir[0]
        + el.fiber_dir[1] * el.fiber_dir[1]
        + el.fiber_dir[2] * el.fiber_dir[2])
        .sqrt()
        .max(1e-9);
    let a0 = [
        el.fiber_dir[0] / a_len,
        el.fiber_dir[1] / a_len,
        el.fiber_dir[2] / a_len,
    ];
    let f_a0 = mat3_vec(f, a0);
    let i4 = dot3(f_a0, f_a0);
    let stretch = i4.sqrt().max(1e-4);

    let kappa = el.holzapfel_kappa.clamp(0.0, 1.0 / 3.0);
    let e_bar = kappa * (i1 - 3.0) + (1.0 - 3.0 * kappa) * (i4 - 1.0);
    let mut k_fib_eff = 0.0;
    if e_bar > 0.0 && el.holzapfel_k1_pa > 0.0 {
        let k2 = el.holzapfel_k2.max(1e-6);
        let exp_term = (k2 * e_bar * e_bar).min(20.0).exp();
        let coeff = 2.0 * el.holzapfel_k1_pa * e_bar * exp_term;
        k_fib_eff += coeff * (1.0 - 3.0 * kappa) * 2.0
            + 4.0 * el.holzapfel_k1_pa * k2 * e_bar * e_bar * exp_term;
        for r in 0..3 {
            for c in 0..3 {
                p[r * 3 + c] +=
                    coeff * (kappa * f[r * 3 + c] + (1.0 - 3.0 * kappa) * f_a0[r] * a0[c]);
            }
        }
    }

    // Hill active contractile stress: sigma_active > 0 pulls inward along a_0.
    let act = el.activation.clamp(0.0, 1.0);
    let eta_myo = el.muscle_quality_eta.clamp(0.0, 1.0);
    if act > 0.0 && eta_myo > 0.0 && el.sigma0_max_pa > 0.0 {
        let f_len = (-((stretch - 1.0) * (stretch - 1.0)) / 0.20).exp();
        let sigma_act = act * eta_myo * el.sigma0_max_pa * f_len;
        let scale_act = sigma_act / stretch;
        k_fib_eff += scale_act;
        for r in 0..3 {
            for c in 0..3 {
                p[r * 3 + c] += scale_act * f_a0[r] * a0[c];
            }
        }
    }

    let k_iso = el.mu_pa + k_dermal + 2.0 * el.holzapfel_k1_pa * kappa;
    let k_vol = el.lambda_vol_pa;
    (p, k_iso, k_vol, k_fib_eff)
}

/// Advances Regime IV soft tissue using graph-colored Vertex Block Descent (`HotZeroHeap`),
/// returning the mean volume determinant $\det(\mathbf{F})$ across elements.
pub fn step_vbd_soft_tissue_zero_heap(
    vertices: &mut [VbdVertex],
    elements: &[VbdTetElement],
    gravity: [f64; 3],
    dt: f64,
    iterations: usize,
    damping: f64,
) -> Result<f64, SkeletalError> {
    let nv = vertices.len();
    let ne = elements.len();
    if nv == 0
        || nv > MAX_VBD_VERTICES
        || ne > MAX_VBD_TETS
        || !dt.is_finite()
        || dt <= 0.0
        || gravity.iter().any(|g| !g.is_finite())
        || !damping.is_finite()
    {
        return Err(SkeletalError::InvalidInput);
    }

    for v in vertices.iter() {
        if !v.mass.is_finite()
            || v.mass <= 0.0
            || v.color >= MAX_VBD_COLORS
            || v.pos.iter().any(|x| !x.is_finite())
            || v.vel.iter().any(|x| !x.is_finite())
        {
            return Err(SkeletalError::InvalidInput);
        }
    }
    for el in elements.iter() {
        if el.indices.iter().any(|&idx| idx as usize >= nv)
            || !el.rest_volume_m3.is_finite()
            || el.rest_volume_m3 <= 0.0
            || !el.mu_pa.is_finite()
            || !el.lambda_vol_pa.is_finite()
        {
            return Err(SkeletalError::InvalidInput);
        }
    }

    let mut y_inertia = [[0.0f64; 3]; MAX_VBD_VERTICES];
    let damp = damping.clamp(0.8, 1.0);
    for i in 0..nv {
        vertices[i].prev_pos = vertices[i].pos;
        if vertices[i].pinned {
            y_inertia[i] = vertices[i].pos;
            vertices[i].vel = [0.0; 3];
        } else {
            let v_pred = add3(scale3(vertices[i].vel, damp), scale3(gravity, dt));
            y_inertia[i] = add3(vertices[i].pos, scale3(v_pred, dt));
            vertices[i].pos = y_inertia[i];
        }
    }

    let dt2 = dt * dt;
    let iters = iterations.clamp(1, 32);

    for _ in 0..iters {
        for color in 0..MAX_VBD_COLORS {
            let mut grads = [[0.0f64; 3]; MAX_VBD_VERTICES];
            let mut hess = [[[0.0f64; 3]; 3]; MAX_VBD_VERTICES];
            let mut touched = [false; MAX_VBD_VERTICES];

            for i in 0..nv {
                if !vertices[i].pinned && vertices[i].color == color {
                    touched[i] = true;
                    let m_dt2 = vertices[i].mass / dt2;
                    let diff = sub3(vertices[i].pos, y_inertia[i]);
                    grads[i] = scale3(diff, m_dt2);
                    hess[i][0][0] = m_dt2;
                    hess[i][1][1] = m_dt2;
                    hess[i][2][2] = m_dt2;
                }
            }

            for el in elements {
                let has_color = el.indices.iter().any(|&idx| touched[idx as usize]);
                if !has_color {
                    continue;
                }

                let p0 = vertices[el.indices[0] as usize].pos;
                let p1 = vertices[el.indices[1] as usize].pos;
                let p2 = vertices[el.indices[2] as usize].pos;
                let p3 = vertices[el.indices[3] as usize].pos;

                let ds = [
                    p1[0] - p0[0],
                    p2[0] - p0[0],
                    p3[0] - p0[0],
                    p1[1] - p0[1],
                    p2[1] - p0[1],
                    p3[1] - p0[1],
                    p1[2] - p0[2],
                    p2[2] - p0[2],
                    p3[2] - p0[2],
                ];
                let f = mat3_mul(ds, el.dm_inv);
                let (pk1, k_iso, k_vol, k_fib) = evaluate_constitutive_pk1(el, f);

                let f0 = [f[0], f[3], f[6]];
                let f1 = [f[1], f[4], f[7]];
                let f2 = [f[2], f[5], f[8]];
                let cof = [
                    cross3(f1, f2)[0],
                    cross3(f2, f0)[0],
                    cross3(f0, f1)[0],
                    cross3(f1, f2)[1],
                    cross3(f2, f0)[1],
                    cross3(f0, f1)[1],
                    cross3(f1, f2)[2],
                    cross3(f2, f0)[2],
                    cross3(f0, f1)[2],
                ];

                // Shape-function gradients g_1, g_2, g_3 = rows of D_m^{-1}; g_0 = -(g_1+g_2+g_3).
                let g1 = [el.dm_inv[0], el.dm_inv[1], el.dm_inv[2]];
                let g2 = [el.dm_inv[3], el.dm_inv[4], el.dm_inv[5]];
                let g3 = [el.dm_inv[6], el.dm_inv[7], el.dm_inv[8]];
                let g0 = [
                    -(g1[0] + g2[0] + g3[0]),
                    -(g1[1] + g2[1] + g3[1]),
                    -(g1[2] + g2[2] + g3[2]),
                ];
                let gk_all = [g0, g1, g2, g3];

                let f_a0 = mat3_vec(f, el.fiber_dir);
                let f_a0_len = dot3(f_a0, f_a0).sqrt().max(1e-6);
                let f_unit = scale3(f_a0, 1.0 / f_a0_len);

                for local_k in 0..4 {
                    let vid = el.indices[local_k] as usize;
                    if !touched[vid] {
                        continue;
                    }
                    let gk = gk_all[local_k];
                    let force_k = scale3(mat3_vec(pk1, gk), el.rest_volume_m3);
                    grads[vid] = add3(grads[vid], force_k);

                    let gg = dot3(gk, gk);
                    let c_vec = mat3_vec(cof, gk);
                    let a_dot_g = dot3(el.fiber_dir, gk);
                    let fib_scale = el.rest_volume_m3 * k_fib * a_dot_g * a_dot_g;

                    for r in 0..3 {
                        hess[vid][r][r] += el.rest_volume_m3 * k_iso * gg;
                        for c in 0..3 {
                            hess[vid][r][c] += el.rest_volume_m3 * k_vol * c_vec[r] * c_vec[c]
                                + fib_scale * f_unit[r] * f_unit[c];
                        }
                    }
                }
            }

            for i in 0..nv {
                if !touched[i] {
                    continue;
                }
                let dx = solve_spd_3x3(hess[i], grads[i]);
                vertices[i].pos = add3(vertices[i].pos, dx);

                // Fascia SDF sliding contact with Coulomb tangential friction.
                if vertices[i].fascia_active {
                    let n_len = dot3(vertices[i].fascia_normal, vertices[i].fascia_normal)
                        .sqrt()
                        .max(1e-9);
                    let n_u = scale3(vertices[i].fascia_normal, 1.0 / n_len);
                    let rel = sub3(vertices[i].pos, vertices[i].fascia_origin_m);
                    let sd = dot3(rel, n_u);
                    if sd < 0.0 {
                        vertices[i].pos = sub3(vertices[i].pos, scale3(n_u, sd));
                        let disp = sub3(vertices[i].pos, vertices[i].prev_pos);
                        let disp_t = sub3(disp, scale3(n_u, dot3(disp, n_u)));
                        let t_len = dot3(disp_t, disp_t).sqrt();
                        if t_len > 1e-12 {
                            let fric = (vertices[i].fascia_friction_mu * (-sd)).min(t_len);
                            let slip = scale3(disp_t, fric / t_len);
                            vertices[i].pos = sub3(vertices[i].pos, slip);
                        }
                    }
                }
            }
        }
    }

    for i in 0..nv {
        if !vertices[i].pinned {
            vertices[i].vel = scale3(sub3(vertices[i].pos, vertices[i].prev_pos), 1.0 / dt);
        }
    }

    if ne == 0 {
        return Ok(1.0);
    }
    let mut sum_j = 0.0f64;
    for el in elements {
        let p0 = vertices[el.indices[0] as usize].pos;
        let p1 = vertices[el.indices[1] as usize].pos;
        let p2 = vertices[el.indices[2] as usize].pos;
        let p3 = vertices[el.indices[3] as usize].pos;
        let ds = [
            p1[0] - p0[0],
            p2[0] - p0[0],
            p3[0] - p0[0],
            p1[1] - p0[1],
            p2[1] - p0[1],
            p3[1] - p0[1],
            p1[2] - p0[2],
            p2[2] - p0[2],
            p3[2] - p0[2],
        ];
        sum_j += det3(mat3_mul(ds, el.dm_inv));
    }
    Ok(sum_j / (ne as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_tet_and_verts() -> ([VbdVertex; 4], [VbdTetElement; 1]) {
        let p = [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ];
        let mut verts = [VbdVertex::default(); 4];
        for i in 0..4 {
            verts[i] = VbdVertex {
                pos: p[i],
                prev_pos: p[i],
                color: i as u8,
                pinned: i == 0,
                ..Default::default()
            };
        }
        let tet = VbdTetElement::from_rest_positions([0, 1, 2, 3], p[0], p[1], p[2], p[3]).unwrap();
        (verts, [tet])
    }

    #[test]
    fn active_hill_muscle_contracts_along_fiber_and_preserves_volume() {
        let (mut verts, mut tets) = unit_tet_and_verts();
        tets[0].fiber_dir = [1.0, 0.0, 0.0];
        tets[0].activation = 1.0;
        tets[0].muscle_quality_eta = 1.0;
        tets[0].sigma0_max_pa = 250_000.0;
        tets[0].lambda_vol_pa = 1_500_000.0;

        let mut mean_j = 1.0;
        for _ in 0..5 {
            mean_j = step_vbd_soft_tissue_zero_heap(&mut verts, &tets, [0.0; 3], 0.002, 12, 0.98)
                .unwrap();
        }
        assert!(
            verts[1].pos[0] < 0.1,
            "active fiber should contract vertex 1 inward"
        );
        assert!(
            (mean_j - 1.0).abs() < 0.15,
            "quasi-incompressible det(F)={mean_j}"
        );
    }

    #[test]
    fn dermal_strain_hardening_resists_large_stretch() {
        let (mut verts_soft, mut tets_soft) = unit_tet_and_verts();
        let (mut verts_derm, mut tets_derm) = unit_tet_and_verts();
        verts_soft[1].pos[0] = 0.22;
        verts_derm[1].pos[0] = 0.22;
        tets_soft[0].dermal_mu_pa = 0.0;
        tets_derm[0].dermal_mu_pa = 80_000.0;
        tets_derm[0].dermal_jm = 1.6;

        step_vbd_soft_tissue_zero_heap(&mut verts_soft, &tets_soft, [0.0; 3], 0.002, 4, 0.99)
            .unwrap();
        step_vbd_soft_tissue_zero_heap(&mut verts_derm, &tets_derm, [0.0; 3], 0.002, 4, 0.99)
            .unwrap();
        assert!(verts_derm[1].pos[0] < verts_soft[1].pos[0]);
    }

    #[test]
    fn fascia_sdf_sliding_prevents_penetration_while_allowing_tangential_slip() {
        let (mut verts, tets) = unit_tet_and_verts();
        verts[1].pos = [0.1, -0.02, 0.0];
        verts[1].prev_pos = [0.08, -0.02, 0.0];
        verts[1].fascia_active = true;
        verts[1].fascia_origin_m = [0.0, 0.0, 0.0];
        verts[1].fascia_normal = [0.0, 1.0, 0.0];
        verts[1].fascia_friction_mu = 0.05;

        step_vbd_soft_tissue_zero_heap(&mut verts, &tets, [0.0, -9.81, 0.0], 0.002, 4, 0.99)
            .unwrap();
        assert!(verts[1].pos[1] >= -1e-6);
    }

    #[test]
    fn vbd_wgsl_shader_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(VBD_SOFT_TISSUE_WGSL).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
