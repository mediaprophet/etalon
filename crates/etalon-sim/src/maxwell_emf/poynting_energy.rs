//! Poynting Theorem, Spatiotemporal Collocation, and C02 Mode 0/1/3/4/5/6/7 Oracles (`etalon-sim`).

use super::vector_calculus::yee_curl_h_2nd;
use super::yee_fdtd::YeeMaterialParams;
use super::{
    BoundaryCondition, EmfOperatorError, Grid3dSpec, C0_M_PER_S, ELEMENTARY_CHARGE_C,
    EPS0_REF_F_PER_M, MU0_REF_H_PER_M,
};

/// C02 small closed-form `f64` algebraic budget: $64 \times 2^{-52} \times Q$.
#[inline]
pub fn c02_small_op_f64_budget(q_scale: f64) -> f64 {
    64.0 * f64::EPSILON * q_scale.abs()
}

/// C02 derived `f32` preview algebraic budget: $32 \times 2^{-23} \times Q$.
#[inline]
pub fn c02_preview_f32_budget(q_scale: f64) -> f64 {
    32.0 * (f32::EPSILON as f64) * q_scale.abs()
}

/// C02 $N$-term matrix/DFT reduction budget: $16 \gamma_n \sum_i |t_i|$ with $\gamma_n = \frac{n u}{1 - n u}, u = 2^{-53}$.
#[inline]
pub fn c02_reduction_f64_budget(n_terms: usize, sum_abs_terms: f64) -> f64 {
    let u = 0.5 * f64::EPSILON;
    let nu = (n_terms as f64) * u;
    let gamma_n = nu / (1.0 - nu);
    16.0 * gamma_n * sum_abs_terms.abs()
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
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn hypot3_scaled(x: f64, y: f64, z: f64) -> f64 {
    let ax = x.abs();
    let ay = y.abs();
    let az = z.abs();
    let m = ax.max(ay).max(az);
    if m == 0.0 {
        0.0
    } else {
        let sx = ax / m;
        let sy = ay / m;
        let sz = az / m;
        m * (sx * sx + sy * sy + sz * sz).sqrt()
    }
}

/// Mode `0x0100` (`instantaneous_E_B_v1`) instantaneous Poynting vector and energy density.
pub fn instantaneous_poynting_and_energy(
    e_v_per_m: [f64; 3],
    b_tesla: [f64; 3],
    mat: &YeeMaterialParams,
) -> Result<([f64; 3], f64), EmfOperatorError> {
    if e_v_per_m.iter().any(|v| !v.is_finite()) || b_tesla.iter().any(|v| !v.is_finite()) {
        return Err(EmfOperatorError::NonFiniteInput);
    }
    if !mat.epsilon_r.is_finite()
        || !mat.mu_r.is_finite()
        || mat.epsilon_r <= 0.0
        || mat.mu_r <= 0.0
    {
        return Err(EmfOperatorError::InvalidConstitutiveParameter {
            epsilon_r: mat.epsilon_r,
            mu_r: mat.mu_r,
            sigma: mat.sigma_s_per_m,
        });
    }
    let eps = EPS0_REF_F_PER_M * mat.epsilon_r;
    let mu = MU0_REF_H_PER_M * mat.mu_r;
    let h = [b_tesla[0] / mu, b_tesla[1] / mu, b_tesla[2] / mu];
    let s = cross3(e_v_per_m, h);
    let u = 0.5 * (eps * dot3(e_v_per_m, e_v_per_m) + dot3(b_tesla, b_tesla) / mu);
    Ok((s, u))
}

/// Time-averaged Poynting vector from **peak** complex phasors $(\hat{\mathbf{E}}_{\text{peak}}, \hat{\mathbf{H}}_{\text{peak}})$.
pub fn peak_phasor_time_average_poynting(
    e_peak: [(f64, f64); 3],
    h_peak: [(f64, f64); 3],
) -> Result<[f64; 3], EmfOperatorError> {
    for i in 0..3 {
        if !e_peak[i].0.is_finite()
            || !e_peak[i].1.is_finite()
            || !h_peak[i].0.is_finite()
            || !h_peak[i].1.is_finite()
        {
            return Err(EmfOperatorError::NonFiniteInput);
        }
    }
    let re_prod = |a: (f64, f64), b: (f64, f64)| -> f64 { a.0 * b.0 + a.1 * b.1 };
    let sx = 0.5 * (re_prod(e_peak[1], h_peak[2]) - re_prod(e_peak[2], h_peak[1]));
    let sy = 0.5 * (re_prod(e_peak[2], h_peak[0]) - re_prod(e_peak[0], h_peak[2]));
    let sz = 0.5 * (re_prod(e_peak[0], h_peak[1]) - re_prod(e_peak[1], h_peak[0]));
    Ok([sx, sy, sz])
}

/// Time-averaged Poynting vector from **RMS** complex phasors $(\hat{\mathbf{E}}_{\text{rms}}, \hat{\mathbf{H}}_{\text{rms}})$.
pub fn rms_phasor_time_average_poynting(
    e_rms: [(f64, f64); 3],
    h_rms: [(f64, f64); 3],
) -> Result<[f64; 3], EmfOperatorError> {
    let s_half = peak_phasor_time_average_poynting(e_rms, h_rms)?;
    Ok([2.0 * s_half[0], 2.0 * s_half[1], 2.0 * s_half[2]])
}

/// Evaluate vacuum wavelength $\lambda = c_0 / f$ for Mode `0x0100`, strictly rejecting static/aperiodic records.
pub fn mode0_carrier_wavelength_m(
    carrier_hz: f64,
    carrier_present: bool,
) -> Result<f64, EmfOperatorError> {
    if !carrier_present || carrier_hz == 0.0 {
        return Err(EmfOperatorError::StaticCarrierNoWavelength);
    }
    if !carrier_hz.is_finite() || carrier_hz < 0.0 {
        return Err(EmfOperatorError::InvalidFrequencyOrWavelength { value: carrier_hz });
    }
    Ok(C0_M_PER_S / carrier_hz)
}

/// C02-C04 active local-to-parent Hamilton quaternion rotation (`q = [x, y, z, w]`) for $(\mathbf{E}, \mathbf{B})$.
pub fn rotate_eb_hamilton_xyzw(
    q_xyzw: [f64; 4],
    e_local: [f64; 3],
    b_local: [f64; 3],
) -> Result<([f64; 3], [f64; 3], [[f64; 3]; 3]), EmfOperatorError> {
    if q_xyzw.iter().any(|v| !v.is_finite())
        || e_local.iter().any(|v| !v.is_finite())
        || b_local.iter().any(|v| !v.is_finite())
    {
        return Err(EmfOperatorError::NonFiniteInput);
    }
    let [qx, qy, qz, qw] = q_xyzw;
    let n2 = (((qx * qx) + (qy * qy)) + (qz * qz)) + (qw * qw);
    if (n2 - 1.0).abs() > 1e-12 {
        return Err(EmfOperatorError::InvalidFrameTransform {
            det: n2,
            ortho_residual: (n2 - 1.0).abs(),
        });
    }
    let n = n2.sqrt();
    let x = qx / n;
    let y = qy / n;
    let z = qz / n;
    let w = qw / n;

    let r = [
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
    ];

    let apply_r = |v: [f64; 3]| -> [f64; 3] {
        [
            r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
            r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
            r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
        ]
    };

    Ok((apply_r(e_local), apply_r(b_local), r))
}

/// Collocate staggered Yee fields $\mathbf{E}^n$ and $\mathbf{H}^{n\mp 1/2}$ to cell centers at time $t = n\Delta t$.
#[allow(clippy::too_many_arguments)]
pub fn collocate_yee_poynting_and_energy_cell_centers(
    grid: &Grid3dSpec,
    mat: &YeeMaterialParams,
    ex_n: &[f64],
    ey_n: &[f64],
    ez_n: &[f64],
    hx_nm_half: &[f64],
    hy_nm_half: &[f64],
    hz_nm_half: &[f64],
    hx_np_half: &[f64],
    hy_np_half: &[f64],
    hz_np_half: &[f64],
    sx_out: &mut [f64],
    sy_out: &mut [f64],
    sz_out: &mut [f64],
    u_out: &mut [f64],
) -> Result<(), EmfOperatorError> {
    let n = grid.validate()?;
    let eps = mat.epsilon_f_per_m();
    let mu = mat.mu_h_per_m();

    for slice in [
        ex_n,
        ey_n,
        ez_n,
        hx_nm_half,
        hy_nm_half,
        hz_nm_half,
        hx_np_half,
        hy_np_half,
        hz_np_half,
    ] {
        if slice.len() != n {
            return Err(EmfOperatorError::SliceLengthMismatch {
                expected: n,
                provided: slice.len(),
            });
        }
    }
    if sx_out.len() != n || sy_out.len() != n || sz_out.len() != n || u_out.len() != n {
        return Err(EmfOperatorError::SliceLengthMismatch {
            expected: n,
            provided: u_out.len(),
        });
    }

    for iz in 0..grid.nz {
        let izp = (iz + 1) % grid.nz;
        for iy in 0..grid.ny {
            let iyp = (iy + 1) % grid.ny;
            for ix in 0..grid.nx {
                let ixp = (ix + 1) % grid.nx;
                let c = grid.idx(ix, iy, iz);

                let ex_c = 0.25
                    * (ex_n[grid.idx(ix, iy, iz)]
                        + ex_n[grid.idx(ix, iyp, iz)]
                        + ex_n[grid.idx(ix, iy, izp)]
                        + ex_n[grid.idx(ix, iyp, izp)]);
                let ey_c = 0.25
                    * (ey_n[grid.idx(ix, iy, iz)]
                        + ey_n[grid.idx(ixp, iy, iz)]
                        + ey_n[grid.idx(ix, iy, izp)]
                        + ey_n[grid.idx(ixp, iy, izp)]);
                let ez_c = 0.25
                    * (ez_n[grid.idx(ix, iy, iz)]
                        + ez_n[grid.idx(ixp, iy, iz)]
                        + ez_n[grid.idx(ix, iyp, iz)]
                        + ez_n[grid.idx(ixp, iyp, iz)]);

                let hx_c = 0.25
                    * (hx_nm_half[grid.idx(ix, iy, iz)]
                        + hx_nm_half[grid.idx(ixp, iy, iz)]
                        + hx_np_half[grid.idx(ix, iy, iz)]
                        + hx_np_half[grid.idx(ixp, iy, iz)]);
                let hy_c = 0.25
                    * (hy_nm_half[grid.idx(ix, iy, iz)]
                        + hy_nm_half[grid.idx(ix, iyp, iz)]
                        + hy_np_half[grid.idx(ix, iy, iz)]
                        + hy_np_half[grid.idx(ix, iyp, iz)]);
                let hz_c = 0.25
                    * (hz_nm_half[grid.idx(ix, iy, iz)]
                        + hz_nm_half[grid.idx(ix, iy, izp)]
                        + hz_np_half[grid.idx(ix, iy, iz)]
                        + hz_np_half[grid.idx(ix, iy, izp)]);

                let s = cross3([ex_c, ey_c, ez_c], [hx_c, hy_c, hz_c]);
                sx_out[c] = s[0];
                sy_out[c] = s[1];
                sz_out[c] = s[2];
                u_out[c] = 0.5
                    * (eps * (ex_c * ex_c + ey_c * ey_c + ez_c * ez_c)
                        + mu * (hx_c * hx_c + hy_c * hy_c + hz_c * hz_c));
            }
        }
    }
    Ok(())
}

/// Evaluate the exact local staggered Poynting balance residual at each cell `(i, j, k)`.
#[allow(clippy::too_many_arguments)]
pub fn verify_discrete_poynting_power_balance(
    grid: &Grid3dSpec,
    bc: BoundaryCondition,
    mat: &YeeMaterialParams,
    dt: f64,
    ex_n: &[f64],
    ey_n: &[f64],
    ez_n: &[f64],
    ex_np1: &[f64],
    ey_np1: &[f64],
    ez_np1: &[f64],
    hx_half: &[f64],
    hy_half: &[f64],
    hz_half: &[f64],
    jx_half: &[f64],
    jy_half: &[f64],
    jz_half: &[f64],
    scratch_cx: &mut [f64],
    scratch_cy: &mut [f64],
    scratch_cz: &mut [f64],
) -> Result<f64, EmfOperatorError> {
    let n = grid.validate()?;
    yee_curl_h_2nd(
        grid,
        bc,
        hx_half,
        hy_half,
        hz_half,
        scratch_cx,
        scratch_cy,
        scratch_cz,
    )?;

    let eps = mat.epsilon_f_per_m();
    let sigma = mat.sigma_s_per_m;
    let inv_dt = 1.0 / dt;
    let mut max_res = 0.0_f64;

    for i in 0..n {
        let em_x = 0.5 * (ex_n[i] + ex_np1[i]);
        let em_y = 0.5 * (ey_n[i] + ey_np1[i]);
        let em_z = 0.5 * (ez_n[i] + ez_np1[i]);

        let de_dt_x = (ex_np1[i] - ex_n[i]) * inv_dt;
        let de_dt_y = (ey_np1[i] - ey_n[i]) * inv_dt;
        let de_dt_z = (ez_np1[i] - ez_n[i]) * inv_dt;

        let du_e_dt = eps * (de_dt_x * em_x + de_dt_y * em_y + de_dt_z * em_z);
        let curl_h_dot_e = scratch_cx[i] * em_x + scratch_cy[i] * em_y + scratch_cz[i] * em_z;
        let p_joule = sigma * (em_x * em_x + em_y * em_y + em_z * em_z);
        let p_impressed = jx_half[i] * em_x + jy_half[i] * em_y + jz_half[i] * em_z;

        let res = (du_e_dt - curl_h_dot_e + p_joule + p_impressed).abs();
        max_res = max_res.max(res);
    }
    Ok(max_res)
}

/// Admit a physical Stokes vector $[I, Q, U, V]$ per C02 Mode 1.
pub fn admit_stokes_vector(stokes: [f64; 4]) -> Result<[f64; 4], EmfOperatorError> {
    if stokes.iter().any(|v| !v.is_finite()) {
        return Err(EmfOperatorError::NonFiniteInput);
    }
    let [i, q, u, v] = stokes;
    if i < 0.0 {
        return Err(EmfOperatorError::InvalidPolarimetricState { i, q, u, v });
    }
    let p_norm = hypot3_scaled(q, u, v);
    let tol = 64.0 * f64::EPSILON * i.max(p_norm);
    if p_norm > i + tol {
        return Err(EmfOperatorError::InvalidPolarimetricState { i, q, u, v });
    }
    Ok(stokes)
}

/// Convert a complex transverse Jones vector $(E_1, E_2)$ to Stokes parameters $[I, Q, U, V]$.
pub fn jones_to_stokes_candidate_v1(
    e1_re_im: (f64, f64),
    e2_re_im: (f64, f64),
) -> Result<[f64; 4], EmfOperatorError> {
    if !e1_re_im.0.is_finite()
        || !e1_re_im.1.is_finite()
        || !e2_re_im.0.is_finite()
        || !e2_re_im.1.is_finite()
    {
        return Err(EmfOperatorError::NonFiniteInput);
    }
    let i1 = e1_re_im.0 * e1_re_im.0 + e1_re_im.1 * e1_re_im.1;
    let i2 = e2_re_im.0 * e2_re_im.0 + e2_re_im.1 * e2_re_im.1;
    let i = i1 + i2;
    let q = i1 - i2;
    let u = 2.0 * (e1_re_im.0 * e2_re_im.0 + e1_re_im.1 * e2_re_im.1);
    let v = 2.0 * (e1_re_im.1 * e2_re_im.0 - e1_re_im.0 * e2_re_im.1);
    admit_stokes_vector([i, q, u, v])
}

/// Passive transverse basis rotation of a Stokes vector by angle $\theta$.
pub fn rotate_stokes_basis_passive(
    stokes: [f64; 4],
    theta_rad: f64,
) -> Result<[f64; 4], EmfOperatorError> {
    admit_stokes_vector(stokes)?;
    if !theta_rad.is_finite() {
        return Err(EmfOperatorError::NonFiniteInput);
    }
    let (s2, c2) = (2.0 * theta_rad).sin_cos();
    let q_rot = stokes[1] * c2 + stokes[2] * s2;
    let u_rot = -stokes[1] * s2 + stokes[2] * c2;
    Ok([stokes[0], q_rot, u_rot, stokes[3]])
}

/// Geometric ray propagation delay $t = \text{path} / v$.
pub fn ray_geometric_delay_s(
    path_m: f64,
    mat: &YeeMaterialParams,
) -> Result<f64, EmfOperatorError> {
    if !path_m.is_finite() || path_m < 0.0 {
        return Err(EmfOperatorError::NonFiniteInput);
    }
    let v = mat.phase_velocity_m_per_s()?;
    Ok(path_m / v)
}

/// Mode `0x0103` spectral density Jacobian conversion and $N$-bin midpoint integration.
pub fn spectral_density_wavelength_midpoint_integral(
    f_min_hz: f64,
    f_max_hz: f64,
    f_f_const_w_m2_hz: f64,
    n_bins: usize,
) -> Result<(f64, f64), EmfOperatorError> {
    if !f_min_hz.is_finite()
        || !f_max_hz.is_finite()
        || !f_f_const_w_m2_hz.is_finite()
        || f_min_hz <= 0.0
        || f_max_hz <= f_min_hz
        || f_f_const_w_m2_hz < 0.0
        || n_bins == 0
    {
        return Err(EmfOperatorError::InvalidFrequencyOrWavelength { value: f_min_hz });
    }
    let l_min = C0_M_PER_S / f_max_hz;
    let l_max = C0_M_PER_S / f_min_hz;
    let analytic_integral = f_f_const_w_m2_hz * C0_M_PER_S * (1.0 / l_min - 1.0 / l_max);

    let h = (l_max - l_min) / (n_bins as f64);
    let mut midpoint_sum = 0.0_f64;
    for i in 0..n_bins {
        let lam = l_min + ((i as f64) + 0.5) * h;
        let f_lambda = f_f_const_w_m2_hz * C0_M_PER_S / (lam * lam);
        midpoint_sum += f_lambda * h;
    }
    Ok((analytic_integral, midpoint_sum))
}

/// Mode `0x0104` $4 \times 4$ Mueller matrix multiplication $\mathbf{S}_{\text{out}} = M \mathbf{S}_{\text{in}}$.
pub fn mueller_apply(
    m: &[[f64; 4]; 4],
    stokes_in: [f64; 4],
) -> Result<[f64; 4], EmfOperatorError> {
    admit_stokes_vector(stokes_in)?;
    let mut out = [0.0_f64; 4];
    for r in 0..4 {
        for c in 0..4 {
            if !m[r][c].is_finite() {
                return Err(EmfOperatorError::NonFiniteInput);
            }
            out[r] += m[r][c] * stokes_in[c];
        }
    }
    admit_stokes_vector(out)
}

/// Ideal horizontal ($+x$) linear polarizer $4 \times 4$ Mueller matrix.
pub const fn mueller_ideal_linear_polarizer_x() -> [[f64; 4]; 4] {
    [
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
}

/// Rotate a $4 \times 4$ Mueller matrix by passive basis angle $\theta$.
pub fn rotate_mueller_matrix_passive(m: &[[f64; 4]; 4], theta_rad: f64) -> [[f64; 4]; 4] {
    let (s2, c2) = (2.0 * theta_rad).sin_cos();
    let r_pos = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c2, s2, 0.0],
        [0.0, -s2, c2, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let r_neg = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c2, -s2, 0.0],
        [0.0, s2, c2, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut tmp = [[0.0_f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                tmp[i][j] += m[i][k] * r_pos[k][j];
            }
        }
    }
    let mut out = [[0.0_f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                out[i][j] += r_neg[i][k] * tmp[k][j];
            }
        }
    }
    out
}

/// Mode `0x0105` forward unnormalized DFT.
pub fn dft_forward_unscaled(
    z_re_im: &[(f64, f64)],
    window: Option<&[f64]>,
    out_bins: &mut [(f64, f64)],
) -> Result<(), EmfOperatorError> {
    let n = z_re_im.len();
    if n == 0 || out_bins.len() != n {
        return Err(EmfOperatorError::SliceLengthMismatch {
            expected: n,
            provided: out_bins.len(),
        });
    }
    if let Some(w) = window {
        if w.len() != n {
            return Err(EmfOperatorError::SliceLengthMismatch {
                expected: n,
                provided: w.len(),
            });
        }
    }
    let two_pi_over_n = 2.0 * core::f64::consts::PI / (n as f64);
    for (k, bin) in out_bins.iter_mut().enumerate() {
        let mut sum_re = 0.0_f64;
        let mut sum_im = 0.0_f64;
        for (idx, &(zr, zi)) in z_re_im.iter().enumerate() {
            if !zr.is_finite() || !zi.is_finite() {
                return Err(EmfOperatorError::NonFiniteInput);
            }
            let w_val = match window {
                Some(w) => {
                    if !w[idx].is_finite() {
                        return Err(EmfOperatorError::NonFiniteInput);
                    }
                    w[idx]
                }
                None => 1.0,
            };
            let angle = -two_pi_over_n * ((k * idx) as f64);
            let (s, c) = angle.sin_cos();
            let wr = zr * w_val;
            let wi = zi * w_val;
            sum_re += wr * c - wi * s;
            sum_im += wr * s + wi * c;
        }
        *bin = (sum_re, sum_im);
    }
    Ok(())
}

/// Mode `0x0105` real RF waveform reconstruction under EMF negative-exponential carrier.
#[inline]
pub fn reconstruct_rf_emf_negative_carrier(
    iq: (f64, f64),
    carrier_hz: f64,
    dt_s: f64,
    lo_phase_rad: f64,
) -> f64 {
    let phase = 2.0 * core::f64::consts::PI * carrier_hz * dt_s + lo_phase_rad;
    let (s, c) = phase.sin_cos();
    iq.0 * c + iq.1 * s
}

/// Real RF waveform reconstruction under QAC positive-exponential carrier.
#[inline]
pub fn reconstruct_rf_qac_positive_carrier(
    iq_qac: (f64, f64),
    carrier_hz: f64,
    dt_s: f64,
    lo_phase_rad: f64,
) -> f64 {
    let phase = 2.0 * core::f64::consts::PI * carrier_hz * dt_s + lo_phase_rad;
    let (s, c) = phase.sin_cos();
    iq_qac.0 * c - iq_qac.1 * s
}

/// Convert between EMF negative-carrier `(I, Q)` and QAC positive-carrier `(I, Q)` per C02-C08.
#[inline]
pub fn convert_iq_emf_to_qac_conjugated(iq_emf: (f64, f64)) -> (f64, f64) {
    (iq_emf.0, -iq_emf.1)
}

/// Mode `0x0106` monoenergetic uncollided Beer-Lambert photon fluence attenuation and independent dose.
pub fn mode6_photon_attenuation_and_independent_dose(
    density_kg_m3: f64,
    mass_attenuation_m2_kg: f64,
    path_m: f64,
    fluence_in_per_m2: f64,
    independent_energy_j: f64,
    independent_mass_kg: f64,
    photon_energy_ev: f64,
) -> Result<(f64, f64, f64, f64), EmfOperatorError> {
    if !density_kg_m3.is_finite()
        || !mass_attenuation_m2_kg.is_finite()
        || !path_m.is_finite()
        || !fluence_in_per_m2.is_finite()
        || !independent_energy_j.is_finite()
        || !independent_mass_kg.is_finite()
        || !photon_energy_ev.is_finite()
        || density_kg_m3 <= 0.0
        || mass_attenuation_m2_kg < 0.0
        || path_m < 0.0
        || fluence_in_per_m2 < 0.0
        || independent_energy_j < 0.0
        || independent_mass_kg <= 0.0
        || photon_energy_ev <= 0.0
    {
        return Err(EmfOperatorError::NonFiniteInput);
    }
    let mu_per_m = density_kg_m3 * mass_attenuation_m2_kg;
    let fluence_out = fluence_in_per_m2 * (-mu_per_m * path_m).exp();
    let absorbed_dose_gy = independent_energy_j / independent_mass_kg;
    let photon_energy_j = photon_energy_ev * ELEMENTARY_CHARGE_C;
    Ok((mu_per_m, fluence_out, absorbed_dose_gy, photon_energy_j))
}

/// Mode `0x0107` synthetic infinite homogeneous isotropic quasi-static volume-conductor dipole potential.
pub fn mode7_quasi_static_dipole_potential_v(
    dipole_a_m: [f64; 3],
    source_m: [f64; 3],
    observation_m: [f64; 3],
    conductivity_s_per_m: f64,
) -> Result<f64, EmfOperatorError> {
    if dipole_a_m.iter().any(|v| !v.is_finite())
        || source_m.iter().any(|v| !v.is_finite())
        || observation_m.iter().any(|v| !v.is_finite())
        || !conductivity_s_per_m.is_finite()
        || conductivity_s_per_m <= 0.0
    {
        return Err(EmfOperatorError::NonFiniteInput);
    }
    let dr = [
        observation_m[0] - source_m[0],
        observation_m[1] - source_m[1],
        observation_m[2] - source_m[2],
    ];
    let r_norm = hypot3_scaled(dr[0], dr[1], dr[2]);
    if r_norm == 0.0 {
        return Err(EmfOperatorError::SingularDipoleSourceEvaluation);
    }
    let q_dot_dr = dot3(dipole_a_m, dr);
    let denom = 4.0 * core::f64::consts::PI * conductivity_s_per_m * r_norm * r_norm * r_norm;
    Ok(q_dot_dr / denom)
}
