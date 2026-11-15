//! 3D Cartesian Yee FDTD Solver, CFL Gate, and Discrete Dispersion Oracle (`etalon-sim`).
//!
//! Provides the pure continuum 3D Cartesian Yee FDTD solver, exact 3D CFL admission gate,
//! analytical discrete dispersion relation, and periodic plane-wave convergence runner.

use super::vector_calculus::{yee_curl_e_2nd, yee_curl_h_2nd};
use super::{
    BoundaryCondition, EmfOperatorError, Grid3dSpec, C0_M_PER_S, EPS0_REF_F_PER_M, MU0_REF_H_PER_M,
};

/// Canonical model URI for Mode 2 lossless periodic no-phi Yee profile.
pub const YEE_LOSSLESS_PERIODIC_NO_PHI_URI: &str =
    "urn:qualia:emf:yee_cartesian_lossless_periodic_no_phi_v1";

/// Canonical model URI for Mode 2 conductive periodic no-phi Yee profile.
pub const YEE_CONDUCTIVE_PERIODIC_NO_PHI_URI: &str =
    "urn:qualia:emf:yee_cartesian_conductive_periodic_no_phi_v1";

/// Supported Mode 2 Yee solver profiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YeeSolverProfile {
    LosslessPeriodicNoPhiV1,
    ConductivePeriodicNoPhiV1,
}

/// Homogeneous isotropic constitutive parameters for a 3D Yee grid domain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YeeMaterialParams {
    pub epsilon_r: f64,
    pub mu_r: f64,
    pub sigma_s_per_m: f64,
}

impl YeeMaterialParams {
    pub const fn vacuum() -> Self {
        Self {
            epsilon_r: 1.0,
            mu_r: 1.0,
            sigma_s_per_m: 0.0,
        }
    }

    pub fn validate(&self, profile: YeeSolverProfile) -> Result<(), EmfOperatorError> {
        if !self.epsilon_r.is_finite()
            || !self.mu_r.is_finite()
            || !self.sigma_s_per_m.is_finite()
            || self.epsilon_r <= 0.0
            || self.mu_r <= 0.0
            || self.sigma_s_per_m < 0.0
        {
            return Err(EmfOperatorError::InvalidConstitutiveParameter {
                epsilon_r: self.epsilon_r,
                mu_r: self.mu_r,
                sigma: self.sigma_s_per_m,
            });
        }
        if profile == YeeSolverProfile::LosslessPeriodicNoPhiV1 && self.sigma_s_per_m != 0.0 {
            return Err(EmfOperatorError::InvalidConstitutiveParameter {
                epsilon_r: self.epsilon_r,
                mu_r: self.mu_r,
                sigma: self.sigma_s_per_m,
            });
        }
        Ok(())
    }

    #[inline]
    pub fn phase_velocity_m_per_s(&self) -> Result<f64, EmfOperatorError> {
        self.validate(YeeSolverProfile::ConductivePeriodicNoPhiV1)?;
        Ok(C0_M_PER_S / (self.epsilon_r * self.mu_r).sqrt())
    }

    #[inline]
    pub fn epsilon_f_per_m(&self) -> f64 {
        EPS0_REF_F_PER_M * self.epsilon_r
    }

    #[inline]
    pub fn mu_h_per_m(&self) -> f64 {
        MU0_REF_H_PER_M * self.mu_r
    }
}

/// Compute the exact 3D Cartesian Yee Courant-Friedrichs-Lewy (CFL) maximum stable timestep:
/// $$\Delta t_{\text{CFL}} = \frac{1}{v \sqrt{\Delta x^{-2} + \Delta y^{-2} + \Delta z^{-2}}}$$
pub fn yee_cfl_limit_3d(
    grid: &Grid3dSpec,
    mat: &YeeMaterialParams,
) -> Result<f64, EmfOperatorError> {
    grid.validate()?;
    let v = mat.phase_velocity_m_per_s()?;
    let inv_norm = (1.0 / (grid.dx * grid.dx)
        + 1.0 / (grid.dy * grid.dy)
        + 1.0 / (grid.dz * grid.dz))
        .sqrt();
    Ok(1.0 / (v * inv_norm))
}

/// Admit a candidate solver timestep $\Delta t$ against the 3D Cartesian Yee CFL bound.
pub fn admit_yee_timestep(
    grid: &Grid3dSpec,
    mat: &YeeMaterialParams,
    dt: f64,
) -> Result<f64, EmfOperatorError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(EmfOperatorError::NonFiniteInput);
    }
    let cfl_limit = yee_cfl_limit_3d(grid, mat)?;
    if dt > cfl_limit {
        return Err(EmfOperatorError::CflViolation { dt, cfl_limit });
    }
    Ok(cfl_limit)
}

/// Evaluate the exact 3D Cartesian Yee discrete dispersion relation for wavevector $\mathbf{k}$:
/// $$\sin^2\left(\frac{\omega_{\text{num}} \Delta t}{2}\right) = (v \Delta t)^2 \sum_{j \in \{x,y,z\}} \frac{\sin^2(k_j \Delta x_j / 2)}{\Delta x_j^2}$$
pub fn yee_discrete_omega_rad_per_s(
    grid: &Grid3dSpec,
    mat: &YeeMaterialParams,
    dt: f64,
    k_vec: [f64; 3],
) -> Result<f64, EmfOperatorError> {
    admit_yee_timestep(grid, mat, dt)?;
    if k_vec.iter().any(|k| !k.is_finite()) {
        return Err(EmfOperatorError::NonFiniteInput);
    }
    let v = mat.phase_velocity_m_per_s()?;
    let sx = (0.5 * k_vec[0] * grid.dx).sin() / grid.dx;
    let sy = (0.5 * k_vec[1] * grid.dy).sin() / grid.dy;
    let sz = (0.5 * k_vec[2] * grid.dz).sin() / grid.dz;

    let rhs = (v * dt) * (v * dt) * (sx * sx + sy * sy + sz * sz);
    if rhs > 1.0 {
        return Err(EmfOperatorError::DispersionEvanescent { rhs });
    }
    let arg = rhs.sqrt().clamp(0.0, 1.0);
    Ok((2.0 / dt) * arg.asin())
}

/// Advance a 3D staggered Yee electromagnetic state by one full leapfrog step $\Delta t$.
#[allow(clippy::too_many_arguments)]
pub fn yee_step_leapfrog_in_place(
    grid: &Grid3dSpec,
    profile: YeeSolverProfile,
    bc: BoundaryCondition,
    mat: &YeeMaterialParams,
    dt: f64,
    ex: &mut [f64],
    ey: &mut [f64],
    ez: &mut [f64],
    hx: &mut [f64],
    hy: &mut [f64],
    hz: &mut [f64],
    jx: &[f64],
    jy: &[f64],
    jz: &[f64],
    scratch_curl_x: &mut [f64],
    scratch_curl_y: &mut [f64],
    scratch_curl_z: &mut [f64],
) -> Result<(), EmfOperatorError> {
    let n = grid.validate()?;
    mat.validate(profile)?;
    admit_yee_timestep(grid, mat, dt)?;

    if jx.len() != n || jy.len() != n || jz.len() != n {
        return Err(EmfOperatorError::SliceLengthMismatch {
            expected: n,
            provided: jx.len(),
        });
    }

    let eps = mat.epsilon_f_per_m();
    let mu = mat.mu_h_per_m();
    let loss_half = 0.5 * mat.sigma_s_per_m * dt / eps;
    let c_ee = (1.0 - loss_half) / (1.0 + loss_half);
    let c_eh = (dt / eps) / (1.0 + loss_half);
    let c_he = dt / mu;

    yee_curl_h_2nd(
        grid,
        bc,
        hx,
        hy,
        hz,
        scratch_curl_x,
        scratch_curl_y,
        scratch_curl_z,
    )?;
    for i in 0..n {
        ex[i] = c_ee * ex[i] + c_eh * (scratch_curl_x[i] - jx[i]);
        ey[i] = c_ee * ey[i] + c_eh * (scratch_curl_y[i] - jy[i]);
        ez[i] = c_ee * ez[i] + c_eh * (scratch_curl_z[i] - jz[i]);
    }

    yee_curl_e_2nd(
        grid,
        bc,
        ex,
        ey,
        ez,
        scratch_curl_x,
        scratch_curl_y,
        scratch_curl_z,
    )?;
    for i in 0..n {
        hx[i] -= c_he * scratch_curl_x[i];
        hy[i] -= c_he * scratch_curl_y[i];
        hz[i] -= c_he * scratch_curl_z[i];
    }

    Ok(())
}

/// Result of a periodic 3D Yee plane-wave propagation and exact-time phase evaluation at $T = L / c_0$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YeePlaneWaveResult {
    pub n_grid: usize,
    pub dt: f64,
    pub cfl_limit: f64,
    pub omega_analytical_discrete: f64,
    pub omega_measured_one_step: f64,
    pub one_step_rel_error: f64,
    pub phase_at_period_rad: f64,
    pub continuum_phase_error_rad: f64,
    pub discrete_oracle_phase_error_rad: f64,
    pub relative_energy_oscillation: f64,
}

/// Evaluate the C02 Mode 2 periodic plane-wave fixture on a 3D isotropic Yee lattice.
pub fn run_yee_periodic_plane_wave_fixture(
    n_x: usize,
    l_m: f64,
    courant_fraction: f64,
    mat: &YeeMaterialParams,
) -> Result<YeePlaneWaveResult, EmfOperatorError> {
    if n_x < 4 || !l_m.is_finite() || l_m <= 0.0 || !courant_fraction.is_finite() {
        return Err(EmfOperatorError::NonFiniteInput);
    }
    let dx = l_m / (n_x as f64);
    let grid = Grid3dSpec {
        nx: n_x,
        ny: 2,
        nz: 2,
        dx,
        dy: dx,
        dz: dx,
    };
    let cfl_limit = yee_cfl_limit_3d(&grid, mat)?;
    let dt = courant_fraction * cfl_limit;
    admit_yee_timestep(&grid, mat, dt)?;

    let k_x = 2.0 * core::f64::consts::PI / l_m;
    let omega_num = yee_discrete_omega_rad_per_s(&grid, mat, dt, [k_x, 0.0, 0.0])?;
    let v = mat.phase_velocity_m_per_s()?;
    let eps = mat.epsilon_f_per_m();
    let mu = mat.mu_h_per_m();

    let sin_k_half = (0.5 * k_x * dx).sin();
    let sin_w_half = (0.5 * omega_num * dt).sin();
    let h0_eigen = (dt / (mu * dx)) * (sin_k_half / sin_w_half);

    let total = grid.validate()?;
    let mut ex = vec![0.0_f64; total];
    let mut ey = vec![0.0_f64; total];
    let mut ez = vec![0.0_f64; total];
    let mut hx = vec![0.0_f64; total];
    let mut hy = vec![0.0_f64; total];
    let mut hz = vec![0.0_f64; total];
    let j_zero = vec![0.0_f64; total];
    let mut cx = vec![0.0_f64; total];
    let mut cy = vec![0.0_f64; total];
    let mut cz = vec![0.0_f64; total];

    for iz in 0..grid.nz {
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let c = grid.idx(ix, iy, iz);
                let x_e = (ix as f64) * dx;
                let x_h = ((ix as f64) + 0.5) * dx;
                ey[c] = (k_x * x_e).cos();
                hz[c] = h0_eigen * (k_x * x_h - 0.5 * omega_num * dt).cos();
            }
        }
    }

    let extract_fourier_mode = |ey_slice: &[f64]| -> (f64, f64) {
        let mut c_cos = 0.0_f64;
        let mut c_sin = 0.0_f64;
        for ix in 0..grid.nx {
            let x_e = (ix as f64) * dx;
            let val = ey_slice[grid.idx(ix, 0, 0)];
            c_cos += val * (k_x * x_e).cos();
            c_sin += val * (k_x * x_e).sin();
        }
        let scale = 2.0 / (grid.nx as f64);
        (c_cos * scale, c_sin * scale)
    };

    let collocated_energy = |ey_t: &[f64], hz_prev: &[f64], hz_next: &[f64]| -> f64 {
        let mut u_sum = 0.0_f64;
        for ix in 0..grid.nx {
            let xm = (ix + grid.nx - 1) % grid.nx;
            let e_val = ey_t[grid.idx(ix, 0, 0)];
            let h_prev_col = 0.5 * (hz_prev[grid.idx(xm, 0, 0)] + hz_prev[grid.idx(ix, 0, 0)]);
            let h_next_col = 0.5 * (hz_next[grid.idx(xm, 0, 0)] + hz_next[grid.idx(ix, 0, 0)]);
            let h_col = 0.5 * (h_prev_col + h_next_col);
            u_sum += 0.5 * (eps * e_val * e_val + mu * h_col * h_col);
        }
        u_sum * dx
    };

    let mut hz_minus_half = vec![0.0_f64; total];
    for iz in 0..grid.nz {
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let c = grid.idx(ix, iy, iz);
                let x_h = ((ix as f64) + 0.5) * dx;
                hz_minus_half[c] = h0_eigen * (k_x * x_h + 0.5 * omega_num * dt).cos();
            }
        }
    }
    let u_initial = collocated_energy(&ey, &hz_minus_half, &hz);
    let mut u_min = u_initial;
    let mut u_max = u_initial;

    let mut hz_prev = hz.clone();
    yee_step_leapfrog_in_place(
        &grid,
        YeeSolverProfile::LosslessPeriodicNoPhiV1,
        BoundaryCondition::Periodic,
        mat,
        dt,
        &mut ex,
        &mut ey,
        &mut ez,
        &mut hx,
        &mut hy,
        &mut hz,
        &j_zero,
        &j_zero,
        &j_zero,
        &mut cx,
        &mut cy,
        &mut cz,
    )?;

    let (re1, im1) = extract_fourier_mode(&ey);
    let omega_one_step = im1.atan2(re1) / dt;
    let one_step_rel_error = ((omega_one_step - omega_num) / omega_num).abs();

    let u_step1 = collocated_energy(&ey, &hz_prev, &hz);
    u_min = u_min.min(u_step1);
    u_max = u_max.max(u_step1);

    let period_s = l_m / v;
    let n_floor = (period_s / dt).floor() as usize;
    let alpha = (period_s - (n_floor as f64) * dt) / dt;

    let mut ey_at_n_floor = if n_floor == 1 {
        ey.clone()
    } else {
        vec![0.0_f64; total]
    };

    for step in 2..=(n_floor + 1) {
        hz_prev.copy_from_slice(&hz);
        yee_step_leapfrog_in_place(
            &grid,
            YeeSolverProfile::LosslessPeriodicNoPhiV1,
            BoundaryCondition::Periodic,
            mat,
            dt,
            &mut ex,
            &mut ey,
            &mut ez,
            &mut hx,
            &mut hy,
            &mut hz,
            &j_zero,
            &j_zero,
            &j_zero,
            &mut cx,
            &mut cy,
            &mut cz,
        )?;
        let u_step = collocated_energy(&ey, &hz_prev, &hz);
        u_min = u_min.min(u_step);
        u_max = u_max.max(u_step);
        if step == n_floor {
            ey_at_n_floor.copy_from_slice(&ey);
        }
    }

    let (re_a, im_a) = extract_fourier_mode(&ey_at_n_floor);
    let (re_b, im_b) = extract_fourier_mode(&ey);
    let phase_a = im_a.atan2(re_a).rem_euclid(2.0 * core::f64::consts::PI);
    let mut phase_b = im_b.atan2(re_b).rem_euclid(2.0 * core::f64::consts::PI);
    if phase_b < phase_a {
        phase_b += 2.0 * core::f64::consts::PI;
    }
    let phase_interp = (1.0 - alpha) * phase_a + alpha * phase_b;
    let continuum_phase = 2.0 * core::f64::consts::PI;
    let discrete_expected_phase = omega_num * period_s;

    let continuum_phase_error_rad = (phase_interp - continuum_phase).abs();
    let discrete_oracle_phase_error_rad = (phase_interp - discrete_expected_phase).abs();
    let relative_energy_oscillation = (u_max - u_min) / u_initial.max(1e-30);

    Ok(YeePlaneWaveResult {
        n_grid: n_x,
        dt,
        cfl_limit,
        omega_analytical_discrete: omega_num,
        omega_measured_one_step: omega_one_step,
        one_step_rel_error,
        phase_at_period_rad: phase_interp,
        continuum_phase_error_rad,
        discrete_oracle_phase_error_rad,
        relative_energy_oscillation,
    })
}
