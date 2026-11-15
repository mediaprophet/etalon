//! 3D Roden-Gedney Convolutional PML (`CPML`) Stretched-Coordinate Boundary & Multi-Pole
//! Debye/Drude/Lorentz ADE Coupled Leapfrog Stepper with Bitwise Restart (`Schema 103` Keys `26`/`29`).
//!
//! Closes R03 Gap #3 for electromagnetic dispersive/absorbing continuum solvers by providing:
//! 1. 3D Roden-Gedney CPML stretched-coordinate recursive convolution ($\psi_E, \psi_H$ with
//!    $b_u = \exp(-(\sigma_u/\kappa_u + \alpha_u)\Delta t/\varepsilon_0)$ and $c_u$).
//! 2. Coupled 3D Yee FDTD + CPML + multi-pole Debye/Drude/Lorentz ADE time-stepping.
//! 3. Deterministic little-endian bitwise checkpoint serialization and restoration matching
//!    `Schema 103` key `26` (`ade_auxiliary_state`) and key `29` (`pml_auxiliary_state`),
//!    guaranteeing `max_ulp_diff == 0` trajectory parity across checkpoint/restore.

use super::dispersion_ade::{DebyePole, DrudePole, LorentzPole};
use super::yee_fdtd::{admit_yee_timestep, YeeMaterialParams, YeeSolverProfile};
use super::{BoundaryCondition, EmfOperatorError, Grid3dSpec, EPS0_REF_F_PER_M};

pub const SCHEMA_103_KEY_ADE_AUXILIARY_STATE: u16 = 26;
pub const SCHEMA_103_KEY_PML_AUXILIARY_STATE: u16 = 29;
pub const CPML_ADE_RESTART_MAGIC: [u8; 8] = *b"Q42CPML1";

pub const PSI_XY: usize = 0;
pub const PSI_XZ: usize = 1;
pub const PSI_YZ: usize = 2;
pub const PSI_YX: usize = 3;
pub const PSI_ZX: usize = 4;
pub const PSI_ZY: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpmlGradingSpec {
    pub thickness_cells: usize,
    pub m_order: f64,
    pub sigma_max_s_per_m: f64,
    pub kappa_max: f64,
    pub alpha_max_s_per_m: f64,
}

impl CpmlGradingSpec {
    pub fn validate(&self, grid: &Grid3dSpec) -> Result<(), EmfOperatorError> {
        grid.validate()?;
        if !self.m_order.is_finite()
            || !self.sigma_max_s_per_m.is_finite()
            || !self.kappa_max.is_finite()
            || !self.alpha_max_s_per_m.is_finite()
            || self.m_order < 1.0
            || self.sigma_max_s_per_m < 0.0
            || self.kappa_max < 1.0
            || self.alpha_max_s_per_m < 0.0
        {
            return Err(EmfOperatorError::NonFiniteInput);
        }
        if self.thickness_cells * 2 >= grid.nx.min(grid.ny).min(grid.nz) {
            return Err(EmfOperatorError::InvalidGridDimensions { nx: grid.nx, ny: grid.ny, nz: grid.nz });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CpmlAxisCoeffs {
    pub inv_kappa_e: Vec<f64>,
    pub b_e: Vec<f64>,
    pub c_e: Vec<f64>,
    pub inv_kappa_h: Vec<f64>,
    pub b_h: Vec<f64>,
    pub c_h: Vec<f64>,
}

impl CpmlAxisCoeffs {
    pub fn build(n_cells: usize, dt: f64, spec: &CpmlGradingSpec) -> Self {
        let mut out = Self {
            inv_kappa_e: vec![1.0; n_cells],
            b_e: vec![0.0; n_cells],
            c_e: vec![0.0; n_cells],
            inv_kappa_h: vec![1.0; n_cells],
            b_h: vec![0.0; n_cells],
            c_h: vec![0.0; n_cells],
        };
        if spec.thickness_cells == 0 {
            return out;
        }
        let d = spec.thickness_cells as f64;
        let eval_coeff = |depth: f64| -> (f64, f64, f64) {
            if depth <= 0.0 { return (1.0, 0.0, 0.0); }
            let xi = (depth / d).clamp(0.0, 1.0);
            let xi_m = xi.powf(spec.m_order);
            let sigma = spec.sigma_max_s_per_m * xi_m;
            let kappa = 1.0 + (spec.kappa_max - 1.0) * xi_m;
            let alpha = spec.alpha_max_s_per_m * (1.0 - xi);
            let b = (-(sigma / kappa + alpha) * dt / EPS0_REF_F_PER_M).exp();
            let denom = sigma * kappa + kappa * kappa * alpha;
            let c = if denom > 0.0 { sigma * (b - 1.0) / denom } else { 0.0 };
            (1.0 / kappa, b, c)
        };
        let t = spec.thickness_cells;
        for i in 0..n_cells {
            let de = if i < t { (t - i) as f64 } else if i >= n_cells - t { (i - (n_cells - t) + 1) as f64 } else { 0.0 };
            (out.inv_kappa_e[i], out.b_e[i], out.c_e[i]) = eval_coeff(de);
            let dh = if i < t { (t - i) as f64 - 0.5 } else if i >= n_cells - t { (i - (n_cells - t)) as f64 + 0.5 } else { 0.0 };
            (out.inv_kappa_h[i], out.b_h[i], out.c_h[i]) = eval_coeff(dh);
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdePoleAuxiliaryState {
    pub p_n: [Vec<f64>; 3],
    pub p_nm1: [Vec<f64>; 3],
    pub j_n: [Vec<f64>; 3],
}

impl AdePoleAuxiliaryState {
    pub fn zeros(n: usize) -> Self {
        Self {
            p_n: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            p_nm1: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            j_n: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
        }
    }

    pub fn validate(&self, n: usize) -> Result<(), EmfOperatorError> {
        for s in self.p_n.iter().chain(&self.p_nm1).chain(&self.j_n) {
            if s.len() != n {
                return Err(EmfOperatorError::SliceLengthMismatch { expected: n, provided: s.len() });
            }
            if s.iter().any(|v| !v.is_finite()) {
                return Err(EmfOperatorError::NonFiniteInput);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CpmlAdeState {
    pub grid: Grid3dSpec,
    pub material: YeeMaterialParams,
    pub boundary: BoundaryCondition,
    pub cpml_spec: CpmlGradingSpec,
    pub dt: f64,
    pub step_index: u64,
    pub debye_poles: Vec<DebyePole>,
    pub drude_poles: Vec<DrudePole>,
    pub lorentz_poles: Vec<LorentzPole>,
    pub e: [Vec<f64>; 3],
    pub h: [Vec<f64>; 3],
    pub psi_e: [Vec<f64>; 6],
    pub psi_h: [Vec<f64>; 6],
    pub debye_state: Vec<AdePoleAuxiliaryState>,
    pub drude_state: Vec<AdePoleAuxiliaryState>,
    pub lorentz_state: Vec<AdePoleAuxiliaryState>,
}

impl CpmlAdeState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grid: Grid3dSpec,
        material: YeeMaterialParams,
        boundary: BoundaryCondition,
        cpml_spec: CpmlGradingSpec,
        dt: f64,
        debye_poles: Vec<DebyePole>,
        drude_poles: Vec<DrudePole>,
        lorentz_poles: Vec<LorentzPole>,
    ) -> Result<Self, EmfOperatorError> {
        let n = grid.validate()?;
        material.validate(YeeSolverProfile::ConductivePeriodicNoPhiV1)?;
        cpml_spec.validate(&grid)?;
        admit_yee_timestep(&grid, &material, dt)?;
        for p in &debye_poles { p.validate()?; }
        for p in &drude_poles { p.validate()?; }
        for p in &lorentz_poles { p.validate()?; }
        Ok(Self {
            grid, material, boundary, cpml_spec, dt, step_index: 0,
            debye_state: (0..debye_poles.len()).map(|_| AdePoleAuxiliaryState::zeros(n)).collect(),
            drude_state: (0..drude_poles.len()).map(|_| AdePoleAuxiliaryState::zeros(n)).collect(),
            lorentz_state: (0..lorentz_poles.len()).map(|_| AdePoleAuxiliaryState::zeros(n)).collect(),
            debye_poles, drude_poles, lorentz_poles,
            e: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            h: [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            psi_e: core::array::from_fn(|_| vec![0.0; n]),
            psi_h: core::array::from_fn(|_| vec![0.0; n]),
        })
    }

    pub fn validate(&self) -> Result<usize, EmfOperatorError> {
        let n = self.grid.validate()?;
        self.material.validate(YeeSolverProfile::ConductivePeriodicNoPhiV1)?;
        self.cpml_spec.validate(&self.grid)?;
        admit_yee_timestep(&self.grid, &self.material, self.dt)?;
        for s in self.e.iter().chain(&self.h).chain(&self.psi_e).chain(&self.psi_h) {
            if s.len() != n { return Err(EmfOperatorError::SliceLengthMismatch { expected: n, provided: s.len() }); }
            if s.iter().any(|v| !v.is_finite()) { return Err(EmfOperatorError::NonFiniteInput); }
        }
        if self.debye_poles.len() != self.debye_state.len()
            || self.drude_poles.len() != self.drude_state.len()
            || self.lorentz_poles.len() != self.lorentz_state.len()
        {
            return Err(EmfOperatorError::IncompleteRestartState { reason: "ADE pole count mismatch" });
        }
        for (p, s) in self.debye_poles.iter().zip(&self.debye_state) { p.validate()?; s.validate(n)?; }
        for (p, s) in self.drude_poles.iter().zip(&self.drude_state) { p.validate()?; s.validate(n)?; }
        for (p, s) in self.lorentz_poles.iter().zip(&self.lorentz_state) { p.validate()?; s.validate(n)?; }
        Ok(n)
    }

    pub fn step_in_place(&mut self) -> Result<(), EmfOperatorError> {
        self.validate()?;
        let ax = CpmlAxisCoeffs::build(self.grid.nx, self.dt, &self.cpml_spec);
        let ay = CpmlAxisCoeffs::build(self.grid.ny, self.dt, &self.cpml_spec);
        let az = CpmlAxisCoeffs::build(self.grid.nz, self.dt, &self.cpml_spec);
        let (inv_dx, inv_dy, inv_dz) = (1.0 / self.grid.dx, 1.0 / self.grid.dy, 1.0 / self.grid.dz);
        let (eps, mu) = (self.material.epsilon_f_per_m(), self.material.mu_h_per_m());
        let loss_half = 0.5 * self.material.sigma_s_per_m * self.dt / eps;
        let (c_ee, c_eh, c_he) = ((1.0 - loss_half) / (1.0 + loss_half), (self.dt / eps) / (1.0 + loss_half), self.dt / mu);
        let e_old = self.e.clone();

        for iz in 0..self.grid.nz {
            for iy in 0..self.grid.ny {
                for ix in 0..self.grid.nx {
                    let c = self.grid.idx(ix, iy, iz);
                    let xm = if ix > 0 { Some(ix - 1) } else if self.boundary == BoundaryCondition::Periodic { Some(self.grid.nx - 1) } else { None };
                    let ym = if iy > 0 { Some(iy - 1) } else if self.boundary == BoundaryCondition::Periodic { Some(self.grid.ny - 1) } else { None };
                    let zm = if iz > 0 { Some(iz - 1) } else if self.boundary == BoundaryCondition::Periodic { Some(self.grid.nz - 1) } else { None };

                    let dhz_dy = (self.h[2][c] - ym.map_or(0.0, |y0| self.h[2][self.grid.idx(ix, y0, iz)])) * inv_dy;
                    let dhy_dz = (self.h[1][c] - zm.map_or(0.0, |z0| self.h[1][self.grid.idx(ix, iy, z0)])) * inv_dz;
                    let dhx_dz = (self.h[0][c] - zm.map_or(0.0, |z0| self.h[0][self.grid.idx(ix, iy, z0)])) * inv_dz;
                    let dhz_dx = (self.h[2][c] - xm.map_or(0.0, |x0| self.h[2][self.grid.idx(x0, iy, iz)])) * inv_dx;
                    let dhy_dx = (self.h[1][c] - xm.map_or(0.0, |x0| self.h[1][self.grid.idx(x0, iy, iz)])) * inv_dx;
                    let dhx_dy = (self.h[0][c] - ym.map_or(0.0, |y0| self.h[0][self.grid.idx(ix, y0, iz)])) * inv_dy;

                    self.psi_e[PSI_XY][c] = ay.b_e[iy] * self.psi_e[PSI_XY][c] + ay.c_e[iy] * dhz_dy;
                    self.psi_e[PSI_XZ][c] = az.b_e[iz] * self.psi_e[PSI_XZ][c] + az.c_e[iz] * dhy_dz;
                    self.psi_e[PSI_YZ][c] = az.b_e[iz] * self.psi_e[PSI_YZ][c] + az.c_e[iz] * dhx_dz;
                    self.psi_e[PSI_YX][c] = ax.b_e[ix] * self.psi_e[PSI_YX][c] + ax.c_e[ix] * dhz_dx;
                    self.psi_e[PSI_ZX][c] = ax.b_e[ix] * self.psi_e[PSI_ZX][c] + ax.c_e[ix] * dhy_dx;
                    self.psi_e[PSI_ZY][c] = ay.b_e[iy] * self.psi_e[PSI_ZY][c] + ay.c_e[iy] * dhx_dy;

                    let curl_h = [
                        ay.inv_kappa_e[iy] * dhz_dy - az.inv_kappa_e[iz] * dhy_dz + self.psi_e[PSI_XY][c] - self.psi_e[PSI_XZ][c],
                        az.inv_kappa_e[iz] * dhx_dz - ax.inv_kappa_e[ix] * dhz_dx + self.psi_e[PSI_YZ][c] - self.psi_e[PSI_YX][c],
                        ax.inv_kappa_e[ix] * dhy_dx - ay.inv_kappa_e[iy] * dhx_dy + self.psi_e[PSI_ZX][c] - self.psi_e[PSI_ZY][c],
                    ];
                    for axis in 0..3 {
                        let mut j_ade = 0.0_f64;
                        for s in self.debye_state.iter().chain(&self.drude_state).chain(&self.lorentz_state) {
                            j_ade += s.j_n[axis][c];
                        }
                        self.e[axis][c] = c_ee * self.e[axis][c] + c_eh * (curl_h[axis] - j_ade);
                    }
                }
            }
        }

        let ([ex_n, ey_n, ez_n], [ex_np1, ey_np1, ez_np1]) = (&e_old, &self.e);
        for (pole, st) in self.debye_poles.iter().zip(&mut self.debye_state) {
            let ([px, py, pz], [jx, jy, jz]) = (&mut st.p_n, &mut st.j_n);
            pole.step_ade_in_place(self.dt, ex_n, ey_n, ez_n, ex_np1, ey_np1, ez_np1, px, py, pz, jx, jy, jz)?;
        }
        for (pole, st) in self.drude_poles.iter().zip(&mut self.drude_state) {
            let [jx, jy, jz] = &mut st.j_n;
            pole.step_ade_in_place(self.dt, ex_n, ey_n, ez_n, ex_np1, ey_np1, ez_np1, jx, jy, jz)?;
        }
        for (pole, st) in self.lorentz_poles.iter().zip(&mut self.lorentz_state) {
            let ([px_nm1, py_nm1, pz_nm1], [px_n, py_n, pz_n], [jx, jy, jz]) = (&mut st.p_nm1, &mut st.p_n, &mut st.j_n);
            pole.step_ade_in_place(self.dt, ex_np1, ey_np1, ez_np1, px_nm1, py_nm1, pz_nm1, px_n, py_n, pz_n, jx, jy, jz)?;
        }

        for iz in 0..self.grid.nz {
            for iy in 0..self.grid.ny {
                for ix in 0..self.grid.nx {
                    let c = self.grid.idx(ix, iy, iz);
                    let xp = if ix + 1 < self.grid.nx { Some(ix + 1) } else if self.boundary == BoundaryCondition::Periodic { Some(0) } else { None };
                    let yp = if iy + 1 < self.grid.ny { Some(iy + 1) } else if self.boundary == BoundaryCondition::Periodic { Some(0) } else { None };
                    let zp = if iz + 1 < self.grid.nz { Some(iz + 1) } else if self.boundary == BoundaryCondition::Periodic { Some(0) } else { None };

                    let dez_dy = (yp.map_or(0.0, |y1| self.e[2][self.grid.idx(ix, y1, iz)]) - self.e[2][c]) * inv_dy;
                    let dey_dz = (zp.map_or(0.0, |z1| self.e[1][self.grid.idx(ix, iy, z1)]) - self.e[1][c]) * inv_dz;
                    let dex_dz = (zp.map_or(0.0, |z1| self.e[0][self.grid.idx(ix, iy, z1)]) - self.e[0][c]) * inv_dz;
                    let dez_dx = (xp.map_or(0.0, |x1| self.e[2][self.grid.idx(x1, iy, iz)]) - self.e[2][c]) * inv_dx;
                    let dey_dx = (xp.map_or(0.0, |x1| self.e[1][self.grid.idx(x1, iy, iz)]) - self.e[1][c]) * inv_dx;
                    let dex_dy = (yp.map_or(0.0, |y1| self.e[0][self.grid.idx(ix, y1, iz)]) - self.e[0][c]) * inv_dy;

                    self.psi_h[PSI_XY][c] = ay.b_h[iy] * self.psi_h[PSI_XY][c] + ay.c_h[iy] * dez_dy;
                    self.psi_h[PSI_XZ][c] = az.b_h[iz] * self.psi_h[PSI_XZ][c] + az.c_h[iz] * dey_dz;
                    self.psi_h[PSI_YZ][c] = az.b_h[iz] * self.psi_h[PSI_YZ][c] + az.c_h[iz] * dex_dz;
                    self.psi_h[PSI_YX][c] = ax.b_h[ix] * self.psi_h[PSI_YX][c] + ax.c_h[ix] * dez_dx;
                    self.psi_h[PSI_ZX][c] = ax.b_h[ix] * self.psi_h[PSI_ZX][c] + ax.c_h[ix] * dey_dx;
                    self.psi_h[PSI_ZY][c] = ay.b_h[iy] * self.psi_h[PSI_ZY][c] + ay.c_h[iy] * dex_dy;

                    let curl_e = [
                        ay.inv_kappa_h[iy] * dez_dy - az.inv_kappa_h[iz] * dey_dz + self.psi_h[PSI_XY][c] - self.psi_h[PSI_XZ][c],
                        az.inv_kappa_h[iz] * dex_dz - ax.inv_kappa_h[ix] * dez_dx + self.psi_h[PSI_YZ][c] - self.psi_h[PSI_YX][c],
                        ax.inv_kappa_h[ix] * dey_dx - ay.inv_kappa_h[iy] * dex_dy + self.psi_h[PSI_ZX][c] - self.psi_h[PSI_ZY][c],
                    ];
                    for axis in 0..3 { self.h[axis][c] -= c_he * curl_e[axis]; }
                }
            }
        }
        self.step_index += 1;
        Ok(())
    }
}

fn push_f64_slice_le(dst: &mut Vec<u8>, slice: &[f64]) {
    for &v in slice { dst.extend_from_slice(&v.to_bits().to_le_bytes()); }
}

fn read_u64_le(bytes: &[u8], cursor: &mut usize) -> Result<u64, EmfOperatorError> {
    if *cursor + 8 > bytes.len() {
        return Err(EmfOperatorError::IncompleteRestartState { reason: "truncated u64 in restart stream" });
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[*cursor..*cursor + 8]);
    *cursor += 8;
    Ok(u64::from_le_bytes(b))
}

fn read_f64_le(bytes: &[u8], cursor: &mut usize) -> Result<f64, EmfOperatorError> {
    let val = f64::from_bits(read_u64_le(bytes, cursor)?);
    if !val.is_finite() { return Err(EmfOperatorError::NonFiniteInput); }
    Ok(val)
}

fn read_f64_vec_le(bytes: &[u8], cursor: &mut usize, n: usize) -> Result<Vec<f64>, EmfOperatorError> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n { out.push(read_f64_le(bytes, cursor)?); }
    Ok(out)
}

pub fn serialize_cpml_ade_restart_into(
    state: &CpmlAdeState,
    out: &mut Vec<u8>,
) -> Result<(), EmfOperatorError> {
    state.validate()?;
    out.clear();
    out.extend_from_slice(&CPML_ADE_RESTART_MAGIC);
    out.push(state.boundary as u8);
    for v in [state.grid.nx as u64, state.grid.ny as u64, state.grid.nz as u64] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for f in [state.grid.dx, state.grid.dy, state.grid.dz, state.material.epsilon_r, state.material.mu_r, state.material.sigma_s_per_m] {
        out.extend_from_slice(&f.to_bits().to_le_bytes());
    }
    out.extend_from_slice(&(state.cpml_spec.thickness_cells as u64).to_le_bytes());
    for f in [state.cpml_spec.m_order, state.cpml_spec.sigma_max_s_per_m, state.cpml_spec.kappa_max, state.cpml_spec.alpha_max_s_per_m, state.dt] {
        out.extend_from_slice(&f.to_bits().to_le_bytes());
    }
    out.extend_from_slice(&state.step_index.to_le_bytes());
    for s in state.e.iter().chain(&state.h) { push_f64_slice_le(out, s); }

    out.extend_from_slice(&SCHEMA_103_KEY_PML_AUXILIARY_STATE.to_le_bytes());
    for s in state.psi_e.iter().chain(&state.psi_h) { push_f64_slice_le(out, s); }

    out.extend_from_slice(&SCHEMA_103_KEY_ADE_AUXILIARY_STATE.to_le_bytes());
    for len in [state.debye_poles.len(), state.drude_poles.len(), state.lorentz_poles.len()] {
        out.extend_from_slice(&(len as u64).to_le_bytes());
    }
    for (p, st) in state.debye_poles.iter().zip(&state.debye_state) {
        for f in [p.delta_epsilon, p.tau_s] { out.extend_from_slice(&f.to_bits().to_le_bytes()); }
        for s in st.p_n.iter().chain(&st.j_n) { push_f64_slice_le(out, s); }
    }
    for (p, st) in state.drude_poles.iter().zip(&state.drude_state) {
        for f in [p.omega_p_rad_s, p.gamma_rad_s] { out.extend_from_slice(&f.to_bits().to_le_bytes()); }
        for s in &st.j_n { push_f64_slice_le(out, s); }
    }
    for (p, st) in state.lorentz_poles.iter().zip(&state.lorentz_state) {
        for f in [p.delta_epsilon, p.omega_0_rad_s, p.gamma_rad_s] { out.extend_from_slice(&f.to_bits().to_le_bytes()); }
        for s in st.p_n.iter().chain(&st.p_nm1).chain(&st.j_n) { push_f64_slice_le(out, s); }
    }
    Ok(())
}

pub fn restore_cpml_ade_restart_from(bytes: &[u8]) -> Result<CpmlAdeState, EmfOperatorError> {
    if bytes.len() < 9 || bytes[0..8] != CPML_ADE_RESTART_MAGIC {
        return Err(EmfOperatorError::IncompleteRestartState { reason: "invalid Q42CPML1 magic header" });
    }
    let boundary = match bytes[8] {
        0 => BoundaryCondition::Periodic,
        1 => BoundaryCondition::DirichletPec,
        _ => return Err(EmfOperatorError::IncompleteRestartState { reason: "invalid boundary condition tag" }),
    };
    let mut c = 9usize;
    let grid = Grid3dSpec {
        nx: read_u64_le(bytes, &mut c)? as usize, ny: read_u64_le(bytes, &mut c)? as usize, nz: read_u64_le(bytes, &mut c)? as usize,
        dx: read_f64_le(bytes, &mut c)?, dy: read_f64_le(bytes, &mut c)?, dz: read_f64_le(bytes, &mut c)?,
    };
    let n = grid.validate()?;
    let material = YeeMaterialParams {
        epsilon_r: read_f64_le(bytes, &mut c)?, mu_r: read_f64_le(bytes, &mut c)?, sigma_s_per_m: read_f64_le(bytes, &mut c)?,
    };
    let cpml_spec = CpmlGradingSpec {
        thickness_cells: read_u64_le(bytes, &mut c)? as usize, m_order: read_f64_le(bytes, &mut c)?,
        sigma_max_s_per_m: read_f64_le(bytes, &mut c)?, kappa_max: read_f64_le(bytes, &mut c)?, alpha_max_s_per_m: read_f64_le(bytes, &mut c)?,
    };
    let (dt, step_index) = (read_f64_le(bytes, &mut c)?, read_u64_le(bytes, &mut c)?);
    let e = [read_f64_vec_le(bytes, &mut c, n)?, read_f64_vec_le(bytes, &mut c, n)?, read_f64_vec_le(bytes, &mut c, n)?];
    let h = [read_f64_vec_le(bytes, &mut c, n)?, read_f64_vec_le(bytes, &mut c, n)?, read_f64_vec_le(bytes, &mut c, n)?];
    if c + 2 > bytes.len() || u16::from_le_bytes([bytes[c], bytes[c + 1]]) != SCHEMA_103_KEY_PML_AUXILIARY_STATE {
        return Err(EmfOperatorError::IncompleteRestartState { reason: "missing Schema 103 key 29 (pml_auxiliary_state)" });
    }
    c += 2;
    let psi_e = [
        read_f64_vec_le(bytes, &mut c, n)?, read_f64_vec_le(bytes, &mut c, n)?, read_f64_vec_le(bytes, &mut c, n)?,
        read_f64_vec_le(bytes, &mut c, n)?, read_f64_vec_le(bytes, &mut c, n)?, read_f64_vec_le(bytes, &mut c, n)?,
    ];
    let psi_h = [
        read_f64_vec_le(bytes, &mut c, n)?, read_f64_vec_le(bytes, &mut c, n)?, read_f64_vec_le(bytes, &mut c, n)?,
        read_f64_vec_le(bytes, &mut c, n)?, read_f64_vec_le(bytes, &mut c, n)?, read_f64_vec_le(bytes, &mut c, n)?,
    ];
    if c + 2 > bytes.len() || u16::from_le_bytes([bytes[c], bytes[c + 1]]) != SCHEMA_103_KEY_ADE_AUXILIARY_STATE {
        return Err(EmfOperatorError::IncompleteRestartState { reason: "missing Schema 103 key 26 (ade_auxiliary_state)" });
    }
    c += 2;
    let (n_debye, n_drude, n_lorentz) = (read_u64_le(bytes, &mut c)? as usize, read_u64_le(bytes, &mut c)? as usize, read_u64_le(bytes, &mut c)? as usize);
    let (mut debye_poles, mut debye_state) = (Vec::with_capacity(n_debye), Vec::with_capacity(n_debye));
    for _ in 0..n_debye {
        debye_poles.push(DebyePole { delta_epsilon: read_f64_le(bytes, &mut c)?, tau_s: read_f64_le(bytes, &mut c)? });
        let mut st = AdePoleAuxiliaryState::zeros(n);
        for axis in 0..3 { st.p_n[axis] = read_f64_vec_le(bytes, &mut c, n)?; }
        for axis in 0..3 { st.j_n[axis] = read_f64_vec_le(bytes, &mut c, n)?; }
        debye_state.push(st);
    }
    let (mut drude_poles, mut drude_state) = (Vec::with_capacity(n_drude), Vec::with_capacity(n_drude));
    for _ in 0..n_drude {
        drude_poles.push(DrudePole { omega_p_rad_s: read_f64_le(bytes, &mut c)?, gamma_rad_s: read_f64_le(bytes, &mut c)? });
        let mut st = AdePoleAuxiliaryState::zeros(n);
        for axis in 0..3 { st.j_n[axis] = read_f64_vec_le(bytes, &mut c, n)?; }
        drude_state.push(st);
    }
    let (mut lorentz_poles, mut lorentz_state) = (Vec::with_capacity(n_lorentz), Vec::with_capacity(n_lorentz));
    for _ in 0..n_lorentz {
        lorentz_poles.push(LorentzPole { delta_epsilon: read_f64_le(bytes, &mut c)?, omega_0_rad_s: read_f64_le(bytes, &mut c)?, gamma_rad_s: read_f64_le(bytes, &mut c)? });
        let mut st = AdePoleAuxiliaryState::zeros(n);
        for axis in 0..3 { st.p_n[axis] = read_f64_vec_le(bytes, &mut c, n)?; }
        for axis in 0..3 { st.p_nm1[axis] = read_f64_vec_le(bytes, &mut c, n)?; }
        for axis in 0..3 { st.j_n[axis] = read_f64_vec_le(bytes, &mut c, n)?; }
        lorentz_state.push(st);
    }
    if c != bytes.len() {
        return Err(EmfOperatorError::IncompleteRestartState { reason: "trailing bytes in CPML+ADE restart stream" });
    }
    let restored = CpmlAdeState {
        grid, material, boundary, cpml_spec, dt, step_index,
        debye_poles, drude_poles, lorentz_poles,
        e, h, psi_e, psi_h, debye_state, drude_state, lorentz_state,
    };
    restored.validate()?;
    Ok(restored)
}

pub fn verify_cpml_ade_restart_parity(
    initial: &CpmlAdeState,
    half_steps: usize,
) -> Result<u64, EmfOperatorError> {
    let (mut direct, mut split) = (initial.clone(), initial.clone());
    for _ in 0..half_steps {
        direct.step_in_place()?;
        split.step_in_place()?;
    }
    let mut buf = Vec::new();
    serialize_cpml_ade_restart_into(&split, &mut buf)?;
    let mut restored = restore_cpml_ade_restart_from(&buf)?;
    for _ in 0..half_steps {
        direct.step_in_place()?;
        restored.step_in_place()?;
    }
    let mut max_ulp = 0u64;
    for (a, b) in direct.e.iter().chain(&direct.h).chain(&direct.psi_e).chain(&direct.psi_h)
        .zip(restored.e.iter().chain(&restored.h).chain(&restored.psi_e).chain(&restored.psi_h))
    {
        for (&x, &y) in a.iter().zip(b.iter()) {
            max_ulp = max_ulp.max(x.to_bits().abs_diff(y.to_bits()));
        }
    }
    Ok(max_ulp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpml_ade_checkpoint_restore_is_bitwise_identical() {
        let grid = Grid3dSpec { nx: 6, ny: 6, nz: 6, dx: 1e-3, dy: 1e-3, dz: 1e-3 };
        let mat = YeeMaterialParams { epsilon_r: 2.25, mu_r: 1.0, sigma_s_per_m: 0.01 };
        let cpml = CpmlGradingSpec { thickness_cells: 2, m_order: 3.0, sigma_max_s_per_m: 1.2, kappa_max: 5.0, alpha_max_s_per_m: 0.05 };
        let mut st = CpmlAdeState::new(
            grid, mat, BoundaryCondition::DirichletPec, cpml, 1e-12,
            vec![DebyePole { delta_epsilon: 0.5, tau_s: 5e-12 }],
            vec![DrudePole { omega_p_rad_s: 2e10, gamma_rad_s: 1e9 }],
            vec![LorentzPole { delta_epsilon: 0.2, omega_0_rad_s: 5e10, gamma_rad_s: 2e9 }],
        ).unwrap();
        let center = grid.idx(3, 3, 3);
        st.e[2][center] = 1.0;
        st.h[0][center] = 2.5e-3;
        assert_eq!(verify_cpml_ade_restart_parity(&st, 4).unwrap(), 0);
    }
}
