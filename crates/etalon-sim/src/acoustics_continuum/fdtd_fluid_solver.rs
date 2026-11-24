//! 3D Staggered Acoustic Pressure-Velocity FDTD State, Leapfrog Stepper & Collocation.

use super::*;
use core::convert::TryInto;

/// Complete restartable state for the 3D staggered fluid acoustic FDTD solver.
#[derive(Debug, Clone, PartialEq)]
pub struct FluidFdtdState {
    /// Current step index $n$.
    pub step_index: u64,
    /// Cell-centered pressure $p^n(i,j,k)$ in Pa.
    pub p: Vec<f64>,
    /// Face-centered velocity $v_x^{n+1/2}(i+1/2,j,k)$ in m/s.
    pub vx: Vec<f64>,
    /// Face-centered velocity $v_y^{n+1/2}(i,j+1/2,k)$ in m/s.
    pub vy: Vec<f64>,
    /// Face-centered velocity $v_z^{n+1/2}(i,j,k+1/2)$ in m/s.
    pub vz: Vec<f64>,
    /// Prior half-step velocity $v_x^{n-1/2}$ for exact temporal collocation at $t = n\Delta t$.
    pub prev_vx: Vec<f64>,
    /// Prior half-step velocity $v_y^{n-1/2}$ for exact temporal collocation at $t = n\Delta t$.
    pub prev_vy: Vec<f64>,
    /// Prior half-step velocity $v_z^{n-1/2}$ for exact temporal collocation at $t = n\Delta t$.
    pub prev_vz: Vec<f64>,
    /// CPML auxiliary memory variables $\psi_{v_x}, \psi_{v_y}, \psi_{v_z}, \psi_{p_x}, \psi_{p_y}, \psi_{p_z}$.
    pub cpml_psi: Vec<f64>,
}

const FLUID_RESTART_MAGIC: &[u8; 8] = b"QACFDTD0";

impl FluidFdtdState {
    /// Allocate a zero-initialized state for the given validated solver configuration.
    pub fn new(cfg: &FluidFdtdConfig) -> Result<Self, NumericalError> {
        cfg.validate()?;
        let n = cfg.cell_count();
        let cpml_len = if matches!(cfg.boundary, FluidBoundaryCondition::Cpml { .. }) {
            6 * n
        } else {
            0
        };
        Ok(Self {
            step_index: 0,
            p: vec![0.0; n],
            vx: vec![0.0; n],
            vy: vec![0.0; n],
            vz: vec![0.0; n],
            prev_vx: vec![0.0; n],
            prev_vy: vec![0.0; n],
            prev_vz: vec![0.0; n],
            cpml_psi: vec![0.0; cpml_len],
        })
    }

    /// Serialize complete internal and boundary state for deterministic bitwise restart (C02-C09).
    pub fn encode_restart_bytes(&self) -> Vec<u8> {
        let n = self.p.len() as u64;
        let cpml_n = self.cpml_psi.len() as u64;
        let total_f64 = 7 * (n as usize) + (cpml_n as usize);
        let mut out = Vec::with_capacity(32 + total_f64 * 8);
        out.extend_from_slice(FLUID_RESTART_MAGIC);
        out.extend_from_slice(&self.step_index.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&cpml_n.to_le_bytes());
        for slice in [
            &self.p,
            &self.vx,
            &self.vy,
            &self.vz,
            &self.prev_vx,
            &self.prev_vy,
            &self.prev_vz,
            &self.cpml_psi,
        ] {
            for &val in slice {
                out.extend_from_slice(&val.to_le_bytes());
            }
        }
        out
    }

    /// Reconstruct complete state from deterministic restart bytes.
    pub fn decode_restart_bytes(
        cfg: &FluidFdtdConfig,
        bytes: &[u8],
    ) -> Result<Self, NumericalError> {
        cfg.validate()?;
        if bytes.len() < 32 || &bytes[0..8] != FLUID_RESTART_MAGIC {
            return Err(NumericalError::WireEncoding(
                "invalid FluidFdtdState restart header",
            ));
        }
        let step_index = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let n = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        let cpml_n = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
        let expected_n = cfg.cell_count();
        let expected_cpml = if matches!(cfg.boundary, FluidBoundaryCondition::Cpml { .. }) {
            6 * expected_n
        } else {
            0
        };
        if n != expected_n || cpml_n != expected_cpml {
            return Err(NumericalError::WireEncoding(
                "FluidFdtdState restart dimensions do not match FluidFdtdConfig",
            ));
        }
        if bytes.len() != 32 + (7 * n + cpml_n) * 8 {
            return Err(NumericalError::WireEncoding(
                "FluidFdtdState restart byte length mismatch",
            ));
        }
        let mut cursor = 32usize;
        let mut read_vec = |count: usize| -> Result<Vec<f64>, NumericalError> {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                let val = f64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
                if !val.is_finite() {
                    return Err(NumericalError::WireEncoding(
                        "non-finite value in FluidFdtdState restart bytes",
                    ));
                }
                v.push(val);
                cursor += 8;
            }
            Ok(v)
        };
        Ok(Self {
            step_index,
            p: read_vec(n)?,
            vx: read_vec(n)?,
            vy: read_vec(n)?,
            vz: read_vec(n)?,
            prev_vx: read_vec(n)?,
            prev_vy: read_vec(n)?,
            prev_vz: read_vec(n)?,
            cpml_psi: read_vec(cpml_n)?,
        })
    }
}

/// Advance the 3D staggered-grid fluid FDTD state by one full leapfrog step $\Delta t$
/// with zero heap allocations on the hot path.
pub fn step_fluid_fdtd_in_place(
    cfg: &FluidFdtdConfig,
    state: &mut FluidFdtdState,
) -> Result<(), NumericalError> {
    cfg.validate()?;
    let [nx, ny, nz] = cfg.dims;
    let n = cfg.cell_count();
    if [
        state.p.len(),
        state.vx.len(),
        state.vy.len(),
        state.vz.len(),
        state.prev_vx.len(),
        state.prev_vy.len(),
        state.prev_vz.len(),
    ]
    .iter()
    .any(|&len| len != n)
    {
        return Err(NumericalError::BufferLengthMismatch {
            buffer: "FluidFdtdState primary arrays",
            expected: n,
            actual: state.p.len(),
        });
    }

    state.prev_vx.copy_from_slice(&state.vx);
    state.prev_vy.copy_from_slice(&state.vy);
    state.prev_vz.copy_from_slice(&state.vz);

    let [dx, dy, dz] = cfg.spacing_m;
    let dt = cfg.dt_s;
    let rho = cfg.medium.density_kg_m3;
    let k_mod = cfg.medium.bulk_modulus_pa;
    let damp = cfg.medium.damping_per_s;
    let v_decay = (1.0 - 0.5 * damp * dt) / (1.0 + 0.5 * damp * dt);
    let v_scale = (dt / rho) / (1.0 + 0.5 * damp * dt);
    let p_decay = v_decay;
    let p_scale = (k_mod * dt) / (1.0 + 0.5 * damp * dt);
    let idx = |ix: usize, iy: usize, iz: usize| -> usize { (ix * ny + iy) * nz + iz };

    if matches!(cfg.boundary, FluidBoundaryCondition::Cpml { .. }) && state.cpml_psi.len() != 6 * n
    {
        return Err(NumericalError::BufferLengthMismatch {
            buffer: "FluidFdtdState::cpml_psi",
            expected: 6 * n,
            actual: state.cpml_psi.len(),
        });
    }

    // 1. Update staggered velocities vx(i+1/2,j,k), vy(i,j+1/2,k), vz(i,j,k+1/2).
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let c_idx = idx(ix, iy, iz);
                let p_curr = state.p[c_idx];
                let px = if ix + 1 < nx {
                    Some(state.p[idx(ix + 1, iy, iz)])
                } else {
                    None
                };
                let py = if iy + 1 < ny {
                    Some(state.p[idx(ix, iy + 1, iz)])
                } else {
                    None
                };
                let pz = if iz + 1 < nz {
                    Some(state.p[idx(ix, iy, iz + 1)])
                } else {
                    None
                };
                state.vx[c_idx] = update_face_velocity(
                    state.vx[c_idx],
                    p_curr,
                    px,
                    state.p[idx(0, iy, iz)],
                    ix,
                    nx,
                    dx,
                    dt,
                    v_decay,
                    v_scale,
                    &cfg.boundary,
                    state.cpml_psi.get_mut(c_idx),
                );
                state.vy[c_idx] = update_face_velocity(
                    state.vy[c_idx],
                    p_curr,
                    py,
                    state.p[idx(ix, 0, iz)],
                    iy,
                    ny,
                    dy,
                    dt,
                    v_decay,
                    v_scale,
                    &cfg.boundary,
                    state.cpml_psi.get_mut(n + c_idx),
                );
                state.vz[c_idx] = update_face_velocity(
                    state.vz[c_idx],
                    p_curr,
                    pz,
                    state.p[idx(ix, iy, 0)],
                    iz,
                    nz,
                    dz,
                    dt,
                    v_decay,
                    v_scale,
                    &cfg.boundary,
                    state.cpml_psi.get_mut(2 * n + c_idx),
                );
            }
        }
    }

    // 2. Update cell-centered pressure p(i, j, k).
    let c_sound = cfg.medium.sound_speed_m_s();
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let c_idx = idx(ix, iy, iz);
                let on_outer =
                    ix == 0 || ix + 1 == nx || iy == 0 || iy + 1 == ny || iz == 0 || iz + 1 == nz;
                if matches!(cfg.boundary, FluidBoundaryCondition::PressureRelease) && on_outer {
                    state.p[c_idx] = 0.0;
                    continue;
                }

                let vx_m = minus_velocity(
                    ix,
                    nx,
                    state.vx[c_idx],
                    || state.vx[idx(ix - 1, iy, iz)],
                    || state.vx[idx(nx - 1, iy, iz)],
                    &cfg.boundary,
                );
                let vy_m = minus_velocity(
                    iy,
                    ny,
                    state.vy[c_idx],
                    || state.vy[idx(ix, iy - 1, iz)],
                    || state.vy[idx(ix, ny - 1, iz)],
                    &cfg.boundary,
                );
                let vz_m = minus_velocity(
                    iz,
                    nz,
                    state.vz[c_idx],
                    || state.vz[idx(ix, iy, iz - 1)],
                    || state.vz[idx(ix, iy, nz - 1)],
                    &cfg.boundary,
                );

                let dvx_dx = (state.vx[c_idx] - vx_m) / dx;
                let dvy_dy = (state.vy[c_idx] - vy_m) / dy;
                let dvz_dz = (state.vz[c_idx] - vz_m) / dz;

                let div_v = if let FluidBoundaryCondition::Cpml {
                    layers,
                    sigma_max,
                    kappa_max,
                    alpha_max,
                } = cfg.boundary
                {
                    let gx = apply_cpml_grad(
                        dvx_dx,
                        ix as f64,
                        nx,
                        layers,
                        dx,
                        dt,
                        sigma_max,
                        kappa_max,
                        alpha_max,
                        &mut state.cpml_psi[3 * n + c_idx],
                    );
                    let gy = apply_cpml_grad(
                        dvy_dy,
                        iy as f64,
                        ny,
                        layers,
                        dy,
                        dt,
                        sigma_max,
                        kappa_max,
                        alpha_max,
                        &mut state.cpml_psi[4 * n + c_idx],
                    );
                    let gz = apply_cpml_grad(
                        dvz_dz,
                        iz as f64,
                        nz,
                        layers,
                        dz,
                        dt,
                        sigma_max,
                        kappa_max,
                        alpha_max,
                        &mut state.cpml_psi[5 * n + c_idx],
                    );
                    gx + gy + gz
                } else {
                    dvx_dx + dvy_dy + dvz_dz
                };

                if matches!(
                    cfg.boundary,
                    FluidBoundaryCondition::EngquistMajdaFirstOrder
                ) && on_outer
                {
                    let i_in = if ix == 0 {
                        1
                    } else if ix + 1 == nx {
                        nx - 2
                    } else {
                        ix
                    };
                    let j_in = if iy == 0 {
                        1
                    } else if iy + 1 == ny {
                        ny - 2
                    } else {
                        iy
                    };
                    let k_in = if iz == 0 {
                        1
                    } else if iz + 1 == nz {
                        nz - 2
                    } else {
                        iz
                    };
                    let p_inner = state.p[idx(i_in, j_in, k_in)];
                    let s = c_sound * dt / dx.min(dy).min(dz);
                    state.p[c_idx] += s * (p_inner - state.p[c_idx]);
                } else {
                    state.p[c_idx] = p_decay * state.p[c_idx] - p_scale * div_v;
                }
            }
        }
    }

    state.step_index += 1;
    Ok(())
}

/// Interpolate staggered velocities in space and time to cell center `(ix, iy, iz)` at step $n$.
pub fn collocate_fluid_cell(
    cfg: &FluidFdtdConfig,
    p_at_n: &[f64],
    v_minus_half: (&[f64], &[f64], &[f64]),
    v_plus_half: (&[f64], &[f64], &[f64]),
    cell: [usize; 3],
) -> Result<(f64, [f64; 3], f64, [f64; 3]), NumericalError> {
    let [nx, ny, nz] = cfg.dims;
    let [ix, iy, iz] = cell;
    if ix >= nx || iy >= ny || iz >= nz {
        return Err(NumericalError::InvalidGrid("cell index out of bounds"));
    }
    let idx = |x: usize, y: usize, z: usize| -> usize { (x * ny + y) * nz + z };
    let c_idx = idx(ix, iy, iz);
    let periodic = matches!(cfg.boundary, FluidBoundaryCondition::Periodic);
    let xm = if ix > 0 {
        ix - 1
    } else if periodic {
        nx - 1
    } else {
        0
    };
    let ym = if iy > 0 {
        iy - 1
    } else if periodic {
        ny - 1
    } else {
        0
    };
    let zm = if iz > 0 {
        iz - 1
    } else if periodic {
        nz - 1
    } else {
        0
    };

    let (vx_m, vy_m, vz_m) = v_minus_half;
    let (vx_p, vy_p, vz_p) = v_plus_half;
    let vx_col = 0.25 * (vx_m[c_idx] + vx_m[idx(xm, iy, iz)] + vx_p[c_idx] + vx_p[idx(xm, iy, iz)]);
    let vy_col = 0.25 * (vy_m[c_idx] + vy_m[idx(ix, ym, iz)] + vy_p[c_idx] + vy_p[idx(ix, ym, iz)]);
    let vz_col = 0.25 * (vz_m[c_idx] + vz_m[idx(ix, iy, zm)] + vz_p[c_idx] + vz_p[idx(ix, iy, zm)]);

    let p_col = p_at_n[c_idx];
    let v_col = [vx_col, vy_col, vz_col];
    let e_col = cfg.medium.energy_density_j_m3(p_col, v_col);
    let i_col = FluidMedium::instantaneous_intensity_w_m2(p_col, v_col);
    Ok((p_col, v_col, e_col, i_col))
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn update_face_velocity(
    v_curr: f64,
    p_curr: f64,
    p_next: Option<f64>,
    p_wrap0: f64,
    coord: usize,
    n_dim: usize,
    dh: f64,
    dt: f64,
    v_decay: f64,
    v_scale: f64,
    boundary: &FluidBoundaryCondition,
    psi_slot: Option<&mut f64>,
) -> f64 {
    if let Some(p_hi) = p_next {
        let dp = (p_hi - p_curr) / dh;
        let grad = if let (
            FluidBoundaryCondition::Cpml {
                layers,
                sigma_max,
                kappa_max,
                alpha_max,
            },
            Some(psi),
        ) = (*boundary, psi_slot)
        {
            apply_cpml_grad(
                dp,
                coord as f64 + 0.5,
                n_dim,
                layers,
                dh,
                dt,
                sigma_max,
                kappa_max,
                alpha_max,
                psi,
            )
        } else {
            dp
        };
        v_decay * v_curr - v_scale * grad
    } else {
        match boundary {
            FluidBoundaryCondition::Periodic => {
                v_decay * v_curr - v_scale * ((p_wrap0 - p_curr) / dh)
            }
            FluidBoundaryCondition::Rigid
            | FluidBoundaryCondition::EngquistMajdaFirstOrder
            | FluidBoundaryCondition::Cpml { .. } => 0.0,
            FluidBoundaryCondition::PressureRelease => {
                v_decay * v_curr - v_scale * ((-p_curr) / dh)
            }
        }
    }
}

#[inline]
fn minus_velocity<F1: FnOnce() -> f64, F2: FnOnce() -> f64>(
    coord: usize,
    _n_dim: usize,
    v_curr: f64,
    prev_cell: F1,
    wrap_last: F2,
    boundary: &FluidBoundaryCondition,
) -> f64 {
    if coord > 0 {
        prev_cell()
    } else {
        match boundary {
            FluidBoundaryCondition::Periodic => wrap_last(),
            FluidBoundaryCondition::Rigid
            | FluidBoundaryCondition::EngquistMajdaFirstOrder
            | FluidBoundaryCondition::Cpml { .. } => 0.0,
            FluidBoundaryCondition::PressureRelease => v_curr,
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn apply_cpml_grad(
    grad: f64,
    coord: f64,
    n_dim: usize,
    layers: usize,
    dh: f64,
    dt: f64,
    sigma_max: f64,
    kappa_max: f64,
    alpha_max: f64,
    psi: &mut f64,
) -> f64 {
    let l = layers as f64;
    let dist = if coord < l {
        (l - coord) / l
    } else if coord > (n_dim - 1) as f64 - l {
        (coord - ((n_dim - 1) as f64 - l)) / l
    } else {
        0.0
    };
    if dist <= 0.0 {
        return grad;
    }
    let d = dist.clamp(0.0, 1.0);
    let d3 = d * d * d;
    let sigma = sigma_max * d3;
    let kappa = 1.0 + (kappa_max - 1.0) * d3;
    let alpha = alpha_max * (1.0 - d);
    let b = (-(sigma / kappa + alpha) * dt).exp();
    let denom = kappa * (sigma + kappa * alpha) * dh;
    let a = if denom.abs() > 1e-30 {
        sigma * (b - 1.0) / denom
    } else {
        0.0
    };
    *psi = b * (*psi) + a * grad;
    grad / kappa + *psi
}
