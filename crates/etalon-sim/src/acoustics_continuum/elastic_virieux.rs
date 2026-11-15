//! 3D Elastic Stress-Velocity Virieux Solver & Tensor Mechanics (QAC Mode `0x0001`).
//!
//! Implements the C02/C03 contract for `elastic_stress_velocity_continuum` and Mode 03
//! normal-incidence impedance interfaces:
//! - 64-byte wire record (`ElasticStressVelocityRecord`) with all 6 independent Cauchy stress
//!   components $(\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{yz},\sigma_{xz})$
//!   (tensile-positive, $p = -\operatorname{tr}(\boldsymbol{\sigma})/3$).
//! - Explicit distinction between tensor small strain $\varepsilon_{ij}$ and engineering Voigt
//!   shear strain $\gamma_{ij} = 2\varepsilon_{ij}$ (C02-C04).
//! - Isotropic Hooke's law $\boldsymbol{\sigma} = \lambda\operatorname{tr}(\boldsymbol{\varepsilon})\mathbf{I} + 2\mu\boldsymbol{\varepsilon}$,
//!   compressional wave speed $c_P = \sqrt{(K + \tfrac{4}{3}\mu)/\rho}$, and shear wave speed
//!   $c_S = \sqrt{\mu/\rho}$.
//! - Active Hamilton `[x, y, z, w]` quaternion rotation with C02-C04 ordered squared-norm admission
//!   (`<= 1e-12`), derived orthogonal rotation $\mathbf{R}$, vector map $\mathbf{R}\mathbf{v}$, and
//!   stress map $\boldsymbol{\sigma}' = \mathbf{R}\boldsymbol{\sigma}\mathbf{R}^T$.
//! - Free-surface vacuum boundary ($\boldsymbol{\sigma}\cdot\mathbf{n} = \mathbf{0}$), fluid-solid
//!   traction continuity, and normal-incidence impedance reflection/transmission.
//! - 3D Virieux staggered-grid velocity-stress leapfrog solver with Courant gate.

use core::convert::TryInto;
use super::fdtd_fluid::NumericalError;

#[path = "elastic_virieux_mechanics.rs"]
mod elastic_virieux_mechanics;
pub use elastic_virieux_mechanics::*;

/// Canonical QAC mode identifier for `elastic_stress_velocity_continuum`.
pub const MODE_ELASTIC_STRESS_VELOCITY_CONTINUUM: u16 = 0x0001;
/// Exact record stride in bytes for Mode `0x0001`.
pub const ELASTIC_STRESS_VELOCITY_RECORD_STRIDE: usize = 64;

/// Validity bit for particle velocity `velocity_m_s`.
pub const ESV_VALID_VELOCITY: u32 = 1 << 0;
/// Validity bit for 6-component Cauchy stress `stress_xx_yy_zz_xy_yz_xz_pa`.
pub const ESV_VALID_STRESS: u32 = 1 << 1;
/// Validity bit for particle displacement `displacement_m`.
pub const ESV_VALID_DISPLACEMENT: u32 = 1 << 2;
/// Mask of all defined validity bits in Mode `0x0001`.
pub const ESV_DEFINED_VALIDITY_MASK: u32 =
    ESV_VALID_VELOCITY | ESV_VALID_STRESS | ESV_VALID_DISPLACEMENT;

/// Mode `0x0001` (`elastic_stress_velocity_continuum`) 64-byte wire record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticStressVelocityRecord {
    /// Particle velocity vector in $\text{m}\cdot\text{s}^{-1}$ (`0..12`).
    pub velocity_m_s: [f32; 3],
    /// Material reference ID (`12..16`).
    pub material_ref: u32,
    /// Six-component Cauchy stress `[xx, yy, zz, xy, yz, xz]` in Pa (`16..40`).
    pub stress_xx_yy_zz_xy_yz_xz_pa: [f32; 6],
    /// Particle displacement vector in metres (`40..52`).
    pub displacement_m: [f32; 3],
    /// Validity bitmask (`52..56`).
    pub validity_bits: u32,
}

impl ElasticStressVelocityRecord {
    /// Validate finiteness, validity bits, and C02-C01 invalid-lane zeroing.
    pub fn validate(&self) -> Result<(), NumericalError> {
        if (self.validity_bits & !ESV_DEFINED_VALIDITY_MASK) != 0 {
            return Err(NumericalError::WireEncoding(
                "reserved validity_bits set in ElasticStressVelocityRecord",
            ));
        }
        if self.velocity_m_s.iter().any(|v| !v.is_finite())
            || self.stress_xx_yy_zz_xy_yz_xz_pa.iter().any(|v| !v.is_finite())
            || self.displacement_m.iter().any(|v| !v.is_finite())
        {
            return Err(NumericalError::WireEncoding(
                "non-finite float in ElasticStressVelocityRecord",
            ));
        }
        if (self.validity_bits & ESV_VALID_STRESS) == 0
            && self.stress_xx_yy_zz_xy_yz_xz_pa.iter().any(|v| v.to_bits() != 0)
        {
            return Err(NumericalError::WireEncoding(
                "invalid stress lane must carry canonical +0.0 placeholders",
            ));
        }
        Ok(())
    }

    /// Mechanical pressure $p = -\operatorname{tr}(\boldsymbol{\sigma})/3$ in Pa.
    #[inline]
    pub fn mechanical_pressure_pa(&self) -> f64 {
        let s = &self.stress_xx_yy_zz_xy_yz_xz_pa;
        -((s[0] as f64) + (s[1] as f64) + (s[2] as f64)) / 3.0
    }

    /// Encode to exact 64-byte little-endian wire layout.
    pub fn to_wire_bytes(
        &self,
    ) -> Result<[u8; ELASTIC_STRESS_VELOCITY_RECORD_STRIDE], NumericalError> {
        self.validate()?;
        let mut out = [0u8; ELASTIC_STRESS_VELOCITY_RECORD_STRIDE];
        for i in 0..3 {
            out[i * 4..i * 4 + 4].copy_from_slice(&self.velocity_m_s[i].to_le_bytes());
        }
        out[12..16].copy_from_slice(&self.material_ref.to_le_bytes());
        for i in 0..6 {
            let off = 16 + i * 4;
            out[off..off + 4].copy_from_slice(&self.stress_xx_yy_zz_xy_yz_xz_pa[i].to_le_bytes());
        }
        for i in 0..3 {
            let off = 40 + i * 4;
            out[off..off + 4].copy_from_slice(&self.displacement_m[i].to_le_bytes());
        }
        out[52..56].copy_from_slice(&self.validity_bits.to_le_bytes());
        Ok(out)
    }

    /// Decode from exact 64-byte little-endian wire layout.
    pub fn from_wire_bytes(bytes: &[u8]) -> Result<Self, NumericalError> {
        if bytes.len() != ELASTIC_STRESS_VELOCITY_RECORD_STRIDE {
            return Err(NumericalError::WireEncoding(
                "ElasticStressVelocityRecord must be exactly 64 bytes",
            ));
        }
        if bytes[56..64].iter().any(|&b| b != 0) {
            return Err(NumericalError::WireEncoding(
                "ElasticStressVelocityRecord reserved bytes 56..64 must be zero",
            ));
        }
        let rf = |off: usize| f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let rec = Self {
            velocity_m_s: [rf(0), rf(4), rf(8)],
            material_ref: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            stress_xx_yy_zz_xy_yz_xz_pa: [rf(16), rf(20), rf(24), rf(28), rf(32), rf(36)],
            displacement_m: [rf(40), rf(44), rf(48)],
            validity_bits: u32::from_le_bytes(bytes[52..56].try_into().unwrap()),
        };
        rec.validate()?;
        Ok(rec)
    }
}

/// Boundary condition for the 3D Virieux staggered-grid elastic solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElasticBoundaryCondition {
    /// Periodic wrap-around across opposite domain faces.
    Periodic,
    /// Vacuum free-surface boundary ($\boldsymbol{\sigma}\cdot\mathbf{n} = \mathbf{0}$).
    FreeSurface,
}

/// Configuration for the 3D Virieux staggered-grid elastic velocity-stress solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticVirieuxConfig {
    /// Grid dimensions `(nx, ny, nz)`.
    pub dims: [usize; 3],
    /// Grid spacing `(dx, dy, dz)` in metres.
    pub spacing_m: [f64; 3],
    /// Timestep $\Delta t$ in seconds.
    pub dt_s: f64,
    /// Isotropic elastic medium parameters.
    pub medium: ElasticMedium,
    /// Boundary condition kind.
    pub boundary: ElasticBoundaryCondition,
}

impl ElasticVirieuxConfig {
    /// Validate grid parameters and the 3D P-wave Courant stability condition:
    /// $\Delta t \, c_P \sqrt{\Delta x^{-2} + \Delta y^{-2} + \Delta z^{-2}} \le 1$.
    pub fn validate(&self) -> Result<f64, NumericalError> {
        let [nx, ny, nz] = self.dims;
        if nx < 2 || ny < 2 || nz < 2 {
            return Err(NumericalError::InvalidGrid(
                "ElasticVirieuxConfig dimensions must each be >= 2",
            ));
        }
        let [dx, dy, dz] = self.spacing_m;
        if !dx.is_finite() || !dy.is_finite() || !dz.is_finite() || dx <= 0.0 || dy <= 0.0 || dz <= 0.0
        {
            return Err(NumericalError::InvalidGrid(
                "ElasticVirieuxConfig spacing_m must be finite and > 0",
            ));
        }
        if !self.dt_s.is_finite() || self.dt_s <= 0.0 {
            return Err(NumericalError::CourantViolation { courant: f64::NAN, max_admissible: 1.0 });
        }
        let cp = self.medium.p_wave_speed_m_s();
        let inv_h = (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz)).sqrt();
        let courant = self.dt_s * cp * inv_h;
        if !courant.is_finite() || courant > 1.0 {
            return Err(NumericalError::CourantViolation { courant, max_admissible: 1.0 });
        }
        Ok(courant)
    }

    /// Total number of grid cells `nx * ny * nz`.
    #[inline]
    pub const fn cell_count(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }
}

/// Complete state of the 3D Virieux staggered-grid velocity-stress solver.
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticVirieuxState {
    /// Step index $n$.
    pub step_index: u64,
    /// Staggered velocity components `vx, vy, vz` in m/s.
    pub vx: Vec<f64>,
    /// Velocity `vy` in m/s.
    pub vy: Vec<f64>,
    /// Velocity `vz` in m/s.
    pub vz: Vec<f64>,
    /// Normal stress $\sigma_{xx}$ at cell center in Pa.
    pub sxx: Vec<f64>,
    /// Normal stress $\sigma_{yy}$ at cell center in Pa.
    pub syy: Vec<f64>,
    /// Normal stress $\sigma_{zz}$ at cell center in Pa.
    pub szz: Vec<f64>,
    /// Shear stress $\sigma_{xy}$ at $(i+1/2, j+1/2, k)$ in Pa.
    pub sxy: Vec<f64>,
    /// Shear stress $\sigma_{yz}$ at $(i, j+1/2, k+1/2)$ in Pa.
    pub syz: Vec<f64>,
    /// Shear stress $\sigma_{xz}$ at $(i+1/2, j, k+1/2)$ in Pa.
    pub sxz: Vec<f64>,
}

impl ElasticVirieuxState {
    /// Allocate a zero-initialized state for the given validated configuration.
    pub fn new(cfg: &ElasticVirieuxConfig) -> Result<Self, NumericalError> {
        cfg.validate()?;
        let n = cfg.cell_count();
        Ok(Self {
            step_index: 0,
            vx: vec![0.0; n],
            vy: vec![0.0; n],
            vz: vec![0.0; n],
            sxx: vec![0.0; n],
            syy: vec![0.0; n],
            szz: vec![0.0; n],
            sxy: vec![0.0; n],
            syz: vec![0.0; n],
            sxz: vec![0.0; n],
        })
    }
}

/// Advance the 3D Virieux staggered-grid elastic state by one full leapfrog step $\Delta t$
/// with zero heap allocations on the hot path.
pub fn step_elastic_virieux_in_place(
    cfg: &ElasticVirieuxConfig,
    state: &mut ElasticVirieuxState,
) -> Result<(), NumericalError> {
    cfg.validate()?;
    let [nx, ny, nz] = cfg.dims;
    let n = cfg.cell_count();
    if [
        state.vx.len(),
        state.vy.len(),
        state.vz.len(),
        state.sxx.len(),
        state.syy.len(),
        state.szz.len(),
        state.sxy.len(),
        state.syz.len(),
        state.sxz.len(),
    ]
    .iter()
    .any(|&len| len != n)
    {
        return Err(NumericalError::BufferLengthMismatch {
            buffer: "ElasticVirieuxState arrays",
            expected: n,
            actual: state.vx.len(),
        });
    }

    let [dx, dy, dz] = cfg.spacing_m;
    let dt = cfg.dt_s;
    let dt_rho = dt / cfg.medium.density_kg_m3;
    let lambda = cfg.medium.lame_lambda_pa();
    let mu = cfg.medium.shear_modulus_pa;
    let l2m = lambda + 2.0 * mu;

    let idx = |ix: usize, iy: usize, iz: usize| -> usize { (ix * ny + iy) * nz + iz };
    let periodic = matches!(cfg.boundary, ElasticBoundaryCondition::Periodic);

    // 1. Update velocities vx(i+1/2,j,k), vy(i,j+1/2,k), vz(i,j,k+1/2) from stress divergence.
    for ix in 0..nx {
        let ixp = if ix + 1 < nx { ix + 1 } else if periodic { 0 } else { ix };
        let ixm = if ix > 0 { ix - 1 } else if periodic { nx - 1 } else { 0 };
        for iy in 0..ny {
            let iyp = if iy + 1 < ny { iy + 1 } else if periodic { 0 } else { iy };
            let iym = if iy > 0 { iy - 1 } else if periodic { ny - 1 } else { 0 };
            for iz in 0..nz {
                let izp = if iz + 1 < nz { iz + 1 } else if periodic { 0 } else { iz };
                let izm = if iz > 0 { iz - 1 } else if periodic { nz - 1 } else { 0 };
                let c = idx(ix, iy, iz);
                state.vx[c] += dt_rho * ((state.sxx[idx(ixp, iy, iz)] - state.sxx[c]) / dx + (state.sxy[c] - state.sxy[idx(ix, iym, iz)]) / dy + (state.sxz[c] - state.sxz[idx(ix, iy, izm)]) / dz);
                state.vy[c] += dt_rho * ((state.sxy[c] - state.sxy[idx(ixm, iy, iz)]) / dx + (state.syy[idx(ix, iyp, iz)] - state.syy[c]) / dy + (state.syz[c] - state.syz[idx(ix, iy, izm)]) / dz);
                state.vz[c] += dt_rho * ((state.sxz[c] - state.sxz[idx(ixm, iy, iz)]) / dx + (state.syz[c] - state.syz[idx(ix, iym, iz)]) / dy + (state.szz[idx(ix, iy, izp)] - state.szz[c]) / dz);
            }
        }
    }

    // 2. Update stresses sxx, syy, szz, sxy, syz, sxz from velocity gradients (Hooke's law rate form).
    for ix in 0..nx {
        let ixp = if ix + 1 < nx { ix + 1 } else if periodic { 0 } else { ix };
        let ixm = if ix > 0 { ix - 1 } else if periodic { nx - 1 } else { 0 };
        for iy in 0..ny {
            let iyp = if iy + 1 < ny { iy + 1 } else if periodic { 0 } else { iy };
            let iym = if iy > 0 { iy - 1 } else if periodic { ny - 1 } else { 0 };
            for iz in 0..nz {
                let izp = if iz + 1 < nz { iz + 1 } else if periodic { 0 } else { iz };
                let izm = if iz > 0 { iz - 1 } else if periodic { nz - 1 } else { 0 };
                let c = idx(ix, iy, iz);

                if !periodic && (ix == 0 || ix + 1 == nx || iy == 0 || iy + 1 == ny || iz == 0 || iz + 1 == nz) {
                    state.sxx[c] = 0.0;
                    state.syy[c] = 0.0;
                    state.szz[c] = 0.0;
                    state.sxy[c] = 0.0;
                    state.syz[c] = 0.0;
                    state.sxz[c] = 0.0;
                    continue;
                }

                let dvx_dx = (state.vx[c] - state.vx[idx(ixm, iy, iz)]) / dx;
                let dvy_dy = (state.vy[c] - state.vy[idx(ix, iym, iz)]) / dy;
                let dvz_dz = (state.vz[c] - state.vz[idx(ix, iy, izm)]) / dz;

                state.sxx[c] += dt * (l2m * dvx_dx + lambda * (dvy_dy + dvz_dz));
                state.syy[c] += dt * (l2m * dvy_dy + lambda * (dvx_dx + dvz_dz));
                state.szz[c] += dt * (l2m * dvz_dz + lambda * (dvx_dx + dvy_dy));
                state.sxy[c] += dt * mu * ((state.vx[idx(ix, iyp, iz)] - state.vx[c]) / dy + (state.vy[idx(ixp, iy, iz)] - state.vy[c]) / dx);
                state.syz[c] += dt * mu * ((state.vy[idx(ix, iy, izp)] - state.vy[c]) / dz + (state.vz[idx(ix, iyp, iz)] - state.vz[c]) / dy);
                state.sxz[c] += dt * mu * ((state.vx[idx(ix, iy, izp)] - state.vx[c]) / dz + (state.vz[idx(ixp, iy, iz)] - state.vz[c]) / dx);
            }
        }
    }

    state.step_index += 1;
    Ok(())
}
