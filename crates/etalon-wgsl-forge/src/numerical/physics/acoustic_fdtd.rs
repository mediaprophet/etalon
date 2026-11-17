//! 3D Staggered Acoustic Fluid Pressure-Velocity & Elastic Virieux Cauchy Stress-Velocity FDTD Kernels.
//!
//! Provides certified `@compute @workgroup_size(64)` WGSL kernels [`ACOUSTIC_FLUID_FDTD_WGSL`]
//! (Mode `0x0000` `pressure_velocity_continuum`) and [`ELASTIC_VIRIEUX_FDTD_WGSL`]
//! (Mode `0x0001` `elastic_stress_velocity_continuum`), exact CPU reference steppers
//! [`acoustic_fluid_fdtd_step_cpu`] and [`elastic_virieux_fdtd_step_cpu`], 3D Courant and capacity
//! validators, and GPU dispatch helpers [`acoustic_fluid_fdtd_step_with_context`] and
//! [`elastic_virieux_fdtd_step_with_context`].

use crate::numerical::ForgeError;

#[path = "acoustic_fdtd_helpers.rs"]
mod acoustic_fdtd_helpers;
pub use acoustic_fdtd_helpers::*;

/// Scalars per cell in the flat 3D fluid FDTD state buffer: `[p, vx, vy, vz]`.
pub const ACOUSTIC_FLUID_CELL_STRIDE: usize = 4;
/// Entry-point name of [`ACOUSTIC_FLUID_FDTD_WGSL`].
pub const ACOUSTIC_FLUID_FDTD_ENTRY: &str = "acoustic_fluid_fdtd_step";
/// Scalars per cell in the flat 3D elastic Virieux state buffer: `[vx, vy, vz, sxx, syy, szz, sxy, syz, sxz]`.
pub const ELASTIC_VIRIEUX_CELL_STRIDE: usize = 9;
/// Entry-point name of [`ELASTIC_VIRIEUX_FDTD_WGSL`].
pub const ELASTIC_VIRIEUX_FDTD_ENTRY: &str = "elastic_virieux_fdtd_step";
/// Maximum cells per single dispatch to guard 32-bit storage buffer indexing.
pub const MAX_FDTD_GRID_CELLS: usize = 16_777_216;

/// WGSL compute kernel for 3D staggered-grid fluid acoustic pressure-velocity FDTD.
///
/// Each cell `c = (ix * ny + iy) * nz + iz` stores 4 `f32` scalars at `base = c * 4`: `[p, vx, vy, vz]`.
/// `params` layout (12 `f32` scalars): `[nx, ny, nz, dx, dy, dz, dt, rho, bulk_k, damping, boundary_mode, sub_pass]`.
pub const ACOUSTIC_FLUID_FDTD_WGSL: &str = r#"// 3D Staggered Acoustic Fluid Pressure-Velocity FDTD Kernel.
@group(0) @binding(0) var<storage, read> state_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> state_out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;

@compute @workgroup_size(64)
fn acoustic_fluid_fdtd_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx = u32(params[0]);
    let ny = u32(params[1]);
    let nz = u32(params[2]);
    let total = nx * ny * nz;
    let c = gid.x;
    if (c >= total) { return; }

    let iz = c % nz;
    let iy = (c / nz) % ny;
    let ix = c / (ny * nz);
    let base = c * 4u;

    let dx = params[3]; let dy = params[4]; let dz = params[5];
    let dt = params[6]; let rho = params[7]; let bulk_k = params[8];
    let damp = params[9]; let bnd = u32(params[10]); let sub_pass = u32(params[11]);

    let half_damp = 0.5 * damp * dt;
    let decay = (1.0 - half_damp) / (1.0 + half_damp);
    let v_scale = (dt / rho) / (1.0 + half_damp);
    let p_scale = (bulk_k * dt) / (1.0 + half_damp);

    let p_curr = state_in[base + 0u];
    let vx_curr = state_in[base + 1u];
    let vy_curr = state_in[base + 2u];
    let vz_curr = state_in[base + 3u];

    if (sub_pass == 0u) {
        var vx_next = 0.0;
        if (ix + 1u < nx) {
            let p_xp = state_in[((ix + 1u) * ny + iy) * nz * 4u + iz * 4u];
            vx_next = decay * vx_curr - v_scale * ((p_xp - p_curr) / dx);
        } else if (bnd == 0u) {
            let p_x0 = state_in[(0u * ny + iy) * nz * 4u + iz * 4u];
            vx_next = decay * vx_curr - v_scale * ((p_x0 - p_curr) / dx);
        } else if (bnd == 2u) {
            vx_next = decay * vx_curr - v_scale * ((-p_curr) / dx);
        }

        var vy_next = 0.0;
        if (iy + 1u < ny) {
            let p_yp = state_in[(ix * ny + (iy + 1u)) * nz * 4u + iz * 4u];
            vy_next = decay * vy_curr - v_scale * ((p_yp - p_curr) / dy);
        } else if (bnd == 0u) {
            let p_y0 = state_in[(ix * ny + 0u) * nz * 4u + iz * 4u];
            vy_next = decay * vy_curr - v_scale * ((p_y0 - p_curr) / dy);
        } else if (bnd == 2u) {
            vy_next = decay * vy_curr - v_scale * ((-p_curr) / dy);
        }

        var vz_next = 0.0;
        if (iz + 1u < nz) {
            let p_zp = state_in[(ix * ny + iy) * nz * 4u + (iz + 1u) * 4u];
            vz_next = decay * vz_curr - v_scale * ((p_zp - p_curr) / dz);
        } else if (bnd == 0u) {
            let p_z0 = state_in[(ix * ny + iy) * nz * 4u + 0u];
            vz_next = decay * vz_curr - v_scale * ((p_z0 - p_curr) / dz);
        } else if (bnd == 2u) {
            vz_next = decay * vz_curr - v_scale * ((-p_curr) / dz);
        }

        state_out[base + 0u] = p_curr;
        state_out[base + 1u] = vx_next;
        state_out[base + 2u] = vy_next;
        state_out[base + 3u] = vz_next;
    } else {
        let on_outer = (ix == 0u) || (ix + 1u == nx) || (iy == 0u) || (iy + 1u == ny) || (iz == 0u) || (iz + 1u == nz);
        if (bnd == 2u && on_outer) {
            state_out[base + 0u] = 0.0;
            state_out[base + 1u] = vx_curr;
            state_out[base + 2u] = vy_curr;
            state_out[base + 3u] = vz_curr;
            return;
        }
        var vx_m = 0.0;
        if (ix > 0u) { vx_m = state_in[((ix - 1u) * ny + iy) * nz * 4u + iz * 4u + 1u]; }
        else if (bnd == 0u) { vx_m = state_in[((nx - 1u) * ny + iy) * nz * 4u + iz * 4u + 1u]; }
        else if (bnd == 2u) { vx_m = vx_curr; }

        var vy_m = 0.0;
        if (iy > 0u) { vy_m = state_in[(ix * ny + (iy - 1u)) * nz * 4u + iz * 4u + 2u]; }
        else if (bnd == 0u) { vy_m = state_in[(ix * ny + (ny - 1u)) * nz * 4u + iz * 4u + 2u]; }
        else if (bnd == 2u) { vy_m = vy_curr; }

        var vz_m = 0.0;
        if (iz > 0u) { vz_m = state_in[(ix * ny + iy) * nz * 4u + (iz - 1u) * 4u + 3u]; }
        else if (bnd == 0u) { vz_m = state_in[(ix * ny + iy) * nz * 4u + (nz - 1u) * 4u + 3u]; }
        else if (bnd == 2u) { vz_m = vz_curr; }

        let div_v = (vx_curr - vx_m) / dx + (vy_curr - vy_m) / dy + (vz_curr - vz_m) / dz;
        state_out[base + 0u] = decay * p_curr - p_scale * div_v;
        state_out[base + 1u] = vx_curr;
        state_out[base + 2u] = vy_curr;
        state_out[base + 3u] = vz_curr;
    }
}
"#;

/// WGSL compute kernel for 3D Virieux staggered-grid elastic velocity-stress FDTD.
///
/// Each cell `c = (ix * ny + iy) * nz + iz` stores 9 `f32` scalars at `base = c * 9`:
/// `[vx, vy, vz, sxx, syy, szz, sxy, syz, sxz]`.
/// `params` layout (12 `f32` scalars): `[nx, ny, nz, dx, dy, dz, dt, rho, lame_lambda, shear_mu, boundary_mode, sub_pass]`.
pub const ELASTIC_VIRIEUX_FDTD_WGSL: &str = r#"// 3D Virieux Staggered-Grid Elastic Velocity-Stress FDTD Kernel.
@group(0) @binding(0) var<storage, read> state_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> state_out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<f32>;

@compute @workgroup_size(64)
fn elastic_virieux_fdtd_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx = u32(params[0]);
    let ny = u32(params[1]);
    let nz = u32(params[2]);
    let total = nx * ny * nz;
    let c = gid.x;
    if (c >= total) { return; }

    let iz = c % nz;
    let iy = (c / nz) % ny;
    let ix = c / (ny * nz);
    let base = c * 9u;

    let dx = params[3]; let dy = params[4]; let dz = params[5];
    let dt = params[6]; let rho = params[7];
    let lame_lambda = params[8]; let mu = params[9];
    let periodic = u32(params[10]) == 0u;
    let sub_pass = u32(params[11]);

    let ixp = select(min(ix + 1u, nx - 1u), (ix + 1u) % nx, periodic);
    let ixm = select(select(0u, ix - 1u, ix > 0u), (ix + nx - 1u) % nx, periodic);
    let iyp = select(min(iy + 1u, ny - 1u), (iy + 1u) % ny, periodic);
    let iym = select(select(0u, iy - 1u, iy > 0u), (iy + ny - 1u) % ny, periodic);
    let izp = select(min(iz + 1u, nz - 1u), (iz + 1u) % nz, periodic);
    let izm = select(select(0u, iz - 1u, iz > 0u), (iz + nz - 1u) % nz, periodic);

    let b_xp = ((ixp * ny + iy) * nz + iz) * 9u;
    let b_xm = ((ixm * ny + iy) * nz + iz) * 9u;
    let b_yp = ((ix * ny + iyp) * nz + iz) * 9u;
    let b_ym = ((ix * ny + iym) * nz + iz) * 9u;
    let b_zp = ((ix * ny + iy) * nz + izp) * 9u;
    let b_zm = ((ix * ny + iy) * nz + izm) * 9u;

    let vx = state_in[base + 0u]; let vy = state_in[base + 1u]; let vz = state_in[base + 2u];
    let sxx = state_in[base + 3u]; let syy = state_in[base + 4u]; let szz = state_in[base + 5u];
    let sxy = state_in[base + 6u]; let syz = state_in[base + 7u]; let sxz = state_in[base + 8u];

    if (sub_pass == 0u) {
        let dt_rho = dt / rho;
        let dvx = ((state_in[b_xp + 3u] - sxx) / dx) + ((sxy - state_in[b_ym + 6u]) / dy) + ((sxz - state_in[b_zm + 8u]) / dz);
        let dvy = ((sxy - state_in[b_xm + 6u]) / dx) + ((state_in[b_yp + 4u] - syy) / dy) + ((syz - state_in[b_zm + 7u]) / dz);
        let dvz = ((sxz - state_in[b_xm + 8u]) / dx) + ((syz - state_in[b_ym + 7u]) / dy) + ((state_in[b_zp + 5u] - szz) / dz);

        state_out[base + 0u] = vx + dt_rho * dvx;
        state_out[base + 1u] = vy + dt_rho * dvy;
        state_out[base + 2u] = vz + dt_rho * dvz;
        state_out[base + 3u] = sxx; state_out[base + 4u] = syy; state_out[base + 5u] = szz;
        state_out[base + 6u] = sxy; state_out[base + 7u] = syz; state_out[base + 8u] = sxz;
    } else {
        state_out[base + 0u] = vx; state_out[base + 1u] = vy; state_out[base + 2u] = vz;
        let on_outer = (ix == 0u) || (ix + 1u == nx) || (iy == 0u) || (iy + 1u == ny) || (iz == 0u) || (iz + 1u == nz);
        if (!periodic && on_outer) {
            state_out[base + 3u] = 0.0; state_out[base + 4u] = 0.0; state_out[base + 5u] = 0.0;
            state_out[base + 6u] = 0.0; state_out[base + 7u] = 0.0; state_out[base + 8u] = 0.0;
            return;
        }
        let l2m = lame_lambda + 2.0 * mu;
        let dvx_dx = (vx - state_in[b_xm + 0u]) / dx;
        let dvy_dy = (vy - state_in[b_ym + 1u]) / dy;
        let dvz_dz = (vz - state_in[b_zm + 2u]) / dz;

        state_out[base + 3u] = sxx + dt * (l2m * dvx_dx + lame_lambda * (dvy_dy + dvz_dz));
        state_out[base + 4u] = syy + dt * (l2m * dvy_dy + lame_lambda * (dvx_dx + dvz_dz));
        state_out[base + 5u] = szz + dt * (l2m * dvz_dz + lame_lambda * (dvx_dx + dvy_dy));
        state_out[base + 6u] = sxy + dt * mu * ((state_in[b_yp + 0u] - vx) / dy + (state_in[b_xp + 1u] - vy) / dx);
        state_out[base + 7u] = syz + dt * mu * ((state_in[b_zp + 1u] - vy) / dz + (state_in[b_yp + 2u] - vz) / dy);
        state_out[base + 8u] = sxz + dt * mu * ((state_in[b_zp + 0u] - vx) / dz + (state_in[b_xp + 2u] - vz) / dx);
    }
}
"#;

/// Boundary mode for GPU/CPU `f32` staggered fluid FDTD stepping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AcousticFluidGpuBoundary {
    /// Periodic wrap-around across all three axes (`0`).
    Periodic = 0,
    /// Rigid sound-hard outer boundary (`1`).
    Rigid = 1,
    /// Pressure-release sound-soft outer boundary (`2`).
    PressureRelease = 2,
}

/// Parameters for a 3D staggered fluid FDTD leapfrog step (`ACOUSTIC_FLUID_FDTD_WGSL`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticFluidFdtdGpuParams {
    /// Grid dimensions `[nx, ny, nz]`.
    pub dims: [u32; 3],
    /// Grid spacing `[dx, dy, dz]` in metres.
    pub spacing_m: [f32; 3],
    /// Timestep `dt` in seconds.
    pub dt_s: f32,
    /// Mass density $\rho > 0$ in $\text{kg}\cdot\text{m}^{-3}$.
    pub density_kg_m3: f32,
    /// Adiabatic bulk modulus $K > 0$ in $\text{Pa}$.
    pub bulk_modulus_pa: f32,
    /// Linear damping rate $\gamma \ge 0$ in $\text{s}^{-1}$.
    pub damping_per_s: f32,
    /// Boundary condition mode.
    pub boundary: AcousticFluidGpuBoundary,
}

impl AcousticFluidFdtdGpuParams {
    /// Validate dimensions, medium positivity, and the 3D Courant stability bound.
    pub fn validate(&self) -> Result<f32, ForgeError> {
        let [nx, ny, nz] = self.dims;
        if nx < 2 || ny < 2 || nz < 2 {
            return Err(ForgeError::GpuValidation("AcousticFluidFdtdGpuParams dims must each be >= 2".to_string()));
        }
        let cells = (nx as usize)
            .checked_mul(ny as usize)
            .and_then(|v| v.checked_mul(nz as usize))
            .ok_or_else(|| ForgeError::GpuValidation("grid cell count overflow".to_string()))?;
        if cells > MAX_FDTD_GRID_CELLS {
            return Err(ForgeError::GpuValidation(format!("grid cell count {cells} exceeds MAX_FDTD_GRID_CELLS ({MAX_FDTD_GRID_CELLS})")));
        }
        let [dx, dy, dz] = self.spacing_m;
        if !dx.is_finite() || !dy.is_finite() || !dz.is_finite() || dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(ForgeError::GpuValidation("spacing_m components must be finite and > 0".to_string()));
        }
        if !self.density_kg_m3.is_finite() || self.density_kg_m3 <= 0.0 {
            return Err(ForgeError::GpuValidation("density_kg_m3 must be finite and > 0".to_string()));
        }
        if !self.bulk_modulus_pa.is_finite() || self.bulk_modulus_pa <= 0.0 {
            return Err(ForgeError::GpuValidation("bulk_modulus_pa must be finite and > 0".to_string()));
        }
        if !self.damping_per_s.is_finite() || self.damping_per_s < 0.0 {
            return Err(ForgeError::GpuValidation("damping_per_s must be finite and >= 0".to_string()));
        }
        if !self.dt_s.is_finite() || self.dt_s <= 0.0 {
            return Err(ForgeError::GpuValidation("dt_s must be finite and > 0".to_string()));
        }
        let c = (self.bulk_modulus_pa / self.density_kg_m3).sqrt();
        let inv_h = (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz)).sqrt();
        let courant = self.dt_s * c * inv_h;
        if !courant.is_finite() || courant > 1.0 {
            return Err(ForgeError::GpuValidation(format!("3D fluid Courant violation: courant={courant} > 1.0")));
        }
        Ok(courant)
    }

    /// Total cell count `nx * ny * nz`.
    #[inline]
    pub fn cell_count(&self) -> usize {
        (self.dims[0] as usize) * (self.dims[1] as usize) * (self.dims[2] as usize)
    }

    #[inline]
    pub(crate) fn to_uniform_words(self, sub_pass: u32) -> [f32; 12] {
        [
            self.dims[0] as f32, self.dims[1] as f32, self.dims[2] as f32,
            self.spacing_m[0], self.spacing_m[1], self.spacing_m[2],
            self.dt_s, self.density_kg_m3, self.bulk_modulus_pa,
            self.damping_per_s, (self.boundary as u32) as f32, sub_pass as f32,
        ]
    }
}

/// Boundary mode for GPU/CPU `f32` Virieux elastic velocity-stress stepping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ElasticVirieuxGpuBoundary {
    /// Periodic wrap-around across all three axes (`0`).
    Periodic = 0,
    /// Vacuum free-surface outer boundary $\boldsymbol{\sigma}\cdot\mathbf{n} = \mathbf{0}$ (`1`).
    FreeSurface = 1,
}

/// Parameters for a 3D Virieux staggered elastic velocity-stress step (`ELASTIC_VIRIEUX_FDTD_WGSL`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticVirieuxFdtdGpuParams {
    /// Grid dimensions `[nx, ny, nz]`.
    pub dims: [u32; 3],
    /// Grid spacing `[dx, dy, dz]` in metres.
    pub spacing_m: [f32; 3],
    /// Timestep `dt` in seconds.
    pub dt_s: f32,
    /// Mass density $\rho > 0$ in $\text{kg}\cdot\text{m}^{-3}$.
    pub density_kg_m3: f32,
    /// Bulk modulus $K > 0$ in $\text{Pa}$.
    pub bulk_modulus_pa: f32,
    /// Shear modulus $\mu \ge 0$ in $\text{Pa}$.
    pub shear_modulus_pa: f32,
    /// Boundary condition mode.
    pub boundary: ElasticVirieuxGpuBoundary,
}

impl ElasticVirieuxFdtdGpuParams {
    /// First LamÃ© parameter $\lambda = K - \frac{2}{3}\mu$ in Pa.
    #[inline]
    pub fn lame_lambda_pa(&self) -> f32 {
        self.bulk_modulus_pa - (2.0 / 3.0) * self.shear_modulus_pa
    }

    /// Compressional P-wave speed $c_P = \sqrt{(K + \frac{4}{3}\mu)/\rho}$ in m/s.
    #[inline]
    pub fn p_wave_speed_m_s(&self) -> f32 {
        ((self.bulk_modulus_pa + (4.0 / 3.0) * self.shear_modulus_pa) / self.density_kg_m3).sqrt()
    }

    /// Validate dimensions, medium positivity, and the 3D P-wave Courant stability bound.
    pub fn validate(&self) -> Result<f32, ForgeError> {
        let [nx, ny, nz] = self.dims;
        if nx < 2 || ny < 2 || nz < 2 {
            return Err(ForgeError::GpuValidation("ElasticVirieuxFdtdGpuParams dims must each be >= 2".to_string()));
        }
        let cells = (nx as usize)
            .checked_mul(ny as usize)
            .and_then(|v| v.checked_mul(nz as usize))
            .ok_or_else(|| ForgeError::GpuValidation("grid cell count overflow".to_string()))?;
        if cells > MAX_FDTD_GRID_CELLS {
            return Err(ForgeError::GpuValidation(format!("grid cell count {cells} exceeds MAX_FDTD_GRID_CELLS ({MAX_FDTD_GRID_CELLS})")));
        }
        let [dx, dy, dz] = self.spacing_m;
        if !dx.is_finite() || !dy.is_finite() || !dz.is_finite() || dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(ForgeError::GpuValidation("spacing_m components must be finite and > 0".to_string()));
        }
        if !self.density_kg_m3.is_finite() || self.density_kg_m3 <= 0.0 {
            return Err(ForgeError::GpuValidation("density_kg_m3 must be finite and > 0".to_string()));
        }
        if !self.bulk_modulus_pa.is_finite() || self.bulk_modulus_pa <= 0.0 {
            return Err(ForgeError::GpuValidation("bulk_modulus_pa must be finite and > 0".to_string()));
        }
        if !self.shear_modulus_pa.is_finite() || self.shear_modulus_pa < 0.0 {
            return Err(ForgeError::GpuValidation("shear_modulus_pa must be finite and >= 0".to_string()));
        }
        if !self.dt_s.is_finite() || self.dt_s <= 0.0 {
            return Err(ForgeError::GpuValidation("dt_s must be finite and > 0".to_string()));
        }
        let cp = self.p_wave_speed_m_s();
        let inv_h = (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz)).sqrt();
        let courant = self.dt_s * cp * inv_h;
        if !courant.is_finite() || courant > 1.0 {
            return Err(ForgeError::GpuValidation(format!("3D elastic P-wave Courant violation: courant={courant} > 1.0")));
        }
        Ok(courant)
    }

    /// Total cell count `nx * ny * nz`.
    #[inline]
    pub fn cell_count(&self) -> usize {
        (self.dims[0] as usize) * (self.dims[1] as usize) * (self.dims[2] as usize)
    }

    #[inline]
    pub(crate) fn to_uniform_words(self, sub_pass: u32) -> [f32; 12] {
        [
            self.dims[0] as f32, self.dims[1] as f32, self.dims[2] as f32,
            self.spacing_m[0], self.spacing_m[1], self.spacing_m[2],
            self.dt_s, self.density_kg_m3, self.lame_lambda_pa(),
            self.shear_modulus_pa, (self.boundary as u32) as f32, sub_pass as f32,
        ]
    }
}
