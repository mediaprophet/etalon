//! 3D Staggered Acoustic Pressure-Velocity FDTD Solver (QAC Mode `0x0000`).
//!
//! Implements the C02/C03 contract for `pressure_velocity_continuum`:
//! - 64-byte wire record (`PressureVelocityRecord`) with strict reserved-tail and finiteness gates.
//! - Linear lossless and damped 3D staggered-grid leapfrog time integration on caller-owned slices.
//! - Mandatory 3D Courant-Friedrichs-Lewy (CFL) stability gate before stepping:
//!   $$\Delta t \, c_{\max} \sqrt{\Delta x^{-2} + \Delta y^{-2} + \Delta z^{-2}} \le 1$$
//! - Explicit spatiotemporal interpolation of staggered velocity $(n-\tfrac{1}{2}, n+\tfrac{1}{2})$
//!   and face-centered $(i\pm\tfrac{1}{2})$ to cell-center $(i,j,k,n\Delta t)$ before evaluating
//!   collocated acoustic energy density $e = \tfrac{1}{2}\rho|\mathbf{v}|^2 + \tfrac{p^2}{2K}$
//!   and instantaneous intensity $\mathbf{I} = p\mathbf{v}$.
//! - Boundary conditions: Periodic, Rigid ($\mathbf{v}\cdot\mathbf{n}=0, r=+1$),
//!   Pressure-Release ($p=0, r=-1$), first-order Engquist-Majda absorbing, and multi-layer
//!   Convolutional Perfectly Matched Layer (CPML) with bit-exact restart serialization.

use core::convert::TryInto;

#[path = "fdtd_fluid_solver.rs"]
mod fdtd_fluid_solver;
pub use fdtd_fluid_solver::*;

/// Canonical QAC mode identifier for `pressure_velocity_continuum`.
pub const MODE_PRESSURE_VELOCITY_CONTINUUM: u16 = 0x0000;
/// Exact record stride in bytes for Mode `0x0000`.
pub const PRESSURE_VELOCITY_RECORD_STRIDE: usize = 64;

/// Validity bit flags for `PressureVelocityRecord`.
pub const PV_VALID_POSITION: u32 = 1 << 0;
/// Validity bit for acoustic pressure perturbation `pressure_pa`.
pub const PV_VALID_PRESSURE: u32 = 1 << 1;
/// Validity bit for particle velocity vector `particle_velocity_m_s`.
pub const PV_VALID_VELOCITY: u32 = 1 << 2;
/// Validity bit for collocated energy density `energy_density_j_m3`.
pub const PV_VALID_ENERGY_DENSITY: u32 = 1 << 3;
/// Validity bit for collocated acoustic intensity `intensity_w_m2`.
pub const PV_VALID_INTENSITY: u32 = 1 << 4;
/// Mask of all defined validity bits in Mode `0x0000`.
pub const PV_DEFINED_VALIDITY_MASK: u32 = PV_VALID_POSITION
    | PV_VALID_PRESSURE
    | PV_VALID_VELOCITY
    | PV_VALID_ENERGY_DENSITY
    | PV_VALID_INTENSITY;

/// Errors raised by QAC numerical continuum and room/dispersion solvers.
#[derive(Debug, Clone, PartialEq)]
pub enum NumericalError {
    /// Physical medium parameter is non-positive or non-finite.
    InvalidMedium(&'static str),
    /// Grid dimensions or spacing are invalid.
    InvalidGrid(&'static str),
    /// Timestep violates the 3D Courant stability bound or is non-positive.
    CourantViolation {
        /// Computed dimensionless Courant number.
        courant: f64,
        /// Maximum admissible Courant number (`1.0`).
        max_admissible: f64,
    },
    /// Caller-supplied state slice length does not match expected cell count.
    BufferLengthMismatch {
        /// Context name of the buffer.
        buffer: &'static str,
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements.
        actual: usize,
    },
    /// Attempted to multiply non-collocated staggered pressure and velocity without interpolation.
    NonCollocatedFieldProduct,
    /// Binary record or restart payload has invalid length, reserved bytes, or non-finite values.
    WireEncoding(&'static str),
    /// Quaternion or frame matrix violates C02-C04 admission tolerance.
    InvalidRotation(&'static str),
    /// Iterative linear solver failed to converge or encountered breakdown.
    SolverConvergence(&'static str),
    /// Room acoustics or attenuation domain error (e.g., absorption >= 1.0 in Eyring).
    DomainError(&'static str),
}

/// Stationary homogeneous fluid medium parameters (Schema 200 subset).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluidMedium {
    /// Mass density $\rho$ in $\text{kg}\cdot\text{m}^{-3}$ ($\rho > 0$).
    pub density_kg_m3: f64,
    /// Adiabatic bulk modulus $K$ in $\text{Pa}$ ($K > 0$).
    pub bulk_modulus_pa: f64,
    /// Optional linear acoustic damping rate $\gamma \ge 0$ in $\text{s}^{-1}$.
    pub damping_per_s: f64,
    /// Material reference ID (Schema 200; `0` = explicitly declared uniform medium in QAC).
    pub material_ref: u32,
}

impl FluidMedium {
    /// Construct and validate a stationary fluid medium.
    pub fn new(
        density_kg_m3: f64,
        bulk_modulus_pa: f64,
        damping_per_s: f64,
        material_ref: u32,
    ) -> Result<Self, NumericalError> {
        if !density_kg_m3.is_finite() || density_kg_m3 <= 0.0 {
            return Err(NumericalError::InvalidMedium(
                "fluid density_kg_m3 must be finite and > 0 (vacuum is a boundary mask, not rho=0)",
            ));
        }
        if !bulk_modulus_pa.is_finite() || bulk_modulus_pa <= 0.0 {
            return Err(NumericalError::InvalidMedium("fluid bulk_modulus_pa must be finite and > 0"));
        }
        if !damping_per_s.is_finite() || damping_per_s < 0.0 {
            return Err(NumericalError::InvalidMedium("fluid damping_per_s must be finite and >= 0"));
        }
        Ok(Self { density_kg_m3, bulk_modulus_pa, damping_per_s, material_ref })
    }

    /// Phase speed of sound $c = \sqrt{K / \rho}$ in $\text{m}\cdot\text{s}^{-1}$.
    #[inline]
    pub fn sound_speed_m_s(&self) -> f64 {
        (self.bulk_modulus_pa / self.density_kg_m3).sqrt()
    }

    /// Characteristic acoustic impedance $Z = \sqrt{\rho K} = \rho c$ in $\text{Pa}\cdot\text{s}\cdot\text{m}^{-1}$ (Rayl).
    #[inline]
    pub fn characteristic_impedance_rayl(&self) -> f64 {
        (self.density_kg_m3 * self.bulk_modulus_pa).sqrt()
    }

    /// Instantaneous collocated acoustic energy density
    /// $e = \frac{1}{2}\rho |\mathbf{v}|^2 + \frac{p^2}{2K}$ in $\text{J}\cdot\text{m}^{-3}$.
    #[inline]
    pub fn energy_density_j_m3(&self, pressure_pa: f64, v: [f64; 3]) -> f64 {
        let v_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        0.5 * self.density_kg_m3 * v_sq + (pressure_pa * pressure_pa) / (2.0 * self.bulk_modulus_pa)
    }

    /// Instantaneous collocated acoustic intensity vector $\mathbf{I} = p\mathbf{v}$ in $\text{W}\cdot\text{m}^{-2}$.
    #[inline]
    pub fn instantaneous_intensity_w_m2(pressure_pa: f64, v: [f64; 3]) -> [f64; 3] {
        [pressure_pa * v[0], pressure_pa * v[1], pressure_pa * v[2]]
    }

    /// Time-averaged harmonic intensity $\langle \mathbf{I} \rangle = \frac{1}{2}\operatorname{Re}(p_{\text{peak}} \mathbf{v}_{\text{peak}}^*)$
    /// for peak-amplitude phasors with relative phase $\Delta\phi$ (in radians).
    #[inline]
    pub fn peak_phasor_mean_intensity_w_m2(
        pressure_peak_pa: f64,
        v_peak: [f64; 3],
        phase_diff_rad: f64,
    ) -> [f64; 3] {
        let f = 0.5 * pressure_peak_pa * phase_diff_rad.cos();
        [f * v_peak[0], f * v_peak[1], f * v_peak[2]]
    }
}

/// Staggered component offsets in fractions of grid spacing and solver clock ticks (AQ-02).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaggeredFluidOffsets {
    /// Spatial offset of pressure in grid cells `(dx, dy, dz)`.
    pub p_space_cell_frac: [f64; 3],
    /// Temporal offset of pressure in solver timesteps.
    pub p_time_step_frac: f64,
    /// Spatial offset of `vx` in grid cells.
    pub vx_space_cell_frac: [f64; 3],
    /// Spatial offset of `vy` in grid cells.
    pub vy_space_cell_frac: [f64; 3],
    /// Spatial offset of `vz` in grid cells.
    pub vz_space_cell_frac: [f64; 3],
    /// Temporal offset of velocity components in solver timesteps.
    pub v_time_step_frac: f64,
}

impl StaggeredFluidOffsets {
    /// Standard Yee/Virieux acoustic staggered grid:
    /// $p$ at $(0,0,0), t=0$; $v_x$ at $(0.5,0,0)$, $v_y$ at $(0,0.5,0)$, $v_z$ at $(0,0,0.5)$ at $t=0.5$.
    pub const fn canonical_staggered() -> Self {
        Self {
            p_space_cell_frac: [0.0, 0.0, 0.0], p_time_step_frac: 0.0,
            vx_space_cell_frac: [0.5, 0.0, 0.0], vy_space_cell_frac: [0.0, 0.5, 0.0],
            vz_space_cell_frac: [0.0, 0.0, 0.5], v_time_step_frac: 0.5,
        }
    }

    /// Already collocated cell-center representation (all offsets zero).
    pub const fn collocated() -> Self {
        Self {
            p_space_cell_frac: [0.0, 0.0, 0.0], p_time_step_frac: 0.0,
            vx_space_cell_frac: [0.0, 0.0, 0.0], vy_space_cell_frac: [0.0, 0.0, 0.0],
            vz_space_cell_frac: [0.0, 0.0, 0.0], v_time_step_frac: 0.0,
        }
    }

    /// Returns `true` if any velocity offset differs from the pressure offset.
    #[inline]
    pub fn is_staggered(&self) -> bool {
        (self.p_time_step_frac - self.v_time_step_frac).abs() > 1e-12
            || (self.p_space_cell_frac[0] - self.vx_space_cell_frac[0]).abs() > 1e-12
            || (self.p_space_cell_frac[1] - self.vy_space_cell_frac[1]).abs() > 1e-12
            || (self.p_space_cell_frac[2] - self.vz_space_cell_frac[2]).abs() > 1e-12
    }
}

/// Evaluate collocated energy density and intensity only when the offsets are collocated;
/// rejects direct multiplication of staggered fields without interpolation (C02 Mode 00).
pub fn evaluate_direct_energy_and_intensity(
    medium: &FluidMedium,
    offsets: &StaggeredFluidOffsets,
    pressure_pa: f64,
    velocity_m_s: [f64; 3],
) -> Result<(f64, [f64; 3]), NumericalError> {
    if offsets.is_staggered() {
        return Err(NumericalError::NonCollocatedFieldProduct);
    }
    Ok((
        medium.energy_density_j_m3(pressure_pa, velocity_m_s),
        FluidMedium::instantaneous_intensity_w_m2(pressure_pa, velocity_m_s),
    ))
}

/// Mode `0x0000` (`pressure_velocity_continuum`) 64-byte wire record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureVelocityRecord {
    /// Local Cartesian position in metres (`0..12`).
    pub position_m: [f32; 3],
    /// Acoustic pressure perturbation in Pascals (`12..16`).
    pub pressure_pa: f32,
    /// Particle velocity vector in $\text{m}\cdot\text{s}^{-1}$ (`16..28`).
    pub particle_velocity_m_s: [f32; 3],
    /// Collocated acoustic energy density in $\text{J}\cdot\text{m}^{-3}$ (`28..32`).
    pub energy_density_j_m3: f32,
    /// Collocated instantaneous intensity in $\text{W}\cdot\text{m}^{-2}$ (`32..44`).
    pub intensity_w_m2: [f32; 3],
    /// Material reference ID (`44..48`).
    pub material_ref: u32,
    /// Bitmask of valid fields (`48..52`).
    pub validity_bits: u32,
}

impl PressureVelocityRecord {
    /// Construct a valid collocated record from physical state and medium.
    pub fn from_collocated_state(
        position_m: [f32; 3],
        pressure_pa: f32,
        particle_velocity_m_s: [f32; 3],
        medium: &FluidMedium,
    ) -> Result<Self, NumericalError> {
        let p64 = pressure_pa as f64;
        let v64 = [particle_velocity_m_s[0] as f64, particle_velocity_m_s[1] as f64, particle_velocity_m_s[2] as f64];
        let e64 = medium.energy_density_j_m3(p64, v64);
        let i64 = FluidMedium::instantaneous_intensity_w_m2(p64, v64);
        let rec = Self {
            position_m, pressure_pa, particle_velocity_m_s,
            energy_density_j_m3: e64 as f32,
            intensity_w_m2: [i64[0] as f32, i64[1] as f32, i64[2] as f32],
            material_ref: medium.material_ref, validity_bits: PV_DEFINED_VALIDITY_MASK,
        };
        rec.validate()?;
        Ok(rec)
    }

    /// Validate C02-C01 finiteness, validity-mask rules, and absent-placeholder zeroing.
    pub fn validate(&self) -> Result<(), NumericalError> {
        if (self.validity_bits & !PV_DEFINED_VALIDITY_MASK) != 0 {
            return Err(NumericalError::WireEncoding("reserved validity_bits set in PressureVelocityRecord"));
        }
        if self.position_m.iter().any(|v| !v.is_finite())
            || !self.pressure_pa.is_finite()
            || self.particle_velocity_m_s.iter().any(|v| !v.is_finite())
            || !self.energy_density_j_m3.is_finite()
            || self.intensity_w_m2.iter().any(|v| !v.is_finite())
        {
            return Err(NumericalError::WireEncoding("non-finite float in PressureVelocityRecord"));
        }
        if (self.validity_bits & PV_VALID_PRESSURE) == 0 && self.pressure_pa.to_bits() != 0 {
            return Err(NumericalError::WireEncoding("invalid pressure lane must carry canonical +0.0 placeholder"));
        }
        if (self.validity_bits & PV_VALID_ENERGY_DENSITY) != 0 && self.energy_density_j_m3 < 0.0 {
            return Err(NumericalError::WireEncoding("acoustic energy_density_j_m3 must be non-negative"));
        }
        Ok(())
    }

    /// Encode into exact 64-byte little-endian wire layout.
    pub fn to_wire_bytes(&self) -> Result<[u8; PRESSURE_VELOCITY_RECORD_STRIDE], NumericalError> {
        self.validate()?;
        let mut out = [0u8; PRESSURE_VELOCITY_RECORD_STRIDE];
        write_f32_le(&mut out, 0, self.position_m[0]);
        write_f32_le(&mut out, 4, self.position_m[1]);
        write_f32_le(&mut out, 8, self.position_m[2]);
        write_f32_le(&mut out, 12, self.pressure_pa);
        write_f32_le(&mut out, 16, self.particle_velocity_m_s[0]);
        write_f32_le(&mut out, 20, self.particle_velocity_m_s[1]);
        write_f32_le(&mut out, 24, self.particle_velocity_m_s[2]);
        write_f32_le(&mut out, 28, self.energy_density_j_m3);
        write_f32_le(&mut out, 32, self.intensity_w_m2[0]);
        write_f32_le(&mut out, 36, self.intensity_w_m2[1]);
        write_f32_le(&mut out, 40, self.intensity_w_m2[2]);
        out[44..48].copy_from_slice(&self.material_ref.to_le_bytes());
        out[48..52].copy_from_slice(&self.validity_bits.to_le_bytes());
        Ok(out)
    }

    /// Decode from exact 64-byte little-endian wire layout.
    pub fn from_wire_bytes(bytes: &[u8]) -> Result<Self, NumericalError> {
        if bytes.len() != PRESSURE_VELOCITY_RECORD_STRIDE {
            return Err(NumericalError::WireEncoding("PressureVelocityRecord must be exactly 64 bytes"));
        }
        if bytes[52..64].iter().any(|&b| b != 0) {
            return Err(NumericalError::WireEncoding("PressureVelocityRecord reserved bytes 52..64 must be zero"));
        }
        let rec = Self {
            position_m: [read_f32_le(bytes, 0), read_f32_le(bytes, 4), read_f32_le(bytes, 8)],
            pressure_pa: read_f32_le(bytes, 12),
            particle_velocity_m_s: [read_f32_le(bytes, 16), read_f32_le(bytes, 20), read_f32_le(bytes, 24)],
            energy_density_j_m3: read_f32_le(bytes, 28),
            intensity_w_m2: [read_f32_le(bytes, 32), read_f32_le(bytes, 36), read_f32_le(bytes, 40)],
            material_ref: u32::from_le_bytes(bytes[44..48].try_into().unwrap()),
            validity_bits: u32::from_le_bytes(bytes[48..52].try_into().unwrap()),
        };
        rec.validate()?;
        Ok(rec)
    }
}

/// Boundary condition kind for the 3D fluid FDTD solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FluidBoundaryCondition {
    /// Periodic wrap-around across opposite domain faces.
    Periodic,
    /// Rigid (sound-hard) boundary: $\mathbf{v}\cdot\mathbf{n} = 0$, pressure reflection $r = +1$.
    Rigid,
    /// Pressure-release (sound-soft / free-surface) boundary: $p = 0$, reflection $r = -1$.
    PressureRelease,
    /// First-order Engquist-Majda (Mur) one-way radiation boundary condition.
    EngquistMajdaFirstOrder,
    /// Multi-layer Convolutional Perfectly Matched Layer (CPML) absorbing boundary.
    Cpml {
        /// Thickness of the absorbing layer in grid cells per face (`1..=16`).
        layers: usize,
        /// Maximum damping conductivity $\sigma_{\max}$ in $\text{s}^{-1}$.
        sigma_max: f64,
        /// Maximum coordinate-stretching factor $\kappa_{\max} \ge 1$.
        kappa_max: f64,
        /// Complex frequency-shift parameter $\alpha_{\max} \ge 0$ in $\text{s}^{-1}$.
        alpha_max: f64,
    },
}

/// Configuration for the 3D staggered-grid fluid acoustic FDTD solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluidFdtdConfig {
    /// Grid cell counts `(nx, ny, nz)`.
    pub dims: [usize; 3],
    /// Grid spacing `(dx, dy, dz)` in metres.
    pub spacing_m: [f64; 3],
    /// Solver timestep $\Delta t$ in seconds.
    pub dt_s: f64,
    /// Fluid medium properties.
    pub medium: FluidMedium,
    /// Boundary condition mode.
    pub boundary: FluidBoundaryCondition,
}

impl FluidFdtdConfig {
    /// Validate grid dimensions, spacing, boundary settings, and the 3D Courant stability gate.
    pub fn validate(&self) -> Result<f64, NumericalError> {
        let [nx, ny, nz] = self.dims;
        if nx < 2 || ny < 2 || nz < 2 {
            return Err(NumericalError::InvalidGrid(
                "FluidFdtdConfig dimensions (nx, ny, nz) must each be >= 2",
            ));
        }
        let [dx, dy, dz] = self.spacing_m;
        if !dx.is_finite() || !dy.is_finite() || !dz.is_finite() || dx <= 0.0 || dy <= 0.0 || dz <= 0.0
        {
            return Err(NumericalError::InvalidGrid(
                "FluidFdtdConfig spacing_m components must be finite and > 0",
            ));
        }
        if !self.dt_s.is_finite() || self.dt_s <= 0.0 {
            return Err(NumericalError::CourantViolation { courant: f64::NAN, max_admissible: 1.0 });
        }
        let c = self.medium.sound_speed_m_s();
        let inv_h = (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz)).sqrt();
        let courant = self.dt_s * c * inv_h;
        if !courant.is_finite() || courant > 1.0 {
            return Err(NumericalError::CourantViolation { courant, max_admissible: 1.0 });
        }
        if let FluidBoundaryCondition::Cpml { layers, sigma_max, kappa_max, alpha_max } =
            self.boundary
        {
            if layers == 0 || layers * 2 >= nx.min(ny).min(nz) {
                return Err(NumericalError::InvalidGrid(
                    "CPML layers must be >= 1 and 2*layers < min(nx, ny, nz)",
                ));
            }
            if !sigma_max.is_finite()
                || sigma_max < 0.0
                || !kappa_max.is_finite()
                || kappa_max < 1.0
                || !alpha_max.is_finite()
                || alpha_max < 0.0
            {
                return Err(NumericalError::InvalidGrid(
                    "CPML parameters require sigma_max>=0, kappa_max>=1, alpha_max>=0",
                ));
            }
        }
        Ok(courant)
    }

    /// Total number of cells `nx * ny * nz`.
    #[inline]
    pub const fn cell_count(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Maximum stable timestep $\Delta t_{\max} = \frac{1}{c\sqrt{\Delta x^{-2}+\Delta y^{-2}+\Delta z^{-2}}}$.
    #[inline]
    pub fn max_stable_dt_s(&self) -> f64 {
        let [dx, dy, dz] = self.spacing_m;
        let inv_h = (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz)).sqrt();
        1.0 / (self.medium.sound_speed_m_s() * inv_h)
    }
}

#[inline]
fn write_f32_le(buf: &mut [u8], offset: usize, val: f32) {
    buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
}

#[inline]
fn read_f32_le(buf: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}
