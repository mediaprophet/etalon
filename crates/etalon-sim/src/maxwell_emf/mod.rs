//! `etalon_sim::maxwell_emf` — 3D Continuum Maxwell Vector Calculus, Cartesian Yee FDTD,
//! Dispersive ADE Media, Poynting Energy Balance, and C02 Electromagnetic Oracles.
//!
//! Pure dependency-free (`core`/`std` + `bytemuck`) continuum electromagnetic simulation
//! substrate consumed by `qualia_core_db::q42::emf_operators`.

pub mod cpml_restart;
pub mod dispersion_ade;
pub mod poynting_energy;
pub mod vector_calculus;
pub mod yee_fdtd;

pub use cpml_restart::*;
pub use dispersion_ade::*;
pub use poynting_energy::*;
pub use vector_calculus::*;
pub use yee_fdtd::*;

/// Exact SI speed of light in vacuum $c_0 = 299\,792\,458\text{ m/s}$.
pub const C0_M_PER_S: f64 = 299_792_458.0;

/// Exact SI Planck constant $h = 6.626\,070\,15 \times 10^{-34}\text{ J}\cdot\text{s}$.
pub const PLANCK_H_J_S: f64 = 6.626_070_15e-34;

/// Exact SI elementary charge $e = 1.602\,176\,634 \times 10^{-19}\text{ C}$.
pub const ELEMENTARY_CHARGE_C: f64 = 1.602_176_634e-19;

/// CODATA 2022 reference vacuum magnetic permeability $\mu_0 = 1.256\,637\,061\,27 \times 10^{-6}\text{ H/m}$.
pub const MU0_REF_H_PER_M: f64 = 1.256_637_061_27e-6;

/// Mutually consistent reference vacuum permittivity $\varepsilon_0 = \frac{1}{\mu_0 c_0^2}\text{ F/m}$.
pub const EPS0_REF_F_PER_M: f64 = 1.0 / (MU0_REF_H_PER_M * C0_M_PER_S * C0_M_PER_S);

/// Mutually consistent reference characteristic impedance of vacuum $Z_0 = \mu_0 c_0\text{ }\Omega$.
pub const Z0_REF_OHM: f64 = MU0_REF_H_PER_M * C0_M_PER_S;

/// Spatial boundary condition for 3D structured electromagnetic operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BoundaryCondition {
    Periodic = 0,
    DirichletPec = 1,
}

/// 3D structured Cartesian grid geometry (`[nx, ny, nz]` cells with spacings `[dx, dy, dz]` in metres).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid3dSpec {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

impl Grid3dSpec {
    #[inline]
    pub fn validate(&self) -> Result<usize, EmfOperatorError> {
        if self.nx == 0 || self.ny == 0 || self.nz == 0 {
            return Err(EmfOperatorError::InvalidGridDimensions {
                nx: self.nx,
                ny: self.ny,
                nz: self.nz,
            });
        }
        if !self.dx.is_finite()
            || !self.dy.is_finite()
            || !self.dz.is_finite()
            || self.dx <= 0.0
            || self.dy <= 0.0
            || self.dz <= 0.0
        {
            return Err(EmfOperatorError::InvalidGridSpacing {
                dx: self.dx,
                dy: self.dy,
                dz: self.dz,
            });
        }
        self.nx
            .checked_mul(self.ny)
            .and_then(|xy| xy.checked_mul(self.nz))
            .ok_or(EmfOperatorError::InvalidGridDimensions {
                nx: self.nx,
                ny: self.ny,
                nz: self.nz,
            })
    }

    #[inline]
    pub fn idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        (iz * self.ny + iy) * self.nx + ix
    }
}

/// Typed error enumeration for 3D electromagnetic operators, Yee FDTD, ADE, and C02 oracles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EmfOperatorError {
    InvalidGridDimensions {
        nx: usize,
        ny: usize,
        nz: usize,
    },
    InvalidGridSpacing {
        dx: f64,
        dy: f64,
        dz: f64,
    },
    SliceLengthMismatch {
        expected: usize,
        provided: usize,
    },
    NonFiniteInput,
    InvalidConstitutiveParameter {
        epsilon_r: f64,
        mu_r: f64,
        sigma: f64,
    },
    CflViolation {
        dt: f64,
        cfl_limit: f64,
    },
    DispersionEvanescent {
        rhs: f64,
    },
    IncompleteRestartState {
        reason: &'static str,
    },
    RestartDigestMismatch,
    InvalidPolarimetricState {
        i: f64,
        q: f64,
        u: f64,
        v: f64,
    },
    InvalidFrequencyOrWavelength {
        value: f64,
    },
    StaticCarrierNoWavelength,
    InvalidFrameTransform {
        det: f64,
        ortho_residual: f64,
    },
    SingularDipoleSourceEvaluation,
    DispersionPoleInvalid,
    GpuParityBudgetExceeded {
        max_abs_err: f64,
        budget: f64,
    },
}

impl core::fmt::Display for EmfOperatorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidGridDimensions { nx, ny, nz } => {
                write!(f, "invalid 3D grid dimensions: [{nx}, {ny}, {nz}]")
            }
            Self::InvalidGridSpacing { dx, dy, dz } => {
                write!(f, "invalid 3D grid spacing: [{dx}, {dy}, {dz}]")
            }
            Self::SliceLengthMismatch { expected, provided } => {
                write!(
                    f,
                    "slice length mismatch: expected {expected}, provided {provided}"
                )
            }
            Self::NonFiniteInput => write!(f, "non-finite or invalid scalar input"),
            Self::InvalidConstitutiveParameter {
                epsilon_r,
                mu_r,
                sigma,
            } => write!(
                f,
                "invalid constitutive parameter: eps_r={epsilon_r}, mu_r={mu_r}, sigma={sigma}"
            ),
            Self::CflViolation { dt, cfl_limit } => {
                write!(
                    f,
                    "3D Yee CFL stability violation: dt={dt} > cfl_limit={cfl_limit}"
                )
            }
            Self::DispersionEvanescent { rhs } => {
                write!(f, "discrete dispersion stopband/evanescent mode: rhs={rhs}")
            }
            Self::IncompleteRestartState { reason } => {
                write!(f, "incomplete Yee checkpoint state for restart: {reason}")
            }
            Self::RestartDigestMismatch => {
                write!(f, "Yee checkpoint BLAKE3 state commitment mismatch")
            }
            Self::InvalidPolarimetricState { i, q, u, v } => {
                write!(f, "invalid physical Stokes vector: [{i}, {q}, {u}, {v}]")
            }
            Self::InvalidFrequencyOrWavelength { value } => {
                write!(f, "invalid frequency or wavelength: {value}")
            }
            Self::StaticCarrierNoWavelength => {
                write!(f, "static/aperiodic field has no carrier wavelength")
            }
            Self::InvalidFrameTransform {
                det,
                ortho_residual,
            } => write!(
                f,
                "invalid frame rotation: det={det}, ortho_residual={ortho_residual}"
            ),
            Self::SingularDipoleSourceEvaluation => {
                write!(f, "singular dipole potential evaluation at source position")
            }
            Self::DispersionPoleInvalid => {
                write!(
                    f,
                    "invalid Debye, Drude, or Lorentz dispersion pole parameter"
                )
            }
            Self::GpuParityBudgetExceeded {
                max_abs_err,
                budget,
            } => write!(
                f,
                "GPU/CPU f32 parity budget exceeded: max_abs_err={max_abs_err} > budget={budget}"
            ),
        }
    }
}

impl std::error::Error for EmfOperatorError {}

#[cfg(test)]
mod tests;
