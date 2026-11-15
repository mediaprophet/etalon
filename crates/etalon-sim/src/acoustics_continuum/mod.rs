//! 3D Continuum Acoustic & Elastic Solvers, Helmholtz, Room Acoustics & Kramers-Kronig Dispersion.
//!
//! Canonical `etalon-sim` continuum acoustics kernel suite (Packet **N02**):
//! - [`fdtd_fluid`]: 3D staggered-grid acoustic pressure-velocity leapfrog solver, 3D Courant
//!   stability gate, explicit spatiotemporal collocation for energy density $e$ and intensity $\mathbf{I}$,
//!   and Periodic, Rigid, Pressure-Release, Engquist-Majda, and multi-layer CPML boundaries with
//!   bit-exact restart bytes and 64-byte record helper (`PressureVelocityRecord`).
//! - [`elastic_virieux`]: 3D Virieux staggered-grid full 6-component symmetric Cauchy stress
//!   $(\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{yz},\sigma_{xz})$ and velocity solver,
//!   tensor strain $\varepsilon_{ij}$ vs engineering shear strain $\gamma_{ij}=2\varepsilon_{ij}$,
//!   active Hamilton quaternion rotation $\boldsymbol{\sigma}' = \mathbf{R}\boldsymbol{\sigma}\mathbf{R}^T$,
//!   free-surface ($\boldsymbol{\sigma}\cdot\mathbf{n}=\mathbf{0}$) and fluid-solid interfaces,
//!   normal-incidence impedance reflection/transmission, and 64-byte record helper (`ElasticStressVelocityRecord`).
//! - [`helmholtz`]: 3D frequency-domain Helmholtz operator $(\nabla^2 + k^2)\hat{p} = -\hat{s}$ with
//!   7-point and 27-point isotropic dispersion-minimizing stencils, zero-allocation damped Jacobi
//!   and BiCGSTAB solvers, free-space 3D Green's function $G(r)=\frac{e^{ikr}}{4\pi r}$, and
//!   rectangular cavity eigenmode oracles.
//! - [`room_acoustics`]: Sabine, Eyring-Norris, and Millington-Sette $T_{60}$ reverberation calculators
//!   and ISO 3382-1 Schroeder backward integration ($\text{EDT}, T_{20}, T_{30}, C_{50}, C_{80}, D_{50}, T_s$).
//! - [`attenuation_kk`]: Kramers-Kronig causal attenuation-dispersion consistency for power-law
//!   media ($0 < y < 3$, including $y=1$ logarithmic branch) and multi-relaxation atmospheric
//!   ($\text{O}_2/\text{N}_2$) and ocean seawater ($\text{B(OH)}_3/\text{MgSO}_4$) profiles.

pub mod attenuation_kk;
pub mod elastic_virieux;
pub mod fdtd_fluid;
pub mod helmholtz;
pub mod room_acoustics;

#[cfg(test)]
mod tests;

pub use attenuation_kk::{MultiRelaxationMedium, PowerLawAttenuation, RelaxationProcess};
pub use elastic_virieux::{
    step_elastic_virieux_in_place, CauchyStress6, ElasticBoundaryCondition, ElasticMedium,
    ElasticStressVelocityRecord, ElasticVirieuxConfig, ElasticVirieuxState,
    HamiltonQuaternionXyzw, NormalIncidenceInterface, TensorStrain6,
    ELASTIC_STRESS_VELOCITY_RECORD_STRIDE, ESV_DEFINED_VALIDITY_MASK, ESV_VALID_DISPLACEMENT,
    ESV_VALID_STRESS, ESV_VALID_VELOCITY, MODE_ELASTIC_STRESS_VELOCITY_CONTINUUM,
};
pub use fdtd_fluid::{
    collocate_fluid_cell, evaluate_direct_energy_and_intensity, step_fluid_fdtd_in_place,
    FluidBoundaryCondition, FluidFdtdConfig, FluidFdtdState, FluidMedium, NumericalError,
    PressureVelocityRecord, StaggeredFluidOffsets, MODE_PRESSURE_VELOCITY_CONTINUUM,
    PRESSURE_VELOCITY_RECORD_STRIDE, PV_DEFINED_VALIDITY_MASK, PV_VALID_ENERGY_DENSITY,
    PV_VALID_INTENSITY, PV_VALID_POSITION, PV_VALID_PRESSURE, PV_VALID_VELOCITY,
};
pub use helmholtz::{
    apply_helmholtz_operator, helmholtz_greens_function_3d,
    rectangular_cavity_eigenfrequency_hz, rectangular_cavity_eigenmode_value,
    solve_helmholtz_bicgstab, step_helmholtz_jacobi, Complex64, HelmholtzBoundary,
    HelmholtzConfig, HelmholtzSolveReport, HelmholtzStencil, HelmholtzWorkspace,
};
pub use room_acoustics::{
    analyze_room_impulse_response, fit_decay_rt60_s, schroeder_backward_integration_db,
    RirAcousticsReport, RoomGeometry, SurfacePatch, RT60_NUMERATOR_55_25,
    RT60_NUMERATOR_EXACT_24LN10,
};

/// Single-precision unit roundoff constant $u_{32} = 2^{-24}$ defined by the C02 `B32` arithmetic budget.
pub const B32_UNIT_ROUNDOFF: f64 = 1.0 / 16_777_216.0;

/// Compute the C02 `B32` scalar absolute error tolerance for a reference value `y_ref`
/// at characteristic physical scale `scale_s`:
/// $$\text{tol}(y_{\text{ref}}, S) = 8 u_{32} S + 2\times 10^{-6} |y_{\text{ref}}|, \qquad u_{32} = 2^{-24}$$
#[inline]
pub fn b32_scalar_tolerance(y_ref: f64, scale_s: f64) -> f64 {
    8.0 * B32_UNIT_ROUNDOFF * scale_s.abs() + 2e-6 * y_ref.abs()
}

/// Assert that `actual` agrees with `y_ref` within the C02 `B32` budget at characteristic scale `scale_s`.
#[inline]
pub fn assert_b32_scalar_budget(actual: f64, y_ref: f64, scale_s: f64) {
    let tol = b32_scalar_tolerance(y_ref, scale_s);
    let err = (actual - y_ref).abs();
    assert!(
        err <= tol,
        "B32 scalar budget exceeded: |{actual} - {y_ref}| = {err} > {tol} (scale S = {scale_s})"
    );
}
