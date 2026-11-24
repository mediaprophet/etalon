//! Kramers-Kronig Acoustic Attenuation-Dispersion Consistency.
//!
//! Enforces causal phase-speed dispersion $c(\omega)$ paired with frequency-dependent acoustic
//! attenuation $\alpha(\omega)$:
//! 1. **Power-Law Attenuation** $\alpha(\omega) = \alpha_0 |\omega|^y$ ($0 < y < 3$):
//!    - Non-linear exponent ($y \ne 1$, Szabo 1994 / Waters et al. 2000):
//!      $$\frac{1}{c(\omega)} - \frac{1}{c(\omega_0)} = \alpha_0 \tan\!\left(\frac{\pi y}{2}\right)\left(|\omega|^{y-1} - |\omega_0|^{y-1}\right)$$
//!    - Linear-with-frequency exponent ($y = 1$, O'Donnell, Jaynes & Miller 1981):
//!      $$\frac{1}{c(\omega)} - \frac{1}{c(\omega_0)} = -\frac{2\alpha_0}{\pi}\ln\!\left(\frac{|\omega|}{|\omega_0|}\right)$$
//! 2. **Multi-Relaxation Media** (atmospheric $\text{O}_2/\text{N}_2$ and seawater boric acid
//!    $\text{B(OH)}_3$ + $\text{MgSO}_4$):
//!    - Exact Debye-type relaxation pair $(\alpha(\omega), c(\omega))$ satisfying the subtracted
//!      Kramers-Kronig dispersion relation:
//!      $$\frac{1}{c(\omega)} - \frac{1}{c(\omega_0)} = \frac{2(\omega^2 - \omega_0^2)}{\pi}\,\mathcal{P}\!\int_0^\infty \frac{\alpha(\Omega)/\Omega}{(\Omega^2 - \omega^2)(\Omega^2 - \omega_0^2)}\,d\Omega$$

use super::fdtd_fluid::NumericalError;
use super::helmholtz::Complex64;
use core::f64::consts::PI;

/// Power-law causal acoustic attenuation and Kramers-Kronig dispersion model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerLawAttenuation {
    /// Power-law prefactor $\alpha_0 \ge 0$ in $\text{Np}\cdot\text{m}^{-1}\cdot(\text{rad}\cdot\text{s}^{-1})^{-y}$.
    pub alpha_0: f64,
    /// Power-law exponent $y \in (0, 3)$ (with $y = 2$ non-dispersive thermoviscous limit).
    pub exponent_y: f64,
    /// Reference angular frequency $\omega_0 > 0$ in $\text{rad}\cdot\text{s}^{-1}$.
    pub omega_0_rad_s: f64,
    /// Reference phase speed $c(\omega_0) > 0$ in $\text{m}\cdot\text{s}^{-1}$.
    pub c_0_m_s: f64,
}

impl PowerLawAttenuation {
    /// Construct and validate a causal power-law attenuation-dispersion model.
    pub fn new(
        alpha_0: f64,
        exponent_y: f64,
        omega_0_rad_s: f64,
        c_0_m_s: f64,
    ) -> Result<Self, NumericalError> {
        if !alpha_0.is_finite() || alpha_0 < 0.0 {
            return Err(NumericalError::DomainError(
                "PowerLawAttenuation alpha_0 must be finite and >= 0",
            ));
        }
        if !exponent_y.is_finite() || exponent_y <= 0.0 || exponent_y >= 3.0 {
            return Err(NumericalError::DomainError(
                "PowerLawAttenuation exponent_y must satisfy 0 < y < 3 for causal Kramers-Kronig convergence",
            ));
        }
        if !omega_0_rad_s.is_finite() || omega_0_rad_s <= 0.0 {
            return Err(NumericalError::DomainError(
                "PowerLawAttenuation omega_0_rad_s must be finite and > 0",
            ));
        }
        if !c_0_m_s.is_finite() || c_0_m_s <= 0.0 {
            return Err(NumericalError::DomainError(
                "PowerLawAttenuation c_0_m_s must be finite and > 0",
            ));
        }
        Ok(Self {
            alpha_0,
            exponent_y,
            omega_0_rad_s,
            c_0_m_s,
        })
    }

    /// Attenuation coefficient $\alpha(\omega) = \alpha_0 |\omega|^y$ in $\text{Np}\cdot\text{m}^{-1}$.
    #[inline]
    pub fn attenuation_np_m(&self, omega_rad_s: f64) -> Result<f64, NumericalError> {
        if !omega_rad_s.is_finite() || omega_rad_s < 0.0 {
            return Err(NumericalError::DomainError(
                "omega_rad_s must be finite and >= 0",
            ));
        }
        if omega_rad_s == 0.0 {
            return Ok(0.0);
        }
        Ok(self.alpha_0 * omega_rad_s.powf(self.exponent_y))
    }

    /// Causal Kramers-Kronig slowness shift $\Delta s(\omega) = \frac{1}{c(\omega)} - \frac{1}{c(\omega_0)}$
    /// in $\text{s}\cdot\text{m}^{-1}$ at $\omega > 0$.
    pub fn slowness_shift_s_m(&self, omega_rad_s: f64) -> Result<f64, NumericalError> {
        if !omega_rad_s.is_finite() || omega_rad_s <= 0.0 {
            return Err(NumericalError::DomainError(
                "causal phase speed requires finite omega_rad_s > 0",
            ));
        }
        if self.alpha_0 == 0.0 || (self.exponent_y - 2.0).abs() <= 1e-14 {
            return Ok(0.0);
        }
        let dy = (self.exponent_y - 1.0).abs();
        if dy <= 1e-9 {
            // Logarithmic limit y -> 1 (Szabo / O'Donnell-Jaynes-Miller):
            Ok(-(2.0 * self.alpha_0 / PI) * (omega_rad_s / self.omega_0_rad_s).ln())
        } else {
            let tan_factor = (0.5 * PI * self.exponent_y).tan();
            let term = omega_rad_s.powf(self.exponent_y - 1.0)
                - self.omega_0_rad_s.powf(self.exponent_y - 1.0);
            Ok(self.alpha_0 * tan_factor * term)
        }
    }

    /// Causal phase speed $c(\omega)$ in $\text{m}\cdot\text{s}^{-1}$ at $\omega > 0$.
    pub fn phase_speed_m_s(&self, omega_rad_s: f64) -> Result<f64, NumericalError> {
        let inv_c = (1.0 / self.c_0_m_s) + self.slowness_shift_s_m(omega_rad_s)?;
        if inv_c <= 0.0 {
            return Err(NumericalError::DomainError(
                "power-law dispersion exceeded physical slowness breakdown bound (1/c <= 0)",
            ));
        }
        Ok(1.0 / inv_c)
    }

    /// Causal complex wavenumber $k(\omega) = \frac{\omega}{c(\omega)} + i\alpha(\omega)$ in $\text{m}^{-1}$.
    pub fn complex_wavenumber(&self, omega_rad_s: f64) -> Result<Complex64, NumericalError> {
        let c = self.phase_speed_m_s(omega_rad_s)?;
        let alpha = self.attenuation_np_m(omega_rad_s)?;
        Ok(Complex64::new(omega_rad_s / c, alpha))
    }
}

/// Single molecular/chemical acoustic relaxation mechanism.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelaxationProcess {
    /// Relaxation angular frequency $\omega_r = 2\pi f_r > 0$ in $\text{rad}\cdot\text{s}^{-1}$.
    pub omega_r_rad_s: f64,
    /// Dimensionless relaxation strength $\Delta_r = \frac{2}{\pi}(\alpha\lambda)_{\max,r} \ge 0$.
    pub strength_delta: f64,
}

/// Multi-relaxation + classical thermoviscous causal acoustic medium.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiRelaxationMedium {
    /// Equilibrium low-frequency phase speed $c_0 > 0$ in $\text{m}\cdot\text{s}^{-1}$.
    pub c_0_m_s: f64,
    /// Classical Stokes-Kirchhoff thermoviscous coefficient $\beta_{\text{visc}} \ge 0$ in $\text{Np}\cdot\text{m}^{-1}\cdot\text{s}^2\cdot\text{rad}^{-2}$.
    pub classical_viscous_coeff: f64,
    /// Discrete molecular relaxation mechanisms.
    pub processes: Vec<RelaxationProcess>,
}

impl MultiRelaxationMedium {
    /// Validate medium and relaxation process parameters.
    pub fn validate(&self) -> Result<(), NumericalError> {
        if !self.c_0_m_s.is_finite() || self.c_0_m_s <= 0.0 {
            return Err(NumericalError::DomainError(
                "MultiRelaxationMedium c_0_m_s must be finite and > 0",
            ));
        }
        if !self.classical_viscous_coeff.is_finite() || self.classical_viscous_coeff < 0.0 {
            return Err(NumericalError::DomainError(
                "MultiRelaxationMedium classical_viscous_coeff must be finite and >= 0",
            ));
        }
        for p in &self.processes {
            if !p.omega_r_rad_s.is_finite()
                || p.omega_r_rad_s <= 0.0
                || !p.strength_delta.is_finite()
                || p.strength_delta < 0.0
            {
                return Err(NumericalError::DomainError(
                    "RelaxationProcess requires omega_r_rad_s > 0 and strength_delta >= 0",
                ));
            }
        }
        Ok(())
    }

    /// Standard atmospheric air ($\text{O}_2$ and $\text{N}_2$ vibrational relaxation at $20^\circ\text{C}$, $1\text{ atm}$, $50\%$ RH).
    pub fn standard_atmosphere_20c_50rh() -> Self {
        Self {
            c_0_m_s: 343.2,
            classical_viscous_coeff: 1.84e-11 / (4.0 * PI * PI),
            processes: vec![
                // Nitrogen vibrational relaxation (~335 Hz)
                RelaxationProcess {
                    omega_r_rad_s: 2.0 * PI * 335.0,
                    strength_delta: 2.1e-4,
                },
                // Oxygen vibrational relaxation (~33.8 kHz)
                RelaxationProcess {
                    omega_r_rad_s: 2.0 * PI * 33_800.0,
                    strength_delta: 8.5e-4,
                },
            ],
        }
    }

    /// Standard ocean seawater ($\text{B(OH)}_3$ boric acid $\sim 1\text{ kHz}$ and $\text{MgSO}_4$ magnesium sulfate $\sim 75\text{ kHz}$
    /// at $10^\circ\text{C}, 35\text{ ppt}, \text{pH}=8.0$).
    pub fn standard_seawater_10c_35ppt() -> Self {
        Self {
            c_0_m_s: 1489.8,
            classical_viscous_coeff: 3.1e-16,
            processes: vec![
                // Boric acid B(OH)3 relaxation (~0.95 kHz)
                RelaxationProcess {
                    omega_r_rad_s: 2.0 * PI * 950.0,
                    strength_delta: 1.2e-5,
                },
                // Magnesium sulfate MgSO4 relaxation (~75 kHz)
                RelaxationProcess {
                    omega_r_rad_s: 2.0 * PI * 75_000.0,
                    strength_delta: 4.8e-5,
                },
            ],
        }
    }

    /// Total attenuation $\alpha(\omega)$ in $\text{Np}\cdot\text{m}^{-1}$:
    /// $$\alpha(\omega) = \frac{1}{2 c_0}\sum_r \Delta_r \frac{\omega^2 / \omega_r}{1 + (\omega/\omega_r)^2} + \beta_{\text{visc}}\omega^2$$
    pub fn attenuation_np_m(&self, omega_rad_s: f64) -> Result<f64, NumericalError> {
        self.validate()?;
        if !omega_rad_s.is_finite() || omega_rad_s < 0.0 {
            return Err(NumericalError::DomainError(
                "omega_rad_s must be finite and >= 0",
            ));
        }
        let mut sum_rel = 0.0f64;
        for p in &self.processes {
            let x = omega_rad_s / p.omega_r_rad_s;
            sum_rel += p.strength_delta * (omega_rad_s * x) / (1.0 + x * x);
        }
        Ok(sum_rel / (2.0 * self.c_0_m_s)
            + self.classical_viscous_coeff * omega_rad_s * omega_rad_s)
    }

    /// Analytical Kramers-Kronig slowness shift $\frac{1}{c(\omega)} - \frac{1}{c(\omega_0)}$ in $\text{s}\cdot\text{m}^{-1}$:
    /// $$\frac{1}{c(\omega)} - \frac{1}{c(\omega_0)} = -\frac{1}{2 c_0}\sum_r \Delta_r \left(\frac{(\omega/\omega_r)^2}{1 + (\omega/\omega_r)^2} - \frac{(\omega_0/\omega_r)^2}{1 + (\omega_0/\omega_r)^2}\right)$$
    pub fn slowness_shift_s_m(
        &self,
        omega_rad_s: f64,
        omega_0_rad_s: f64,
    ) -> Result<f64, NumericalError> {
        self.validate()?;
        if !omega_rad_s.is_finite()
            || omega_rad_s < 0.0
            || !omega_0_rad_s.is_finite()
            || omega_0_rad_s < 0.0
        {
            return Err(NumericalError::DomainError(
                "omega_rad_s and omega_0_rad_s must be finite and >= 0",
            ));
        }
        let mut sum_shift = 0.0f64;
        for p in &self.processes {
            let x = omega_rad_s / p.omega_r_rad_s;
            let x0 = omega_0_rad_s / p.omega_r_rad_s;
            let f_x = (x * x) / (1.0 + x * x);
            let f_x0 = (x0 * x0) / (1.0 + x0 * x0);
            sum_shift += p.strength_delta * (f_x - f_x0);
        }
        Ok(-sum_shift / (2.0 * self.c_0_m_s))
    }

    /// Causal phase speed $c(\omega)$ in $\text{m}\cdot\text{s}^{-1}$ (relative to $c_0 = c(0)$).
    pub fn phase_speed_m_s(&self, omega_rad_s: f64) -> Result<f64, NumericalError> {
        let shift = self.slowness_shift_s_m(omega_rad_s, 0.0)?;
        let inv_c = (1.0 / self.c_0_m_s) + shift;
        if inv_c <= 0.0 {
            return Err(NumericalError::DomainError(
                "multi-relaxation slowness became non-positive",
            ));
        }
        Ok(1.0 / inv_c)
    }

    /// Independent numerical Kramers-Kronig Hilbert quadrature of the relaxation attenuation
    /// using the singularity-subtracted Cauchy principal value identity:
    /// $$\frac{1}{c(\omega)} - \frac{1}{c(\omega_0)} = \frac{2(\omega^2 - \omega_0^2)}{\pi}\int_0^\infty \frac{g(\Omega) - g(\omega)\frac{\Omega^2 - \omega_0^2}{\omega^2 - \omega_0^2} - g(\omega_0)\frac{\omega^2 - \Omega^2}{\omega^2 - \omega_0^2}}{(\Omega^2 - \omega^2)(\Omega^2 - \omega_0^2)}\,d\Omega$$
    /// where $g(\Omega) = \frac{\alpha_{\text{rel}}(\Omega)}{\Omega}$ and the Cauchy principal value of
    /// $\int_0^\infty \frac{d\Omega}{\Omega^2 - \omega^2} = 0$ identically for $\omega > 0$.
    pub fn numerical_kk_slowness_shift_quadrature(
        &self,
        omega_rad_s: f64,
        omega_0_rad_s: f64,
        steps: usize,
    ) -> Result<f64, NumericalError> {
        self.validate()?;
        if omega_rad_s <= 0.0 || omega_0_rad_s <= 0.0 || steps < 64 {
            return Err(NumericalError::DomainError(
                "numerical_kk_slowness_shift_quadrature requires omega, omega_0 > 0 and steps >= 64",
            ));
        }
        if (omega_rad_s - omega_0_rad_s).abs() <= 1e-12 * omega_0_rad_s {
            return Ok(0.0);
        }
        let g = |w: f64| -> f64 {
            let mut s = 0.0f64;
            for p in &self.processes {
                let x = w / p.omega_r_rad_s;
                s += p.strength_delta * (w * x) / (1.0 + x * x);
            }
            s / (2.0 * self.c_0_m_s)
        };
        let w2 = omega_rad_s * omega_rad_s;
        let w02 = omega_0_rad_s * omega_0_rad_s;
        let diff_w2 = w2 - w02;
        let g_w = g(omega_rad_s);
        let g_w0 = g(omega_0_rad_s);

        // Map u in (0, pi/2) via Omega = scale * tan(u), dOmega = scale / cos^2(u) du.
        let scale = (omega_rad_s * omega_0_rad_s).sqrt();
        let du = (0.5 * PI) / (steps as f64);
        let mut integral = 0.0f64;
        for i in 0..steps {
            let u = (i as f64 + 0.5) * du;
            let cos_u = u.cos();
            let big_w = scale * u.tan();
            let big_w2 = big_w * big_w;
            let d1 = big_w2 - w2;
            let d0 = big_w2 - w02;
            if d1.abs() <= 1e-9 * w2 || d0.abs() <= 1e-9 * w02 {
                continue;
            }
            let g_big = g(big_w);
            // Subtracted regular numerator: vanishes at big_w = omega and big_w = omega_0.
            let num = g_big * diff_w2 - g_w * d0 - g_w0 * (w2 - big_w2);
            let integrand = num / (diff_w2 * d1 * d0);
            let jacobian = scale / (cos_u * cos_u);
            integral += integrand * jacobian * du;
        }
        Ok((2.0 * diff_w2 / PI) * integral)
    }
}
