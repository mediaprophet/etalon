//! Debye, Drude, and Lorentz Auxiliary Differential Equation (ADE) Dispersive Media Operators (`etalon-sim`).
//!
//! Implements time-domain ADE polarization and conduction current updates, analytical
//! continuous and discrete frequency-domain complex susceptibilities under the `exp(-i * omega * t)`
//! harmonic convention, and positive-definite time-domain dispersive stored energy density.

use super::{EmfOperatorError, EPS0_REF_F_PER_M};

/// Single-pole Debye relaxation parameters:
/// $$\tau_p \frac{d\mathbf{P}_p}{dt} + \mathbf{P}_p = \varepsilon_0 \Delta\varepsilon_p \mathbf{E}$$
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DebyePole {
    pub delta_epsilon: f64,
    pub tau_s: f64,
}

impl DebyePole {
    pub fn validate(&self) -> Result<(), EmfOperatorError> {
        if !self.delta_epsilon.is_finite()
            || !self.tau_s.is_finite()
            || self.delta_epsilon <= 0.0
            || self.tau_s <= 0.0
        {
            return Err(EmfOperatorError::DispersionPoleInvalid);
        }
        Ok(())
    }

    pub fn continuous_susceptibility(
        &self,
        omega_rad_s: f64,
    ) -> Result<(f64, f64), EmfOperatorError> {
        self.validate()?;
        if !omega_rad_s.is_finite() || omega_rad_s < 0.0 {
            return Err(EmfOperatorError::NonFiniteInput);
        }
        let wt = omega_rad_s * self.tau_s;
        let denom = 1.0 + wt * wt;
        Ok((
            self.delta_epsilon / denom,
            (self.delta_epsilon * wt) / denom,
        ))
    }

    pub fn discrete_susceptibility(
        &self,
        omega_rad_s: f64,
        dt: f64,
    ) -> Result<(f64, f64), EmfOperatorError> {
        self.validate()?;
        if !omega_rad_s.is_finite() || omega_rad_s < 0.0 || !dt.is_finite() || dt <= 0.0 {
            return Err(EmfOperatorError::NonFiniteInput);
        }
        let omega_eff = (2.0 / dt) * (0.5 * omega_rad_s * dt).tan();
        let wt = omega_eff * self.tau_s;
        let denom = 1.0 + wt * wt;
        Ok((
            self.delta_epsilon / denom,
            (self.delta_epsilon * wt) / denom,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_ade_in_place(
        &self,
        dt: f64,
        ex_n: &[f64],
        ey_n: &[f64],
        ez_n: &[f64],
        ex_np1: &[f64],
        ey_np1: &[f64],
        ez_np1: &[f64],
        px: &mut [f64],
        py: &mut [f64],
        pz: &mut [f64],
        jpx: &mut [f64],
        jpy: &mut [f64],
        jpz: &mut [f64],
    ) -> Result<(), EmfOperatorError> {
        self.validate()?;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(EmfOperatorError::NonFiniteInput);
        }
        let n = px.len();
        if ex_n.len() != n
            || ey_n.len() != n
            || ez_n.len() != n
            || ex_np1.len() != n
            || ey_np1.len() != n
            || ez_np1.len() != n
            || py.len() != n
            || pz.len() != n
            || jpx.len() != n
            || jpy.len() != n
            || jpz.len() != n
        {
            return Err(EmfOperatorError::SliceLengthMismatch {
                expected: n,
                provided: ex_n.len(),
            });
        }

        let denom = 2.0 * self.tau_s + dt;
        let k_p = (2.0 * self.tau_s - dt) / denom;
        let beta_p = (EPS0_REF_F_PER_M * self.delta_epsilon * dt) / denom;
        let inv_dt = 1.0 / dt;

        for i in 0..n {
            let px_old = px[i];
            let py_old = py[i];
            let pz_old = pz[i];

            let px_new = k_p * px_old + beta_p * (ex_np1[i] + ex_n[i]);
            let py_new = k_p * py_old + beta_p * (ey_np1[i] + ey_n[i]);
            let pz_new = k_p * pz_old + beta_p * (ez_np1[i] + ez_n[i]);

            jpx[i] = (px_new - px_old) * inv_dt;
            jpy[i] = (py_new - py_old) * inv_dt;
            jpz[i] = (pz_new - pz_old) * inv_dt;

            px[i] = px_new;
            py[i] = py_new;
            pz[i] = pz_new;
        }
        Ok(())
    }

    #[inline]
    pub fn stored_energy_density(&self, p_vec: [f64; 3]) -> Result<f64, EmfOperatorError> {
        self.validate()?;
        let p_sq = p_vec[0] * p_vec[0] + p_vec[1] * p_vec[1] + p_vec[2] * p_vec[2];
        Ok(0.5 * p_sq / (EPS0_REF_F_PER_M * self.delta_epsilon))
    }
}

/// Drude conduction / cold-plasma pole parameters:
/// $$\frac{d\mathbf{J}_D}{dt} + \gamma_D \mathbf{J}_D = \varepsilon_0 \omega_p^2 \mathbf{E}$$
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrudePole {
    pub omega_p_rad_s: f64,
    pub gamma_rad_s: f64,
}

impl DrudePole {
    pub fn validate(&self) -> Result<(), EmfOperatorError> {
        if !self.omega_p_rad_s.is_finite()
            || !self.gamma_rad_s.is_finite()
            || self.omega_p_rad_s <= 0.0
            || self.gamma_rad_s < 0.0
        {
            return Err(EmfOperatorError::DispersionPoleInvalid);
        }
        Ok(())
    }

    pub fn continuous_susceptibility(
        &self,
        omega_rad_s: f64,
    ) -> Result<(f64, f64), EmfOperatorError> {
        self.validate()?;
        if !omega_rad_s.is_finite() || omega_rad_s <= 0.0 {
            return Err(EmfOperatorError::InvalidFrequencyOrWavelength { value: omega_rad_s });
        }
        let wp2 = self.omega_p_rad_s * self.omega_p_rad_s;
        let denom = omega_rad_s * (omega_rad_s * omega_rad_s + self.gamma_rad_s * self.gamma_rad_s);
        let re = -wp2 * omega_rad_s / denom;
        let im = wp2 * self.gamma_rad_s / denom;
        Ok((re, im))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_ade_in_place(
        &self,
        dt: f64,
        ex_n: &[f64],
        ey_n: &[f64],
        ez_n: &[f64],
        ex_np1: &[f64],
        ey_np1: &[f64],
        ez_np1: &[f64],
        jdx: &mut [f64],
        jdy: &mut [f64],
        jdz: &mut [f64],
    ) -> Result<(), EmfOperatorError> {
        self.validate()?;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(EmfOperatorError::NonFiniteInput);
        }
        let n = jdx.len();
        if ex_n.len() != n
            || ey_n.len() != n
            || ez_n.len() != n
            || ex_np1.len() != n
            || ey_np1.len() != n
            || ez_np1.len() != n
            || jdy.len() != n
            || jdz.len() != n
        {
            return Err(EmfOperatorError::SliceLengthMismatch {
                expected: n,
                provided: ex_n.len(),
            });
        }

        let half_g_dt = 0.5 * self.gamma_rad_s * dt;
        let k_d = (1.0 - half_g_dt) / (1.0 + half_g_dt);
        let beta_d =
            (EPS0_REF_F_PER_M * self.omega_p_rad_s * self.omega_p_rad_s * dt) / (1.0 + half_g_dt);

        for i in 0..n {
            jdx[i] = k_d * jdx[i] + 0.5 * beta_d * (ex_n[i] + ex_np1[i]);
            jdy[i] = k_d * jdy[i] + 0.5 * beta_d * (ey_n[i] + ey_np1[i]);
            jdz[i] = k_d * jdz[i] + 0.5 * beta_d * (ez_n[i] + ez_np1[i]);
        }
        Ok(())
    }

    #[inline]
    pub fn stored_energy_density(&self, jd_vec: [f64; 3]) -> Result<f64, EmfOperatorError> {
        self.validate()?;
        let j_sq = jd_vec[0] * jd_vec[0] + jd_vec[1] * jd_vec[1] + jd_vec[2] * jd_vec[2];
        Ok(0.5 * j_sq / (EPS0_REF_F_PER_M * self.omega_p_rad_s * self.omega_p_rad_s))
    }
}

/// Lorentz second-order resonant oscillator pole parameters:
/// $$\frac{d^2\mathbf{P}_L}{dt^2} + \gamma_L \frac{d\mathbf{P}_L}{dt} + \omega_0^2 \mathbf{P}_L = \varepsilon_0 \Delta\varepsilon_L \omega_0^2 \mathbf{E}$$
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LorentzPole {
    pub delta_epsilon: f64,
    pub omega_0_rad_s: f64,
    pub gamma_rad_s: f64,
}

impl LorentzPole {
    pub fn validate(&self) -> Result<(), EmfOperatorError> {
        if !self.delta_epsilon.is_finite()
            || !self.omega_0_rad_s.is_finite()
            || !self.gamma_rad_s.is_finite()
            || self.delta_epsilon <= 0.0
            || self.omega_0_rad_s <= 0.0
            || self.gamma_rad_s < 0.0
        {
            return Err(EmfOperatorError::DispersionPoleInvalid);
        }
        Ok(())
    }

    pub fn continuous_susceptibility(
        &self,
        omega_rad_s: f64,
    ) -> Result<(f64, f64), EmfOperatorError> {
        self.validate()?;
        if !omega_rad_s.is_finite() || omega_rad_s < 0.0 {
            return Err(EmfOperatorError::NonFiniteInput);
        }
        let w0_sq = self.omega_0_rad_s * self.omega_0_rad_s;
        let diff = w0_sq - omega_rad_s * omega_rad_s;
        let damp = self.gamma_rad_s * omega_rad_s;
        let denom = diff * diff + damp * damp;
        if denom == 0.0 {
            return Err(EmfOperatorError::DispersionPoleInvalid);
        }
        let num = self.delta_epsilon * w0_sq;
        Ok((num * diff / denom, num * damp / denom))
    }

    pub fn discrete_susceptibility(
        &self,
        omega_rad_s: f64,
        dt: f64,
    ) -> Result<(f64, f64), EmfOperatorError> {
        self.validate()?;
        if !omega_rad_s.is_finite() || omega_rad_s < 0.0 || !dt.is_finite() || dt <= 0.0 {
            return Err(EmfOperatorError::NonFiniteInput);
        }
        let w0_sq = self.omega_0_rad_s * self.omega_0_rad_s;
        let sin_half = (0.5 * omega_rad_s * dt).sin();
        let w_num_sq = (4.0 / (dt * dt)) * sin_half * sin_half;
        let diff = w0_sq - w_num_sq;
        let damp = self.gamma_rad_s * (omega_rad_s * dt).sin() / dt;
        let denom = diff * diff + damp * damp;
        if denom == 0.0 {
            return Err(EmfOperatorError::DispersionPoleInvalid);
        }
        let num = self.delta_epsilon * w0_sq;
        Ok((num * diff / denom, num * damp / denom))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_ade_in_place(
        &self,
        dt: f64,
        ex_n: &[f64],
        ey_n: &[f64],
        ez_n: &[f64],
        px_nm1: &mut [f64],
        py_nm1: &mut [f64],
        pz_nm1: &mut [f64],
        px_n: &mut [f64],
        py_n: &mut [f64],
        pz_n: &mut [f64],
        jlx_n: &mut [f64],
        jly_n: &mut [f64],
        jlz_n: &mut [f64],
    ) -> Result<(), EmfOperatorError> {
        self.validate()?;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(EmfOperatorError::NonFiniteInput);
        }
        let n = px_n.len();
        if ex_n.len() != n
            || ey_n.len() != n
            || ez_n.len() != n
            || px_nm1.len() != n
            || py_nm1.len() != n
            || pz_nm1.len() != n
            || py_n.len() != n
            || pz_n.len() != n
            || jlx_n.len() != n
            || jly_n.len() != n
            || jlz_n.len() != n
        {
            return Err(EmfOperatorError::SliceLengthMismatch {
                expected: n,
                provided: ex_n.len(),
            });
        }

        let w0_sq = self.omega_0_rad_s * self.omega_0_rad_s;
        let dt_sq = dt * dt;
        let denom = 2.0 + self.gamma_rad_s * dt;
        let alpha_l = (4.0 - 2.0 * w0_sq * dt_sq) / denom;
        let xi_l = (self.gamma_rad_s * dt - 2.0) / denom;
        let zeta_l = (2.0 * EPS0_REF_F_PER_M * self.delta_epsilon * w0_sq * dt_sq) / denom;
        let inv_2dt = 1.0 / (2.0 * dt);

        for i in 0..n {
            let px_np1 = alpha_l * px_n[i] + xi_l * px_nm1[i] + zeta_l * ex_n[i];
            let py_np1 = alpha_l * py_n[i] + xi_l * py_nm1[i] + zeta_l * ey_n[i];
            let pz_np1 = alpha_l * pz_n[i] + xi_l * pz_nm1[i] + zeta_l * ez_n[i];

            jlx_n[i] = (px_np1 - px_nm1[i]) * inv_2dt;
            jly_n[i] = (py_np1 - py_nm1[i]) * inv_2dt;
            jlz_n[i] = (pz_np1 - pz_nm1[i]) * inv_2dt;

            px_nm1[i] = px_n[i];
            py_nm1[i] = py_n[i];
            pz_nm1[i] = pz_n[i];

            px_n[i] = px_np1;
            py_n[i] = py_np1;
            pz_n[i] = pz_np1;
        }
        Ok(())
    }

    #[inline]
    pub fn stored_energy_density(
        &self,
        p_vec: [f64; 3],
        jl_vec: [f64; 3],
    ) -> Result<f64, EmfOperatorError> {
        self.validate()?;
        let p_sq = p_vec[0] * p_vec[0] + p_vec[1] * p_vec[1] + p_vec[2] * p_vec[2];
        let j_sq = jl_vec[0] * jl_vec[0] + jl_vec[1] * jl_vec[1] + jl_vec[2] * jl_vec[2];
        let w0_sq = self.omega_0_rad_s * self.omega_0_rad_s;
        Ok(0.5 * (p_sq + j_sq / w0_sq) / (EPS0_REF_F_PER_M * self.delta_epsilon))
    }
}
