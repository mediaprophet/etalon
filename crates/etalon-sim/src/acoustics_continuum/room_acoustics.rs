//! Room Acoustics: Sabine / Eyring-Norris / Millington-Sette $T_{60}$ & Schroeder RIR Integration.
//!
//! Implements:
//! - Sabine ($T_{60} = \frac{55.25 V}{c ( \sum S_i \alpha_i + 4mV )}$),
//!   Eyring-Norris ($T_{60} = \frac{55.25 V}{c ( -S\ln(1 - \bar{\alpha}) + 4mV )}$), and
//!   Millington-Sette ($T_{60} = \frac{55.25 V}{c ( -\sum S_i\ln(1 - \alpha_i) + 4mV )}$)
//!   reverberation time calculators with strict domain validation (rejecting $\alpha_i \ge 1$ in
//!   Eyring/Millington and zero total absorption).
//! - Schroeder backward energy integration $E(t) = \int_t^\infty h^2(\tau)\,d\tau$ in dB,
//!   Early Decay Time ($\text{EDT}$), $T_{20}$, $T_{30}$, Clarity $C_{50}$ / $C_{80}$,
//!   Definition $D_{50}$, and Centre Time $T_s$ (ISO 3382-1).

use super::fdtd_fluid::NumericalError;

/// Canonical numerator constant $55.25 = \frac{60 \times 4}{\ln(10)^{-1}} \approx 24\ln(10)$
/// used in the Sabine/Eyring/Millington formulas.
pub const RT60_NUMERATOR_55_25: f64 = 55.25;

/// Exact theoretical numerator $24\ln(10) \approx 55.2620422318571$.
pub const RT60_NUMERATOR_EXACT_24LN10: f64 = 55.262042231857095;

/// Surface patch with area $S_i$ ($\text{m}^2$) and random-incidence absorption coefficient $\alpha_i$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfacePatch {
    /// Surface area $S_i > 0$ in $\text{m}^2$.
    pub area_m2: f64,
    /// Energy absorption coefficient $\alpha_i$ ($0 \le \alpha_i \le 1$ for Sabine; $0 \le \alpha_i < 1$ for Eyring/Millington).
    pub absorption_coeff: f64,
}

/// Room geometry and acoustic boundary model for $T_{60}$ calculations.
#[derive(Debug, Clone, PartialEq)]
pub struct RoomGeometry {
    /// Enclosed air volume $V > 0$ in $\text{m}^3$.
    pub volume_m3: f64,
    /// Speed of sound $c > 0$ in $\text{m}\cdot\text{s}^{-1}$ (e.g. $343.0\text{ m/s}$ at $20^\circ\text{C}$).
    pub sound_speed_m_s: f64,
    /// Air intensity attenuation coefficient $m \ge 0$ in $\text{Np}\cdot\text{m}^{-1}$ (enters as $4mV$).
    pub air_attenuation_m_inv: f64,
    /// Boundary surface patches.
    pub surfaces: Vec<SurfacePatch>,
}

impl RoomGeometry {
    /// Validate basic geometric parameters ($V > 0, c > 0, m \ge 0$, non-empty surfaces with $S_i > 0$).
    pub fn validate_base(&self) -> Result<f64, NumericalError> {
        if !self.volume_m3.is_finite() || self.volume_m3 <= 0.0 {
            return Err(NumericalError::DomainError(
                "room volume_m3 must be finite and > 0",
            ));
        }
        if !self.sound_speed_m_s.is_finite() || self.sound_speed_m_s <= 0.0 {
            return Err(NumericalError::DomainError(
                "room sound_speed_m_s must be finite and > 0",
            ));
        }
        if !self.air_attenuation_m_inv.is_finite() || self.air_attenuation_m_inv < 0.0 {
            return Err(NumericalError::DomainError(
                "room air_attenuation_m_inv must be finite and >= 0",
            ));
        }
        if self.surfaces.is_empty() {
            return Err(NumericalError::DomainError(
                "room must contain at least one SurfacePatch",
            ));
        }
        let mut total_area = 0.0f64;
        for patch in &self.surfaces {
            if !patch.area_m2.is_finite() || patch.area_m2 <= 0.0 {
                return Err(NumericalError::DomainError(
                    "SurfacePatch area_m2 must be finite and > 0",
                ));
            }
            if !patch.absorption_coeff.is_finite()
                || patch.absorption_coeff < 0.0
                || patch.absorption_coeff > 1.0
            {
                return Err(NumericalError::DomainError(
                    "SurfacePatch absorption_coeff must be finite and in [0, 1]",
                ));
            }
            total_area += patch.area_m2;
        }
        Ok(total_area)
    }

    /// Sabine equivalent absorption area $A_{\text{Sabine}} = \sum_i S_i \alpha_i + 4 m V$ in $\text{m}^2$ (Sabins).
    pub fn sabine_absorption_area_m2(&self) -> Result<f64, NumericalError> {
        self.validate_base()?;
        let surf_abs: f64 = self
            .surfaces
            .iter()
            .map(|p| p.area_m2 * p.absorption_coeff)
            .sum();
        let total = surf_abs + 4.0 * self.air_attenuation_m_inv * self.volume_m3;
        if total <= 0.0 {
            return Err(NumericalError::DomainError(
                "total Sabine absorption area is zero (undamped room has no finite T60)",
            ));
        }
        Ok(total)
    }

    /// Sabine reverberation time $T_{60,\text{Sabine}} = \frac{55.25 V}{c (\sum_i S_i \alpha_i + 4mV)}$ in seconds.
    pub fn sabine_rt60_s(&self) -> Result<f64, NumericalError> {
        let a = self.sabine_absorption_area_m2()?;
        Ok((RT60_NUMERATOR_55_25 * self.volume_m3) / (self.sound_speed_m_s * a))
    }

    /// Eyring-Norris equivalent absorption area $A_{\text{Eyring}} = -S \ln(1 - \bar{\alpha}) + 4 m V$ in $\text{m}^2$.
    /// Rejects any patch with $\alpha_i \ge 1.0$ or mean $\bar{\alpha} \ge 1.0$.
    pub fn eyring_absorption_area_m2(&self) -> Result<f64, NumericalError> {
        let total_area = self.validate_base()?;
        for patch in &self.surfaces {
            if patch.absorption_coeff >= 1.0 {
                return Err(NumericalError::DomainError(
                    "Eyring-Norris formula rejects absorption_coeff >= 1.0 (logarithmic singularity)",
                ));
            }
        }
        let weighted_abs: f64 = self
            .surfaces
            .iter()
            .map(|p| p.area_m2 * p.absorption_coeff)
            .sum();
        let alpha_bar = weighted_abs / total_area;
        if alpha_bar >= 1.0 {
            return Err(NumericalError::DomainError(
                "Eyring-Norris mean absorption alpha_bar >= 1.0",
            ));
        }
        let a = -total_area * (1.0 - alpha_bar).ln()
            + 4.0 * self.air_attenuation_m_inv * self.volume_m3;
        if a <= 0.0 {
            return Err(NumericalError::DomainError(
                "total Eyring absorption area is zero (undamped room has no finite T60)",
            ));
        }
        Ok(a)
    }

    /// Eyring-Norris reverberation time $T_{60,\text{Eyring}} = \frac{55.25 V}{c (-S\ln(1 - \bar{\alpha}) + 4mV)}$ in seconds.
    pub fn eyring_rt60_s(&self) -> Result<f64, NumericalError> {
        let a = self.eyring_absorption_area_m2()?;
        Ok((RT60_NUMERATOR_55_25 * self.volume_m3) / (self.sound_speed_m_s * a))
    }

    /// Millington-Sette equivalent absorption area $A_{\text{Millington}} = -\sum_i S_i \ln(1 - \alpha_i) + 4 m V$ in $\text{m}^2$.
    /// Rejects any patch with $\alpha_i \ge 1.0$.
    pub fn millington_absorption_area_m2(&self) -> Result<f64, NumericalError> {
        self.validate_base()?;
        let mut sum_log = 0.0f64;
        for patch in &self.surfaces {
            if patch.absorption_coeff >= 1.0 {
                return Err(NumericalError::DomainError(
                    "Millington-Sette formula rejects absorption_coeff >= 1.0 (logarithmic singularity)",
                ));
            }
            sum_log += -patch.area_m2 * (1.0 - patch.absorption_coeff).ln();
        }
        let a = sum_log + 4.0 * self.air_attenuation_m_inv * self.volume_m3;
        if a <= 0.0 {
            return Err(NumericalError::DomainError(
                "total Millington-Sette absorption area is zero (undamped room has no finite T60)",
            ));
        }
        Ok(a)
    }

    /// Millington-Sette reverberation time $T_{60,\text{Millington}} = \frac{55.25 V}{c (-\sum_i S_i \ln(1 - \alpha_i) + 4mV)}$ in seconds.
    pub fn millington_rt60_s(&self) -> Result<f64, NumericalError> {
        let a = self.millington_absorption_area_m2()?;
        Ok((RT60_NUMERATOR_55_25 * self.volume_m3) / (self.sound_speed_m_s * a))
    }
}

/// Complete ISO 3382-1 room impulse response (RIR) decay and clarity analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct RirAcousticsReport {
    /// Normalized Schroeder Energy Decay Curve (EDC) in dB ($0\text{ dB}$ at $t=0$).
    pub schroeder_edc_db: Vec<f64>,
    /// Early Decay Time ($\text{EDT}$, s) fitted over $[0\text{ dB}, -10\text{ dB}]$.
    pub edt_s: f64,
    /// Reverberation time $T_{20}$ (s) fitted over $[-5\text{ dB}, -25\text{ dB}]$.
    pub t20_s: f64,
    /// Reverberation time $T_{30}$ (s) fitted over $[-5\text{ dB}, -35\text{ dB}]$.
    pub t30_s: f64,
    /// Speech clarity index $C_{50}$ in dB ($50\text{ ms}$ split).
    pub c50_db: f64,
    /// Music clarity index $C_{80}$ in dB ($80\text{ ms}$ split).
    pub c80_db: f64,
    /// Definition / Deutlichkeit $D_{50} \in [0, 1]$ ($50\text{ ms}$ early-to-total energy ratio).
    pub d50_ratio: f64,
    /// Centre time $T_s$ in seconds (first temporal moment of $h^2(t)$).
    pub centre_time_s: f64,
}

/// Compute the normalized Schroeder backward energy integration curve in dB:
/// $$E(k) = \sum_{n=k}^{N-1} h[n]^2, \qquad L_{\text{EDC}}(k) = 10\log_{10}\!\left(\frac{E(k)}{E(0)}\right)$$
/// Rejects empty, non-finite, or all-zero silent impulse responses (C02-C01 / AQ-04).
pub fn schroeder_backward_integration_db(impulse_response: &[f64]) -> Result<Vec<f64>, NumericalError> {
    if impulse_response.len() < 4 {
        return Err(NumericalError::DomainError(
            "impulse_response must contain at least 4 samples",
        ));
    }
    if impulse_response.iter().any(|v| !v.is_finite()) {
        return Err(NumericalError::DomainError(
            "impulse_response contains non-finite sample",
        ));
    }
    let n = impulse_response.len();
    let mut cumulative = vec![0.0f64; n];
    let mut acc = 0.0f64;
    for k in (0..n).rev() {
        let h = impulse_response[k];
        acc += h * h;
        cumulative[k] = acc;
    }
    let total_energy = cumulative[0];
    if total_energy <= 0.0 {
        return Err(NumericalError::DomainError(
            "silent impulse_response has zero total energy (logarithmic dB undefined)",
        ));
    }
    let mut edc_db = Vec::with_capacity(n);
    for &e in &cumulative {
        let ratio = (e / total_energy).max(1e-30);
        edc_db.push(10.0 * ratio.log10());
    }
    Ok(edc_db)
}

/// Fit linear least-squares decay slope on `edc_db` between `start_db` (e.g. `-5.0`) and
/// `end_db` (e.g. `-35.0`) and return the extrapolated $-60\text{ dB}$ decay time $T_{60}$ in seconds.
pub fn fit_decay_rt60_s(
    edc_db: &[f64],
    sample_rate_hz: f64,
    start_db: f64,
    end_db: f64,
) -> Result<f64, NumericalError> {
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Err(NumericalError::DomainError(
            "sample_rate_hz must be finite and > 0",
        ));
    }
    if !(start_db > end_db && start_db <= 0.0) {
        return Err(NumericalError::DomainError(
            "fit_decay_rt60_s requires 0 >= start_db > end_db",
        ));
    }
    let i_start = edc_db
        .iter()
        .position(|&db| db <= start_db)
        .ok_or(NumericalError::DomainError(
            "EDC never reaches start_db threshold",
        ))?;
    let i_end = edc_db[i_start..]
        .iter()
        .position(|&db| db <= end_db)
        .map(|off| i_start + off)
        .ok_or(NumericalError::DomainError(
            "EDC dynamic range insufficient to reach end_db threshold",
        ))?;
    if i_end <= i_start + 1 {
        return Err(NumericalError::DomainError(
            "insufficient samples in decay fitting window",
        ));
    }

    let dt = 1.0 / sample_rate_hz;
    let count = (i_end - i_start + 1) as f64;
    let mut sum_t = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_tt = 0.0f64;
    let mut sum_ty = 0.0f64;
    for (idx, &y) in edc_db.iter().enumerate().take(i_end + 1).skip(i_start) {
        let t = (idx as f64) * dt;
        sum_t += t;
        sum_y += y;
        sum_tt += t * t;
        sum_ty += t * y;
    }
    let denom = count * sum_tt - sum_t * sum_t;
    if denom.abs() <= 1e-30 {
        return Err(NumericalError::DomainError(
            "degenerate time variance in decay fit",
        ));
    }
    let slope_db_per_s = (count * sum_ty - sum_t * sum_y) / denom;
    if slope_db_per_s >= 0.0 {
        return Err(NumericalError::DomainError(
            "non-decaying Schroeder curve slope",
        ));
    }
    Ok(-60.0 / slope_db_per_s)
}

/// Perform full ISO 3382-1 analysis on a discrete room impulse response $h[n]$ sampled at `sample_rate_hz`.
pub fn analyze_room_impulse_response(
    impulse_response: &[f64],
    sample_rate_hz: f64,
) -> Result<RirAcousticsReport, NumericalError> {
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Err(NumericalError::DomainError(
            "sample_rate_hz must be finite and > 0",
        ));
    }
    let edc_db = schroeder_backward_integration_db(impulse_response)?;
    let edt_s = fit_decay_rt60_s(&edc_db, sample_rate_hz, 0.0, -10.0)?;
    let t20_s = fit_decay_rt60_s(&edc_db, sample_rate_hz, -5.0, -25.0)?;
    let t30_s = fit_decay_rt60_s(&edc_db, sample_rate_hz, -5.0, -35.0)?;

    let dt = 1.0 / sample_rate_hz;
    let split_50ms = ((0.050 * sample_rate_hz).round() as usize).min(impulse_response.len());
    let split_80ms = ((0.080 * sample_rate_hz).round() as usize).min(impulse_response.len());

    let mut e_early_50 = 0.0f64;
    let mut e_late_50 = 0.0f64;
    let mut e_early_80 = 0.0f64;
    let mut e_late_80 = 0.0f64;
    let mut moment_num = 0.0f64;

    for (n, &h) in impulse_response.iter().enumerate() {
        let e = h * h;
        let t = (n as f64) * dt;
        moment_num += t * e;
        if n < split_50ms {
            e_early_50 += e;
        } else {
            e_late_50 += e;
        }
        if n < split_80ms {
            e_early_80 += e;
        } else {
            e_late_80 += e;
        }
    }

    let e_total = e_early_50 + e_late_50;
    if e_late_50 <= 0.0 || e_late_80 <= 0.0 {
        return Err(NumericalError::DomainError(
            "late RIR energy after 50ms/80ms is zero; C50/C80 undefined",
        ));
    }
    let c50_db = 10.0 * (e_early_50 / e_late_50).log10();
    let c80_db = 10.0 * (e_early_80 / e_late_80).log10();
    let d50_ratio = e_early_50 / e_total;
    let centre_time_s = moment_num / e_total;

    Ok(RirAcousticsReport {
        schroeder_edc_db: edc_db,
        edt_s,
        t20_s,
        t30_s,
        c50_db,
        c80_db,
        d50_ratio,
        centre_time_s,
    })
}
