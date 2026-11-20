//! ISO 532-1 (Zwicker / Moore) 24-Bark Specific & Total Loudness (`sones`, `phons`) and
//! AIUM / NEMA UD 3 Steady-State Pennes Bioheat Thermal Index (`TIS`, `TIB`, `TIC`) (`#![no_std]`).
//!
//! Closes R03 Gap #3 for psychoacoustic sone loudness synthesis and ultrasound thermal index
//! PDE/steady-state evaluation in `etalon-signal` without heap allocation or `std`.

use crate::bs1770::QualityStatus;
use crate::filter::{exp_f64, log10_f64, pow10_f64, pow_f64, sqrt_f64};
use crate::SignalError;

/// Number of standard Zwicker critical bands ($1..24\text{ Bark}$, $\Delta z = 1\text{ Bark}$).
pub const ZWICKER_BARK_BANDS: usize = 24;

/// Center frequencies (Hz) of the 24 Zwicker critical bands (ISO 532-1).
pub const ZWICKER_24_BARK_CENTER_HZ: [f64; ZWICKER_BARK_BANDS] = [
    50.0, 150.0, 250.0, 350.0, 450.0, 570.0, 700.0, 840.0, 1000.0, 1170.0, 1370.0, 1600.0,
    1850.0, 2150.0, 2500.0, 2900.0, 3400.0, 4000.0, 4800.0, 5800.0, 7000.0, 8500.0, 10500.0,
    13500.0,
];

/// Threshold in quiet $L_{\text{TQ},b}$ (dB SPL re $20\text{ }\mu\text{Pa}$) across the 24 Bark bands.
pub const ZWICKER_24_BARK_THRESHOLD_IN_QUIET_DB: [f64; ZWICKER_BARK_BANDS] = [
    38.0, 24.0, 16.0, 11.0, 8.0, 6.0, 5.0, 4.0, 3.0, 3.0, 3.0, 3.0,
    3.0, 2.0, 1.0, 1.0, 2.0, 4.0, 7.0, 12.0, 18.0, 26.0, 36.0, 48.0,
];

/// Canonical URI for AIUM/NEMA UD 3 soft-tissue thermal index (`TIS`).
pub const TIS_MODEL_URI: &str = "urn:qualia:qac:ti:aium_nema_ud3_tis_soft_tissue_v1";

/// Canonical URI for AIUM/NEMA UD 3 bone-at-focus thermal index (`TIB`).
pub const TIB_MODEL_URI: &str = "urn:qualia:qac:ti:aium_nema_ud3_tib_bone_at_focus_v1";

/// Canonical URI for AIUM/NEMA UD 3 cranial-bone thermal index (`TIC`).
pub const TIC_MODEL_URI: &str = "urn:qualia:qac:ti:aium_nema_ud3_tic_cranial_bone_v1";

/// Free-field 1 kHz 40 dB SPL = 1.000 sone normalization constant for the 24-Bark Zwicker integral.
const ZWICKER_FREE_FIELD_NORM: f64 = 1.478_511_482_006_941_8;

/// Summary of Zwicker 24-Bark specific loudness integration (`ISO 532-1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZwickerLoudnessReport {
    pub status: QualityStatus,
    pub total_loudness_sones: f64,
    pub loudness_level_phons: f64,
    pub peak_specific_loudness_sones_per_bark: f64,
}

/// Apply Zwicker critical-band excitation spreading across 24 Bark bands (`in_band_db` -> `out_excitation_linear`).
///
/// Lower-frequency skirt slope is $-27\text{ dB/Bark}$; upper-frequency skirt slope is level- and
/// frequency-dependent: $S_u(b) = -24.0 - \frac{230.0}{f_c(b)} + 0.2 L_b\text{ dB/Bark}$.
pub fn spread_zwicker_bark_excitation_into(
    band_levels_db_spl: &[f64; ZWICKER_BARK_BANDS],
    out_excitation_ratio: &mut [f64; ZWICKER_BARK_BANDS],
) -> Result<(), SignalError> {
    if band_levels_db_spl.iter().any(|x| !x.is_finite()) {
        return Err(SignalError::NonFiniteInput);
    }
    out_excitation_ratio.fill(0.0);
    for src in 0..ZWICKER_BARK_BANDS {
        let l_db = band_levels_db_spl[src];
        if l_db <= -120.0 {
            continue;
        }
        let e_src = pow10_f64(0.1 * l_db);
        let upper_slope_db = (-24.0 - 230.0 / ZWICKER_24_BARK_CENTER_HZ[src] + 0.2 * l_db.clamp(0.0, 110.0))
            .clamp(-45.0, -5.0);
        for (dst, out_slot) in out_excitation_ratio.iter_mut().enumerate().take(ZWICKER_BARK_BANDS) {
            let dz = (dst as f64) - (src as f64);
            let att_db = if dz < 0.0 {
                27.0 * dz
            } else if dz > 0.0 {
                upper_slope_db * dz
            } else {
                0.0
            };
            *out_slot += e_src * pow10_f64(0.1 * att_db);
        }
    }
    Ok(())
}

/// Convert total loudness $S$ (in sones) to loudness level $L_N$ (in phons) per ISO 532-1.
#[inline]
pub fn sones_to_phons_iso532(sones: f64) -> Result<f64, SignalError> {
    if !sones.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    if sones < 0.0 {
        return Err(SignalError::InvalidParameter);
    }
    if sones == 0.0 {
        return Ok(0.0);
    }
    if sones >= 1.0 {
        let log2_s = log10_f64(sones) / core::f64::consts::LOG10_2;
        Ok(40.0 + 10.0 * log2_s)
    } else {
        Ok((40.0 * pow_f64(sones + 0.0005, 0.35)).max(0.0))
    }
}

/// Convert loudness level $L_N$ (in phons, $\ge 40$) to total loudness $S$ (in sones): $S = 2^{(L_N - 40)/10}$.
#[inline]
pub fn phons_to_sones_iso532(phons: f64) -> Result<f64, SignalError> {
    if !phons.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    if phons < 0.0 {
        return Err(SignalError::InvalidParameter);
    }
    if phons == 0.0 {
        return Ok(0.0);
    }
    if phons >= 40.0 {
        Ok(exp_f64(core::f64::consts::LN_2 * (phons - 40.0) / 10.0))
    } else {
        Ok((pow_f64(phons / 40.0, 1.0 / 0.35) - 0.0005).max(0.0))
    }
}

/// Compute 24-Bark specific loudness $N'_b$ (`sones/Bark`) and integrate total loudness $S$ (`sones`)
/// and loudness level $L_N$ (`phons`) from 24 critical-band SPL levels (dB re $20\text{ }\mu\text{Pa}$).
///
/// Uses the Zwicker core power law:
/// $$N'_b = C_{\text{cal}} \cdot 0.08 \left(\frac{E_{\text{TQ},b}}{E_0}\right)^{0.23} \left[ \left(0.5 + 0.5 \frac{E_b}{E_{\text{TQ},b}}\right)^{0.23} - 1 \right]_+$$
pub fn compute_zwicker_specific_and_total_loudness_into(
    band_levels_db_spl: &[f64; ZWICKER_BARK_BANDS],
    out_specific_loudness_sones_bark: &mut [f64; ZWICKER_BARK_BANDS],
) -> Result<ZwickerLoudnessReport, SignalError> {
    let mut excitation_ratio = [0.0_f64; ZWICKER_BARK_BANDS];
    spread_zwicker_bark_excitation_into(band_levels_db_spl, &mut excitation_ratio)?;

    let mut total_sones = 0.0_f64;
    let mut peak_n_prime = 0.0_f64;
    for b in 0..ZWICKER_BARK_BANDS {
        let e_b = excitation_ratio[b];
        let e_tq = pow10_f64(0.1 * ZWICKER_24_BARK_THRESHOLD_IN_QUIET_DB[b]);
        let n_prime = if e_b > e_tq {
            let tq_term = pow_f64(e_tq, 0.23);
            let ratio_term = pow_f64(0.5 + 0.5 * (e_b / e_tq), 0.23) - 1.0;
            (ZWICKER_FREE_FIELD_NORM * 0.08 * tq_term * ratio_term).max(0.0)
        } else {
            0.0
        };
        out_specific_loudness_sones_bark[b] = n_prime;
        total_sones += n_prime;
        if n_prime > peak_n_prime {
            peak_n_prime = n_prime;
        }
    }

    let status = if total_sones <= 0.0 {
        QualityStatus::Silence
    } else {
        QualityStatus::Valid
    };
    let loudness_level_phons = sones_to_phons_iso532(total_sones)?;
    Ok(ZwickerLoudnessReport {
        status,
        total_loudness_sones: total_sones,
        loudness_level_phons,
        peak_specific_loudness_sones_per_bark: peak_n_prime,
    })
}

/// Discriminator for AIUM / NEMA UD 3 Thermal Index (`TI`) categories (`AQ-05` / `Schema 208`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ThermalIndexKind {
    SoftTissueTis = 0,
    BoneAtFocusTib = 1,
    CranialBoneTic = 2,
}

impl ThermalIndexKind {
    #[inline]
    pub const fn model_uri(self) -> &'static str {
        match self {
            Self::SoftTissueTis => TIS_MODEL_URI,
            Self::BoneAtFocusTib => TIB_MODEL_URI,
            Self::CranialBoneTic => TIC_MODEL_URI,
        }
    }
}

/// Context-qualified AIUM / NEMA UD 3 Thermal Index estimate (`TIS`, `TIB`, or `TIC`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThermalIndexEstimate {
    pub kind: ThermalIndexKind,
    pub model_uri: &'static str,
    pub status: QualityStatus,
    pub thermal_index: f64,
}

/// Compute AIUM / NEMA UD 3 unscanned soft-tissue thermal index:
/// $$\text{TIS} = \frac{W_0\text{ (mW)} \cdot f_{\text{MHz}}}{210.0\text{ mW}\cdot\text{MHz}}$$
pub fn compute_tis_soft_tissue(
    power_mw: f64,
    center_freq_mhz: f64,
) -> Result<ThermalIndexEstimate, SignalError> {
    if !power_mw.is_finite() || !center_freq_mhz.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    if power_mw < 0.0 || center_freq_mhz <= 0.0 {
        return Err(SignalError::InvalidParameter);
    }
    Ok(ThermalIndexEstimate {
        kind: ThermalIndexKind::SoftTissueTis,
        model_uri: TIS_MODEL_URI,
        status: QualityStatus::Valid,
        thermal_index: (power_mw * center_freq_mhz) / 210.0,
    })
}

/// Compute AIUM / NEMA UD 3 unscanned bone-at-focus thermal index:
/// $$\text{TIB} = \min\!\left(\frac{\sqrt{W_{.3}\text{ (mW)} \cdot I_{\text{SPTA.3}}\text{ (mW/cm}^2\text{)}}}{50.0}, \frac{W_{.3}\text{ (mW)}}{4.4}\right)$$
pub fn compute_tib_bone_at_focus(
    power_derated_mw: f64,
    i_spta_derated_mw_cm2: f64,
) -> Result<ThermalIndexEstimate, SignalError> {
    if !power_derated_mw.is_finite() || !i_spta_derated_mw_cm2.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    if power_derated_mw < 0.0 || i_spta_derated_mw_cm2 < 0.0 {
        return Err(SignalError::InvalidParameter);
    }
    let focal_branch = sqrt_f64(power_derated_mw * i_spta_derated_mw_cm2) / 50.0;
    let aperture_branch = power_derated_mw / 4.4;
    Ok(ThermalIndexEstimate {
        kind: ThermalIndexKind::BoneAtFocusTib,
        model_uri: TIB_MODEL_URI,
        status: QualityStatus::Valid,
        thermal_index: focal_branch.min(aperture_branch),
    })
}

/// Compute AIUM / NEMA UD 3 cranial-bone at-surface thermal index:
/// $$\text{TIC} = \frac{W_0\text{ (mW)}}{40.0 \cdot D_{\text{eq}}\text{ (cm)}}$$
pub fn compute_tic_cranial_bone(
    power_mw: f64,
    equivalent_aperture_diameter_cm: f64,
) -> Result<ThermalIndexEstimate, SignalError> {
    if !power_mw.is_finite() || !equivalent_aperture_diameter_cm.is_finite() {
        return Err(SignalError::NonFiniteInput);
    }
    if power_mw < 0.0 || equivalent_aperture_diameter_cm <= 0.0 {
        return Err(SignalError::InvalidParameter);
    }
    Ok(ThermalIndexEstimate {
        kind: ThermalIndexKind::CranialBoneTic,
        model_uri: TIC_MODEL_URI,
        status: QualityStatus::Valid,
        thermal_index: power_mw / (40.0 * equivalent_aperture_diameter_cm),
    })
}

/// Tissue and blood perfusion parameters for the 1D steady-state Pennes (1948) bioheat model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PennesBioheatParams {
    pub absorption_np_per_m_mhz: f64,
    pub frequency_mhz: f64,
    pub i_spta_w_per_m2: f64,
    pub metabolic_heat_w_per_m3: f64,
    pub blood_perfusion_rate_s1: f64,
    pub blood_density_kg_m3: f64,
    pub blood_specific_heat_j_kg_k: f64,
    pub tissue_thermal_conductivity_w_m_k: f64,
}

/// Steady-state Pennes bioheat perfusion solution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PennesBioheatResult {
    pub volumetric_heat_w_m3: f64,
    pub perfusion_sink_w_m3_k: f64,
    pub thermal_diffusion_length_m: f64,
    pub steady_state_delta_t_kelvin: f64,
}

/// Solve the steady-state Pennes bioheat perfusion balance:
/// $$q_v = 2 \alpha_{\text{abs}} f_{\text{MHz}} I_{\text{SPTA}} + q_m, \qquad \Delta T = \frac{q_v}{\omega_b \rho_b c_b}$$
pub fn solve_pennes_bioheat_steady_state(
    params: &PennesBioheatParams,
) -> Result<PennesBioheatResult, SignalError> {
    for v in [
        params.absorption_np_per_m_mhz,
        params.frequency_mhz,
        params.i_spta_w_per_m2,
        params.metabolic_heat_w_per_m3,
        params.blood_perfusion_rate_s1,
        params.blood_density_kg_m3,
        params.blood_specific_heat_j_kg_k,
        params.tissue_thermal_conductivity_w_m_k,
    ] {
        if !v.is_finite() {
            return Err(SignalError::NonFiniteInput);
        }
    }
    if params.absorption_np_per_m_mhz < 0.0
        || params.frequency_mhz <= 0.0
        || params.i_spta_w_per_m2 < 0.0
        || params.metabolic_heat_w_per_m3 < 0.0
        || params.blood_perfusion_rate_s1 <= 0.0
        || params.blood_density_kg_m3 <= 0.0
        || params.blood_specific_heat_j_kg_k <= 0.0
        || params.tissue_thermal_conductivity_w_m_k <= 0.0
    {
        return Err(SignalError::InvalidParameter);
    }
    let q_v = 2.0 * params.absorption_np_per_m_mhz * params.frequency_mhz * params.i_spta_w_per_m2
        + params.metabolic_heat_w_per_m3;
    let perfusion_sink =
        params.blood_perfusion_rate_s1 * params.blood_density_kg_m3 * params.blood_specific_heat_j_kg_k;
    let l_th = sqrt_f64(params.tissue_thermal_conductivity_w_m_k / perfusion_sink);
    let delta_t = q_v / perfusion_sink;
    Ok(PennesBioheatResult {
        volumetric_heat_w_m3: q_v,
        perfusion_sink_w_m3_k: perfusion_sink,
        thermal_diffusion_length_m: l_th,
        steady_state_delta_t_kelvin: delta_t,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zwicker_1khz_40db_one_sone_and_aium_nema_ud3_ti() {
        let mut bands = [-100.0_f64; ZWICKER_BARK_BANDS];
        bands[8] = 40.0;
        let mut spec = [0.0_f64; ZWICKER_BARK_BANDS];
        let rep40 = compute_zwicker_specific_and_total_loudness_into(&bands, &mut spec).unwrap();
        assert_eq!(rep40.status, QualityStatus::Valid);
        assert!((rep40.total_loudness_sones - 1.0).abs() < 0.02);
        assert!((rep40.loudness_level_phons - 40.0).abs() < 0.3);

        assert!((phons_to_sones_iso532(40.0).unwrap() - 1.0).abs() < 1e-9);
        assert!((phons_to_sones_iso532(50.0).unwrap() - 2.0).abs() < 1e-9);
        assert!((phons_to_sones_iso532(60.0).unwrap() - 4.0).abs() < 1e-9);

        let tis = compute_tis_soft_tissue(105.0, 2.0).unwrap();
        assert!((tis.thermal_index - 1.0).abs() < 1e-12);
        let tib = compute_tib_bone_at_focus(50.0, 50.0).unwrap();
        assert!((tib.thermal_index - 1.0).abs() < 1e-12);
        let tic = compute_tic_cranial_bone(80.0, 2.0).unwrap();
        assert!((tic.thermal_index - 1.0).abs() < 1e-12);

        let bio = solve_pennes_bioheat_steady_state(&PennesBioheatParams {
            absorption_np_per_m_mhz: 5.0,
            frequency_mhz: 3.0,
            i_spta_w_per_m2: 1000.0,
            metabolic_heat_w_per_m3: 400.0,
            blood_perfusion_rate_s1: 0.008,
            blood_density_kg_m3: 1060.0,
            blood_specific_heat_j_kg_k: 3600.0,
            tissue_thermal_conductivity_w_m_k: 0.5,
        })
        .unwrap();
        assert!(bio.steady_state_delta_t_kelvin > 0.9 && bio.steady_state_delta_t_kelvin < 1.1);
    }
}
