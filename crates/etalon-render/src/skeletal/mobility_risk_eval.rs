//! BIO-WP-10: Coupled Mobility, Effort Reserve, Gait Stability & Bone Fracture Risk Evaluator.
//!
//! Couples neuromuscular capacity ($\text{RER}_j$ modulated by age, sex, PCSA, CT muscle
//! radiodensity $\text{HU}_{\text{muscle}}$, and MRI/CT myosteatosis $\text{IMAT}$), Hof's
//! Extrapolated Center of Mass ($\text{XCoM}$) and Margin of Stability ($\text{MoS}$), and
//! anisotropic bone Structural Factor of Risk ($\Phi_b$ from vBMD, BV/TV, fabric tensor
//! $\mathbf{M}_{\text{fabric}}$, mineralization $\mu_{\text{min}}$, and 4-stage fracture callus)
//! with zero heap allocations (`HotZeroHeap`).

use super::SkeletalError;

/// Biological substrate sex basis for age-trajectory and neuromuscular reference scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiologicalSexBasis {
    Male,
    Female,
    Unspecified,
}

/// Mechanobiological fracture healing stage (Claes & Heigele 1999).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FractureCallusStage {
    Intact,
    AcuteHematoma,
    SoftCallus,
    HardCallus,
    Remodeling,
}

impl FractureCallusStage {
    /// Returns `(modulus_factor, yield_strength_factor)` relative to intact bone.
    #[inline]
    pub const fn mechanical_factors(self) -> (f64, f64) {
        match self {
            Self::Intact => (1.00, 1.00),
            Self::AcuteHematoma => (0.02, 0.05),
            Self::SoftCallus => (0.15, 0.22),
            Self::HardCallus => (0.55, 0.65),
            Self::Remodeling => (0.88, 0.90),
        }
    }
}

/// Anatomic site selecting Morgan, Bayraktar & Keaveny (2003) / Carter & Hayes (1977) power laws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnatomicBoneSite {
    FemoralNeck,
    GreaterTrochanter,
    LumbarVertebra,
    ProximalTibia,
    CorticalDiaphysis,
}

/// Input parameters for a single joint actuator mobility/effort evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointMobilityInput {
    pub joint_id: u32,
    pub required_torque_nm: f64,
    pub age_years: f64,
    pub sex: BiologicalSexBasis,
    pub pcsa_cm2: f64,
    pub moment_arm_m: f64,
    /// Skeletal muscle CT attenuation in Hounsfield Units (healthy: +40..+60 HU, myosteatotic: < +30 HU).
    pub hu_muscle: f64,
    /// Intramuscular adipose tissue fraction $\text{IMAT} \in [0.0, 1.0)$.
    pub imat_fraction: f64,
    pub specific_tension_n_per_cm2: f64,
    pub neural_activation_reserve: f64,
}

/// Computed joint effort reserve and capacity evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointMobilityEvaluation {
    pub joint_id: u32,
    pub required_torque_nm: f64,
    pub max_torque_nm: f64,
    pub effective_muscle_quality_eta: f64,
    pub relative_effort_ratio: f64,
    pub capacity_exceeded: bool,
}

/// Input parameters for Hof's dynamic gait / posture Margin of Stability evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaitStabilityInput {
    pub com_pos_m: [f64; 2],
    pub com_vel_m_s: [f64; 2],
    pub leg_length_m: f64,
    pub gravity_m_s2: f64,
    pub bos_min_m: [f64; 2],
    pub bos_max_m: [f64; 2],
}

/// Computed Hof $\text{XCoM}$ and Margin of Stability evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaitStabilityEvaluation {
    pub xcom_m: [f64; 2],
    pub eigenfrequency_rad_s: f64,
    pub margin_of_stability_m: f64,
    pub is_unstable: bool,
}

/// Input parameters for structural bone fracture load evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoneFractureLoadInput {
    pub bone_id: u32,
    pub site: AnatomicBoneSite,
    pub applied_force_n: f64,
    pub load_bearing_area_mm2: f64,
    /// Volumetric bone mineral density in $\text{mg HA/cm}^3$.
    pub vbmd_mg_ha_cm3: f64,
    /// Trabecular bone volume fraction $\text{BV/TV} \in (0, 1]$.
    pub bv_tv: f64,
    /// Principal trabecular fabric tensor eigenvalues ($\text{tr}(\mathbf{M}_{\text{fabric}}) = 3$).
    pub fabric_eigenvalues: [f64; 3],
    /// Matrix mineralization ratio $\mu_{\text{min}} \in (0, 1.2]$ (< 1 in osteomalacia).
    pub mineralization_mu: f64,
    pub callus_stage: FractureCallusStage,
}

/// Computed bone yield capacity and Structural Factor of Risk $\Phi_b$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoneFractureLoadEvaluation {
    pub bone_id: u32,
    pub applied_force_n: f64,
    pub yield_force_n: f64,
    pub elastic_modulus_gpa: f64,
    pub yield_stress_mpa: f64,
    pub factor_of_risk_phi: f64,
    pub is_fracture_risk_critical: bool,
}

/// Coupled whole-body summary combining mobility reserve, dynamic balance, and fracture risk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoupledMobilityFractureSummary {
    pub joint_count: usize,
    pub bone_count: usize,
    pub max_rer: f64,
    pub limiting_joint_id: u32,
    pub stability: GaitStabilityEvaluation,
    pub max_factor_of_risk_phi: f64,
    pub critical_bone_id: u32,
    pub overall_fall_and_fracture_risk_high: bool,
}

/// Computes age- and sex-modulated neuromuscular force factor $\phi_{\text{neural}}(\text{age}, \text{sex})$.
#[inline]
fn age_sex_neuromuscular_factor(age_years: f64, sex: BiologicalSexBasis) -> f64 {
    let base_sex = match sex {
        BiologicalSexBasis::Male => 1.00,
        BiologicalSexBasis::Female => 0.93,
        BiologicalSexBasis::Unspecified => 0.96,
    };
    let age_factor = if age_years < 20.0 {
        (0.45 + 0.55 * (age_years / 20.0)).clamp(0.35, 1.0)
    } else if age_years <= 35.0 {
        1.0
    } else {
        let decline_rate = match sex {
            BiologicalSexBasis::Female => 0.0095,
            BiologicalSexBasis::Male => 0.0085,
            BiologicalSexBasis::Unspecified => 0.0090,
        };
        (1.0 - decline_rate * (age_years - 35.0)).clamp(0.35, 1.0)
    };
    base_sex * age_factor
}

/// Evaluates $\text{RER}_j = |\tau_{j,\text{req}}| / \tau_{j,\max}$ for a single joint.
pub fn evaluate_joint_mobility(
    input: &JointMobilityInput,
) -> Result<JointMobilityEvaluation, SkeletalError> {
    if !input.required_torque_nm.is_finite()
        || !input.age_years.is_finite()
        || input.age_years <= 0.0
        || !input.pcsa_cm2.is_finite()
        || input.pcsa_cm2 <= 0.0
        || !input.moment_arm_m.is_finite()
        || input.moment_arm_m <= 0.0
        || !input.hu_muscle.is_finite()
        || !input.imat_fraction.is_finite()
        || !(0.0..1.0).contains(&input.imat_fraction)
        || !input.specific_tension_n_per_cm2.is_finite()
        || input.specific_tension_n_per_cm2 <= 0.0
        || !input.neural_activation_reserve.is_finite()
        || input.neural_activation_reserve <= 0.0
    {
        return Err(SkeletalError::InvalidInput);
    }

    // Goodpaster CT attenuation mapping + MRI IMAT myosteatosis penalty + age/sex neural reserve.
    let hu_quality = ((input.hu_muscle + 20.0) / 70.0).clamp(0.15, 1.05);
    let imat_quality = (1.0 - input.imat_fraction).clamp(0.05, 1.0);
    let neural_factor = age_sex_neuromuscular_factor(input.age_years, input.sex);
    let eta_myo = (hu_quality
        * imat_quality
        * neural_factor
        * input.neural_activation_reserve.clamp(0.05, 1.0))
    .clamp(0.02, 1.05);

    let f_max_n = input.pcsa_cm2 * input.specific_tension_n_per_cm2 * eta_myo;
    let tau_max = (f_max_n * input.moment_arm_m).max(1e-6);
    let rer = input.required_torque_nm.abs() / tau_max;

    Ok(JointMobilityEvaluation {
        joint_id: input.joint_id,
        required_torque_nm: input.required_torque_nm,
        max_torque_nm: tau_max,
        effective_muscle_quality_eta: eta_myo,
        relative_effort_ratio: rer,
        capacity_exceeded: rer > 1.0,
    })
}

/// Evaluates Hof's Extrapolated Center of Mass ($\text{XCoM}$) and signed Margin of Stability ($\text{MoS}$).
pub fn evaluate_gait_stability(
    input: &GaitStabilityInput,
) -> Result<GaitStabilityEvaluation, SkeletalError> {
    if input.com_pos_m.iter().any(|v| !v.is_finite())
        || input.com_vel_m_s.iter().any(|v| !v.is_finite())
        || !input.leg_length_m.is_finite()
        || input.leg_length_m <= 0.0
        || !input.gravity_m_s2.is_finite()
        || input.gravity_m_s2 <= 0.0
        || input.bos_min_m.iter().any(|v| !v.is_finite())
        || input.bos_max_m.iter().any(|v| !v.is_finite())
        || input.bos_min_m[0] > input.bos_max_m[0]
        || input.bos_min_m[1] > input.bos_max_m[1]
    {
        return Err(SkeletalError::InvalidInput);
    }

    let omega_0 = (input.gravity_m_s2 / input.leg_length_m).sqrt();
    let xcom = [
        input.com_pos_m[0] + input.com_vel_m_s[0] / omega_0,
        input.com_pos_m[1] + input.com_vel_m_s[1] / omega_0,
    ];

    let dx_out = (input.bos_min_m[0] - xcom[0])
        .max(0.0)
        .max(xcom[0] - input.bos_max_m[0]);
    let dz_out = (input.bos_min_m[1] - xcom[1])
        .max(0.0)
        .max(xcom[1] - input.bos_max_m[1]);

    let mos = if dx_out > 0.0 || dz_out > 0.0 {
        -(dx_out * dx_out + dz_out * dz_out).sqrt()
    } else {
        let d_left = xcom[0] - input.bos_min_m[0];
        let d_right = input.bos_max_m[0] - xcom[0];
        let d_back = xcom[1] - input.bos_min_m[1];
        let d_front = input.bos_max_m[1] - xcom[1];
        d_left.min(d_right).min(d_back).min(d_front)
    };

    Ok(GaitStabilityEvaluation {
        xcom_m: xcom,
        eigenfrequency_rad_s: omega_0,
        margin_of_stability_m: mos,
        is_unstable: mos < 0.0,
    })
}

/// Evaluates Structural Factor of Risk $\Phi_b = F_{\text{applied}} / F_{\text{yield}}$
/// from volumetric BMD, BV/TV, fabric tensor $\mathbf{M}_{\text{fabric}}$, mineralization
/// $\mu_{\text{min}}$, and 4-stage fracture callus state.
pub fn evaluate_bone_fracture_load(
    input: &BoneFractureLoadInput,
) -> Result<BoneFractureLoadEvaluation, SkeletalError> {
    if !input.applied_force_n.is_finite()
        || input.applied_force_n < 0.0
        || !input.load_bearing_area_mm2.is_finite()
        || input.load_bearing_area_mm2 <= 0.0
        || !input.vbmd_mg_ha_cm3.is_finite()
        || input.vbmd_mg_ha_cm3 <= 0.0
        || !input.bv_tv.is_finite()
        || input.bv_tv <= 0.0
        || input.bv_tv > 1.0
        || input
            .fabric_eigenvalues
            .iter()
            .any(|&m| !m.is_finite() || m <= 0.0)
        || !input.mineralization_mu.is_finite()
        || input.mineralization_mu <= 0.0
    {
        return Err(SkeletalError::InvalidInput);
    }

    // Map QCT vBMD (mg HA/cm³) and BV/TV to apparent wet density rho_app (g/cm³).
    let rho_from_vbmd = (input.vbmd_mg_ha_cm3 / 620.0).clamp(0.03, 2.2);
    let rho_from_bvtv = (input.bv_tv * 1.85).clamp(0.03, 2.2);
    let rho_app = 0.6 * rho_from_vbmd + 0.4 * rho_from_bvtv;

    let (e0_gpa, gamma_e, sigma0_mpa, gamma_s) = match input.site {
        AnatomicBoneSite::FemoralNeck => (6.850, 1.49, 85.0, 1.85),
        AnatomicBoneSite::GreaterTrochanter => (15.010, 2.18, 65.0, 1.95),
        AnatomicBoneSite::LumbarVertebra => (4.730, 1.56, 37.0, 1.80),
        AnatomicBoneSite::ProximalTibia => (15.520, 1.93, 75.0, 1.90),
        AnatomicBoneSite::CorticalDiaphysis => (18.0, 2.57, 135.0, 2.10),
    };

    // Normalize fabric eigenvalue along primary load direction relative to isotropic tr(M)=3.
    let trace_m =
        (input.fabric_eigenvalues[0] + input.fabric_eigenvalues[1] + input.fabric_eigenvalues[2])
            .max(1e-6);
    let m1_norm = (3.0 * input.fabric_eigenvalues[0] / trace_m).clamp(0.3, 2.5);
    let fabric_e = m1_norm.powf(1.30);
    let fabric_s = m1_norm.powf(1.20);

    let (callus_e, callus_s) = input.callus_stage.mechanical_factors();
    let mu = input.mineralization_mu.clamp(0.05, 1.25);

    let modulus_gpa = match input.site {
        AnatomicBoneSite::CorticalDiaphysis => {
            e0_gpa * mu * (rho_app / 1.85).powf(gamma_e) * fabric_e * callus_e
        }
        _ => e0_gpa * mu * rho_app.powf(gamma_e) * fabric_e * callus_e,
    };

    let yield_stress_mpa = match input.site {
        AnatomicBoneSite::CorticalDiaphysis => {
            sigma0_mpa * mu * (rho_app / 1.85).powf(gamma_s) * fabric_s * callus_s
        }
        _ => sigma0_mpa * mu * rho_app.powf(gamma_s) * fabric_s * callus_s,
    };

    // 1 MPa = 1 N/mm², so F_yield (N) = sigma_yield (MPa) * area (mm²).
    let yield_force_n = (yield_stress_mpa * input.load_bearing_area_mm2).max(1e-6);
    let phi = input.applied_force_n / yield_force_n;

    Ok(BoneFractureLoadEvaluation {
        bone_id: input.bone_id,
        applied_force_n: input.applied_force_n,
        yield_force_n,
        elastic_modulus_gpa: modulus_gpa,
        yield_stress_mpa,
        factor_of_risk_phi: phi,
        is_fracture_risk_critical: phi >= 1.0,
    })
}

/// Evaluates coupled joint mobility effort ($\text{RER}_j$), Hof dynamic stability ($\text{MoS}$),
/// and bone Structural Factor of Risk ($\Phi_b$) into caller-supplied slices (`HotZeroHeap`).
pub fn evaluate_coupled_mobility_and_fracture_risk(
    joints: &[JointMobilityInput],
    stability_input: GaitStabilityInput,
    bones: &[BoneFractureLoadInput],
    out_joints: &mut [JointMobilityEvaluation],
    out_bones: &mut [BoneFractureLoadEvaluation],
) -> Result<CoupledMobilityFractureSummary, SkeletalError> {
    if out_joints.len() < joints.len() || out_bones.len() < bones.len() {
        return Err(SkeletalError::OutputTooSmall);
    }
    let stability = evaluate_gait_stability(&stability_input)?;

    // Validate and compute into stack/local check before writing if needed, or validate each.
    for j in joints {
        let _ = evaluate_joint_mobility(j)?;
    }
    for b in bones {
        let _ = evaluate_bone_fracture_load(b)?;
    }

    let mut max_rer = 0.0f64;
    let mut limiting_joint_id = 0u32;
    for (idx, j) in joints.iter().enumerate() {
        let eval = evaluate_joint_mobility(j)?;
        if eval.relative_effort_ratio >= max_rer {
            max_rer = eval.relative_effort_ratio;
            limiting_joint_id = eval.joint_id;
        }
        out_joints[idx] = eval;
    }

    let mut max_phi = 0.0f64;
    let mut critical_bone_id = 0u32;
    for (idx, b) in bones.iter().enumerate() {
        let eval = evaluate_bone_fracture_load(b)?;
        if eval.factor_of_risk_phi >= max_phi {
            max_phi = eval.factor_of_risk_phi;
            critical_bone_id = eval.bone_id;
        }
        out_bones[idx] = eval;
    }

    let high_risk = stability.is_unstable || max_rer > 0.85 || max_phi >= 1.0;

    Ok(CoupledMobilityFractureSummary {
        joint_count: joints.len(),
        bone_count: bones.len(),
        max_rer,
        limiting_joint_id,
        stability,
        max_factor_of_risk_phi: max_phi,
        critical_bone_id,
        overall_fall_and_fracture_risk_high: high_risk,
    })
}

/// Estimates sideways-fall trochanteric impact force (N) attenuated by trochanteric soft-tissue
/// padding $h_{\text{troch}}$ and protective stepping reflex attenuation (Robinovitch / Bouxsein).
pub fn sideways_fall_trochanteric_impact_force_n(
    body_mass_kg: f64,
    com_height_m: f64,
    trochanteric_soft_tissue_mm: f64,
    protective_reflex_attenuation: f64,
) -> Result<f64, SkeletalError> {
    if !body_mass_kg.is_finite()
        || body_mass_kg <= 0.0
        || !com_height_m.is_finite()
        || com_height_m <= 0.0
        || !trochanteric_soft_tissue_mm.is_finite()
        || trochanteric_soft_tissue_mm < 0.0
        || !protective_reflex_attenuation.is_finite()
        || !(0.0..1.0).contains(&protective_reflex_attenuation)
    {
        return Err(SkeletalError::InvalidInput);
    }
    // Nominal unattenuated peak impact force scales with sqrt(2 g h) * effective stiffness.
    let v_impact = (2.0 * 9.80665 * (0.5 * com_height_m)).sqrt();
    let k_eff = 71_000.0;
    let f_peak_raw = (0.5 * body_mass_kg * k_eff).sqrt() * v_impact;
    let pad_attenuation = (71.0 * trochanteric_soft_tissue_mm).min(0.65 * f_peak_raw);
    let f_attenuated = (f_peak_raw - pad_attenuation).max(200.0);
    Ok(f_attenuated * (1.0 - protective_reflex_attenuation))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_young_adult_vs_osteoporotic_sarcopenic_fall_and_fracture_risk() {
        // Healthy 28-year-old adult during brisk gait & step recovery.
        let healthy_joints = [
            JointMobilityInput {
                joint_id: 1,
                required_torque_nm: 42.0,
                age_years: 28.0,
                sex: BiologicalSexBasis::Female,
                pcsa_cm2: 42.0,
                moment_arm_m: 0.048,
                hu_muscle: 52.0,
                imat_fraction: 0.04,
                specific_tension_n_per_cm2: 35.0,
                neural_activation_reserve: 1.0,
            },
            JointMobilityInput {
                joint_id: 2,
                required_torque_nm: 35.0,
                age_years: 28.0,
                sex: BiologicalSexBasis::Female,
                pcsa_cm2: 38.0,
                moment_arm_m: 0.045,
                hu_muscle: 50.0,
                imat_fraction: 0.05,
                specific_tension_n_per_cm2: 35.0,
                neural_activation_reserve: 1.0,
            },
        ];
        let healthy_stability = GaitStabilityInput {
            com_pos_m: [0.0, 0.0],
            com_vel_m_s: [0.15, 0.05],
            leg_length_m: 0.92,
            gravity_m_s2: 9.80665,
            bos_min_m: [-0.12, -0.14],
            bos_max_m: [0.12, 0.14],
        };
        let impact_healthy =
            sideways_fall_trochanteric_impact_force_n(62.0, 0.92, 28.0, 0.25).unwrap();
        let healthy_bones = [
            BoneFractureLoadInput {
                bone_id: 10,
                site: AnatomicBoneSite::FemoralNeck,
                applied_force_n: impact_healthy,
                load_bearing_area_mm2: 360.0,
                vbmd_mg_ha_cm3: 340.0,
                bv_tv: 0.36,
                fabric_eigenvalues: [1.35, 0.90, 0.75],
                mineralization_mu: 1.0,
                callus_stage: FractureCallusStage::Intact,
            },
            BoneFractureLoadInput {
                bone_id: 11,
                site: AnatomicBoneSite::LumbarVertebra,
                applied_force_n: 1400.0,
                load_bearing_area_mm2: 1100.0,
                vbmd_mg_ha_cm3: 185.0,
                bv_tv: 0.24,
                fabric_eigenvalues: [1.25, 0.95, 0.80],
                mineralization_mu: 1.0,
                callus_stage: FractureCallusStage::Intact,
            },
        ];

        let mut out_j = [JointMobilityEvaluation {
            joint_id: 0,
            required_torque_nm: 0.0,
            max_torque_nm: 0.0,
            effective_muscle_quality_eta: 0.0,
            relative_effort_ratio: 0.0,
            capacity_exceeded: false,
        }; 2];
        let mut out_b = [BoneFractureLoadEvaluation {
            bone_id: 0,
            applied_force_n: 0.0,
            yield_force_n: 0.0,
            elastic_modulus_gpa: 0.0,
            yield_stress_mpa: 0.0,
            factor_of_risk_phi: 0.0,
            is_fracture_risk_critical: false,
        }; 2];

        let summary_healthy = evaluate_coupled_mobility_and_fracture_risk(
            &healthy_joints,
            healthy_stability,
            &healthy_bones,
            &mut out_j,
            &mut out_b,
        )
        .unwrap();

        assert!(!summary_healthy.stability.is_unstable);
        assert!(summary_healthy.stability.margin_of_stability_m > 0.0);
        assert!(summary_healthy.max_rer < 0.75);
        assert!(summary_healthy.max_factor_of_risk_phi < 0.65);
        assert!(!summary_healthy.overall_fall_and_fracture_risk_high);

        // Osteoporotic + sarcopenic 79-year-old adult with myosteatosis and reduced BoS margin.
        let osteo_joints = [
            JointMobilityInput {
                joint_id: 1,
                required_torque_nm: 42.0,
                age_years: 79.0,
                sex: BiologicalSexBasis::Female,
                pcsa_cm2: 28.0,
                moment_arm_m: 0.048,
                hu_muscle: 22.0,
                imat_fraction: 0.28,
                specific_tension_n_per_cm2: 35.0,
                neural_activation_reserve: 0.85,
            },
            JointMobilityInput {
                joint_id: 2,
                required_torque_nm: 35.0,
                age_years: 79.0,
                sex: BiologicalSexBasis::Female,
                pcsa_cm2: 26.0,
                moment_arm_m: 0.045,
                hu_muscle: 24.0,
                imat_fraction: 0.25,
                specific_tension_n_per_cm2: 35.0,
                neural_activation_reserve: 0.85,
            },
        ];
        let osteo_stability = GaitStabilityInput {
            com_pos_m: [0.06, 0.02],
            com_vel_m_s: [0.42, 0.12],
            leg_length_m: 0.88,
            gravity_m_s2: 9.80665,
            bos_min_m: [-0.11, -0.13],
            bos_max_m: [0.11, 0.13],
        };
        let impact_osteo =
            sideways_fall_trochanteric_impact_force_n(58.0, 0.86, 12.0, 0.05).unwrap();
        let osteo_bones = [
            BoneFractureLoadInput {
                bone_id: 10,
                site: AnatomicBoneSite::FemoralNeck,
                applied_force_n: impact_osteo,
                load_bearing_area_mm2: 320.0,
                vbmd_mg_ha_cm3: 110.0,
                bv_tv: 0.13,
                fabric_eigenvalues: [1.35, 0.90, 0.75],
                mineralization_mu: 0.92,
                callus_stage: FractureCallusStage::Intact,
            },
            BoneFractureLoadInput {
                bone_id: 11,
                site: AnatomicBoneSite::LumbarVertebra,
                applied_force_n: 1400.0,
                load_bearing_area_mm2: 1050.0,
                vbmd_mg_ha_cm3: 68.0,
                bv_tv: 0.10,
                fabric_eigenvalues: [1.25, 0.95, 0.80],
                mineralization_mu: 0.92,
                callus_stage: FractureCallusStage::Intact,
            },
        ];

        let mut out_j_osteo = out_j;
        let mut out_b_osteo = out_b;
        let summary_osteo = evaluate_coupled_mobility_and_fracture_risk(
            &osteo_joints,
            osteo_stability,
            &osteo_bones,
            &mut out_j_osteo,
            &mut out_b_osteo,
        )
        .unwrap();

        assert!(summary_osteo.max_rer > summary_healthy.max_rer);
        assert!(summary_osteo.max_rer > 1.0);
        assert!(summary_osteo.stability.is_unstable);
        assert!(summary_osteo.stability.margin_of_stability_m < 0.0);
        assert!(summary_osteo.max_factor_of_risk_phi > 1.0);
        assert!(summary_osteo.overall_fall_and_fracture_risk_high);
    }

    #[test]
    fn callus_healing_restores_yield_strength_across_stages() {
        let base = BoneFractureLoadInput {
            bone_id: 21,
            site: AnatomicBoneSite::CorticalDiaphysis,
            applied_force_n: 1200.0,
            load_bearing_area_mm2: 400.0,
            vbmd_mg_ha_cm3: 1050.0,
            bv_tv: 0.90,
            fabric_eigenvalues: [1.5, 0.8, 0.7],
            mineralization_mu: 1.0,
            callus_stage: FractureCallusStage::AcuteHematoma,
        };
        let mut soft = base;
        soft.callus_stage = FractureCallusStage::SoftCallus;
        let mut hard = base;
        hard.callus_stage = FractureCallusStage::HardCallus;
        let mut intact = base;
        intact.callus_stage = FractureCallusStage::Intact;

        let e_acute = evaluate_bone_fracture_load(&base).unwrap();
        let e_soft = evaluate_bone_fracture_load(&soft).unwrap();
        let e_hard = evaluate_bone_fracture_load(&hard).unwrap();
        let e_intact = evaluate_bone_fracture_load(&intact).unwrap();

        assert!(e_acute.yield_force_n < e_soft.yield_force_n);
        assert!(e_soft.yield_force_n < e_hard.yield_force_n);
        assert!(e_hard.yield_force_n < e_intact.yield_force_n);
        assert!(e_acute.factor_of_risk_phi > e_intact.factor_of_risk_phi);
    }

    #[test]
    fn invalid_inputs_fail_closed_without_mutating_outputs() {
        let bad_joint = [JointMobilityInput {
            joint_id: 1,
            required_torque_nm: f64::NAN,
            age_years: 30.0,
            sex: BiologicalSexBasis::Male,
            pcsa_cm2: 20.0,
            moment_arm_m: 0.04,
            hu_muscle: 50.0,
            imat_fraction: 0.05,
            specific_tension_n_per_cm2: 35.0,
            neural_activation_reserve: 1.0,
        }];
        let stab = GaitStabilityInput {
            com_pos_m: [0.0, 0.0],
            com_vel_m_s: [0.0, 0.0],
            leg_length_m: 0.9,
            gravity_m_s2: 9.80665,
            bos_min_m: [-0.1, -0.1],
            bos_max_m: [0.1, 0.1],
        };
        let sentinel_j = JointMobilityEvaluation {
            joint_id: 99,
            required_torque_nm: 42.0,
            max_torque_nm: 42.0,
            effective_muscle_quality_eta: 0.5,
            relative_effort_ratio: 1.0,
            capacity_exceeded: false,
        };
        let mut out_j = [sentinel_j];
        let mut out_b = [];
        let res = evaluate_coupled_mobility_and_fracture_risk(
            &bad_joint,
            stab,
            &[],
            &mut out_j,
            &mut out_b,
        );
        assert_eq!(res, Err(SkeletalError::InvalidInput));
        assert_eq!(out_j[0], sentinel_j);
    }
}
