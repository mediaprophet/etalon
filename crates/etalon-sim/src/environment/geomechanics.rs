//! Geotechnical Soil Mechanics & Continuum Earth Rheology (`geomechanics.rs`)
//!
//! Provides zero-heap continuum physics:
//! - Mohr-Coulomb shear strength with pore water pressure & matric suction
//! - Wu-Waldron vegetation root tensile reinforcement
//! - Critical unsupported trench wall depth (Rankine passive/active slip limit)
//! - Topsoil compaction and Kozeny-Carman hydraulic conductivity degradation
//! - Cryo-cementation freeze-thaw and seasonal moisture transitions
//!
//! Re-exports kinematic excavation tools from `excavation.rs`.
//! Adheres strictly to `AGENTS.md` §0 (Zero heap in hot paths, deterministic).

pub use super::excavation::*;

/// Canonical geotechnical soil failure and physical mechanics profile.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoilMechanicsProfile {
    /// Effective cohesion $c'$ in kPa (pure sand ~0, loam ~10, dense clay ~50+).
    pub cohesion_kpa: f32,
    /// Effective internal friction angle $\phi$ in radians (sand ~0.60 rad / 34°, clay ~0.35 rad / 20°).
    pub friction_angle_rad: f32,
    /// Soil unit weight / bulk density $\gamma$ in kN/m³ (typical 15.0 - 20.0 kN/m³).
    pub bulk_density_kn_m3: f32,
    /// Pore water pressure $u$ in kPa (negative for matric suction in damp soil, positive in saturated mud).
    pub pore_water_pressure_kpa: f32,
    /// Wu-Waldron root tensile cohesion reinforcement $\Delta c_{\text{root}}$ in kPa.
    pub root_tensile_reinforcement_kpa: f32,
    /// Soil-tool adhesion $c_a$ in kPa (sticky clay adhering to tool face).
    pub soil_tool_adhesion_kpa: f32,
    /// Volumetric bulking expansion factor $B_f = V_{\text{loose}} / V_{\text{bank}}$ (typically 1.15 - 1.35).
    pub bulking_factor: f32,
    /// Critical angle of repose for loose spoil heap in radians ($\theta_r \approx \phi$).
    pub angle_of_repose_rad: f32,
    /// Unconfined compressive strength $q_u$ in kPa (identifies lithic refusal threshold).
    pub unconfined_compressive_strength_kpa: f32,
}

impl Default for SoilMechanicsProfile {
    fn default() -> Self {
        Self::loam()
    }
}

impl SoilMechanicsProfile {
    /// Surface humic root-mat epipedon (Depth 1: 0–5 cm, dense subalpine grass roots).
    pub fn humic_root_mat() -> Self {
        Self {
            cohesion_kpa: 12.0,
            friction_angle_rad: 0.5236, // 30°
            bulk_density_kn_m3: 13.5,
            pore_water_pressure_kpa: -5.0, // slight matric suction
            root_tensile_reinforcement_kpa: 28.0, // strong root fibrous resistance
            soil_tool_adhesion_kpa: 4.0,
            bulking_factor: 1.30,
            angle_of_repose_rad: 0.6108, // 35°
            unconfined_compressive_strength_kpa: 120.0,
        }
    }

    /// Friable sandy loam topsoil (Depth 2: 5–15 cm).
    pub fn loam() -> Self {
        Self {
            cohesion_kpa: 10.0,
            friction_angle_rad: 0.5585, // 32°
            bulk_density_kn_m3: 16.5,
            pore_water_pressure_kpa: -2.0,
            root_tensile_reinforcement_kpa: 5.0,
            soil_tool_adhesion_kpa: 2.0,
            bulking_factor: 1.25,
            angle_of_repose_rad: 0.5585, // 32°
            unconfined_compressive_strength_kpa: 80.0,
        }
    }

    /// Dense plastic illuvial clay (Depth 4: 30–60 cm).
    pub fn illuvial_clay() -> Self {
        Self {
            cohesion_kpa: 55.0,
            friction_angle_rad: 0.3491, // 20°
            bulk_density_kn_m3: 19.5,
            pore_water_pressure_kpa: 0.0,
            root_tensile_reinforcement_kpa: 0.0,
            soil_tool_adhesion_kpa: 22.0, // very sticky
            bulking_factor: 1.35,
            angle_of_repose_rad: 0.4363, // 25°
            unconfined_compressive_strength_kpa: 250.0,
        }
    }

    /// Weathered gravelly saprolite / bedrock transition (Depth 5: 60–100 cm).
    pub fn saprolite() -> Self {
        Self {
            cohesion_kpa: 30.0,
            friction_angle_rad: 0.6981, // 40° (high friction from angular cobbles)
            bulk_density_kn_m3: 21.0,
            pore_water_pressure_kpa: 0.0,
            root_tensile_reinforcement_kpa: 0.0,
            soil_tool_adhesion_kpa: 0.5,
            bulking_factor: 1.18,
            angle_of_repose_rad: 0.6981, // 40°
            unconfined_compressive_strength_kpa: 850.0,
        }
    }

    /// Solid lithic bedrock (granite / basalt).
    pub fn bedrock() -> Self {
        Self {
            cohesion_kpa: 5_000.0,
            friction_angle_rad: 0.7854, // 45°
            bulk_density_kn_m3: 26.5,
            pore_water_pressure_kpa: 0.0,
            root_tensile_reinforcement_kpa: 0.0,
            soil_tool_adhesion_kpa: 0.0,
            bulking_factor: 1.50,
            angle_of_repose_rad: 0.7854,
            unconfined_compressive_strength_kpa: 65_000.0, // 65 MPa
        }
    }

    /// Adjusts soil mechanical resistance across diurnal freeze-thaw and annual seasons.
    ///
    /// - $T \le 0^\circ\text{C}$: Cryo-cementation (needle ice / permafrost crust) causes
    ///   cohesion and compressive strength to jump dramatically, leading to spade refusal.
    /// - $S_w > 0.80$: Spring snowmelt or saturated rain induces positive pore pressure ($u > 0$),
    ///   dropping effective stress while boosting clay-tool adhesion ($c_a$).
    /// - $S_w < 0.20$: Summer drought creates strong matric suction ($u \ll 0$), baking clay
    ///   into hard clods and generating airborne dust under mechanical impacts.
    pub fn with_seasonal_state(&self, temp_c: f32, moisture_fraction: f32) -> Self {
        let mut s = *self;
        let moisture = moisture_fraction.clamp(0.02, 1.0);

        if temp_c <= 0.0 {
            // Subalpine winter freezing / cryo-cementation
            let frost_severity = (temp_c.abs() * 0.4).min(4.0);
            s.cohesion_kpa += 180.0 * frost_severity;
            s.unconfined_compressive_strength_kpa += 1_200.0 * frost_severity;
            s.soil_tool_adhesion_kpa = 0.0; // frozen soil does not adhere
            s.bulking_factor = 1.12;
        } else if moisture > 0.80 {
            // Saturated spring snowmelt / winter rain
            let excess = (moisture - 0.80) / 0.20;
            s.pore_water_pressure_kpa = 12.0 * excess; // positive pore pressure
            s.cohesion_kpa = (s.cohesion_kpa * (1.0 - 0.60 * excess)).max(1.0);
            s.soil_tool_adhesion_kpa *= 1.7; // sticky clay
        } else if moisture < 0.20 {
            // Summer drought baking
            let drought = (0.20 - moisture) / 0.18;
            s.pore_water_pressure_kpa = -120.0 * drought;
            s.cohesion_kpa += 35.0 * drought;
            s.unconfined_compressive_strength_kpa += 300.0 * drought;
            s.soil_tool_adhesion_kpa *= 0.2;
        }
        s
    }
}

/// Indigenous and ecological seasonal categories for East Coast Australia.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AustralianSeason {
    /// Winter Snow & Frost (June–August): frozen soil crust, subnivean space.
    WinterSnowFrost = 0,
    /// Spring Snowmelt & Budburst (September–October): saturated mud, wattle bloom.
    SpringSnowmelt = 1,
    /// Early Summer Warmth (November–December): friable easy working, peak root growth.
    EarlySummerWarmth = 2,
    /// Late Summer Drought (January–February): baked clay, cracking soil, high fire risk.
    LateSummerDrought = 3,
    /// Early Autumn Smoke (March): post-fire ash crust, cooling nights.
    EarlyAutumnSmoke = 4,
    /// Late Autumn Pre-Winter (April–May): re-wetting, fungal flushes, geosmin release.
    LateAutumnPreWinter = 5,
}

/// Evaluates topsoil compaction under vehicular tracks or footsteps.
pub fn evaluate_soil_compaction(
    contact_pressure_kpa: f32,
    current_porosity: f32,
    soil: &SoilMechanicsProfile,
) -> (f32, f32) {
    if contact_pressure_kpa <= 0.0 {
        return (current_porosity, 1.0);
    }

    // Compression index C_c for topsoil (humic/clay loam C_c ~ 0.20-0.35)
    let c_c = (0.20 + (soil.cohesion_kpa / 100.0) * 0.15).clamp(0.12, 0.45);
    let initial_effective_stress = 15.0; // ~1m overburden
    let delta_e = c_c * ((initial_effective_stress + contact_pressure_kpa) / initial_effective_stress).log10();

    let initial_void_ratio = current_porosity / (1.0 - current_porosity).max(0.01);
    let final_void_ratio = (initial_void_ratio - delta_e).max(0.20);
    let final_porosity = final_void_ratio / (1.0 + final_void_ratio);

    // Kozeny-Carman hydraulic conductivity degradation: K ~ e^3 / (1 + e)
    let initial_kc = (initial_void_ratio * initial_void_ratio * initial_void_ratio) / (1.0 + initial_void_ratio);
    let final_kc = (final_void_ratio * final_void_ratio * final_void_ratio) / (1.0 + final_void_ratio);
    let k_multiplier = (final_kc / initial_kc.max(1e-4)).clamp(0.001, 1.0);

    (final_porosity, k_multiplier)
}

/// Critical maximum unsupported vertical trench wall depth before geotechnical collapse (Rankine slip).
pub fn critical_trench_depth_m(soil: &SoilMechanicsProfile) -> f32 {
    let phi = soil.friction_angle_rad;
    let rankine_ka = (1.0 - phi.sin()) / (1.0 + phi.sin()).max(1e-3);
    let sqrt_ka = rankine_ka.sqrt();

    // H_c = (4 * c) / (gamma * sqrt(K_a))
    let total_c = soil.cohesion_kpa + soil.root_tensile_reinforcement_kpa;
    (4.0 * total_c) / (soil.bulk_density_kn_m3 * sqrt_ka).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_compaction_reduces_permeability() {
        let loam = SoilMechanicsProfile::loam();
        let contact_pressure_kpa = 250.0; // Heavy machine track
        let initial_porosity = 0.45;
        let (new_n, k_ratio) = evaluate_soil_compaction(contact_pressure_kpa, initial_porosity, &loam);
        assert!(new_n < initial_porosity);
        assert!(k_ratio < 0.5); // Permeability significantly reduced
    }

    #[test]
    fn seasonal_frost_and_snowmelt_response() {
        let loam = SoilMechanicsProfile::loam();

        // Subalpine winter frost (-5°C): cryo-cementation causes cohesion surge
        let frozen = loam.with_seasonal_state(-5.0, 0.40);
        assert!(frozen.cohesion_kpa > loam.cohesion_kpa + 150.0);
        assert!(frozen.unconfined_compressive_strength_kpa > 1000.0);

        // Spring snowmelt (95% saturated): positive pore pressure, high clay adhesion
        let spring = loam.with_seasonal_state(12.0, 0.95);
        assert!(spring.pore_water_pressure_kpa > 0.0);
        assert!(spring.soil_tool_adhesion_kpa > loam.soil_tool_adhesion_kpa);
    }

    #[test]
    fn critical_trench_depth_root_reinforcement() {
        let humic = SoilMechanicsProfile::humic_root_mat();
        let loam = SoilMechanicsProfile::loam();

        let h_humic = critical_trench_depth_m(&humic);
        let h_loam = critical_trench_depth_m(&loam);

        // Root mat tensile reinforcement allows significantly deeper unsupported vertical face
        assert!(h_humic > h_loam);
        assert!(h_humic > 3.0);
    }
}
