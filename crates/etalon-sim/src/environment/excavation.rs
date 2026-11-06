//! Geotechnical Excavation Kinematics & Cutting Resistance (`excavation.rs`)
//!
//! Provides zero-heap continuum physics for earthmoving implements:
//! - Human hand spade & foot-step plunge blade mechanics
//! - Point chisel pickaxe dynamic impact fracture
//! - Heavy hydraulic excavator bucket & ripper teeth crowd kinematics
//! - Animal fossorial scratching claws (*Vombatus ursinus*)
//! - McKyes-Ali 3D passive earth pressure cutting wedge analysis
//! - Soil volumetric bulking expansion and lithic refusal detection
//!
//! Adheres strictly to `AGENTS.md` §0 (Zero heap in hot paths, deterministic).

use super::geomechanics::SoilMechanicsProfile;

/// Tool physical class distinguishing human implements, machinery, and abstract clicks.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    /// Mouse pointer or camera raycast: dimensionless observational probe (0 N force).
    ObservationRay = 0,
    /// Manual hand spade or garden shovel (wedge cutting blade, foot drive).
    HandSpade = 1,
    /// Manual pickaxe or mattock (high-impulse point chisel for brittle fracture).
    Pickaxe = 2,
    /// Heavy hydraulic excavator bucket with Ripper teeth.
    ExcavatorBucket = 3,
    /// Animal fossorial scratching claws (Wombat *Vombatus ursinus*).
    FossorialClaw = 4,
}

/// Kinematic tool effector parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToolEffector {
    /// Category of tool or actuator.
    pub kind: ToolKind,
    /// Transverse blade or contact edge width in meters.
    pub blade_width_m: f32,
    /// Forward rake angle relative to soil surface in radians (typically 0.5 - 1.2 rad).
    pub rake_angle_rad: f32,
    /// Sharpness wedge angle at the cutting tip in radians.
    pub wedge_angle_rad: f32,
    /// Maximum normal/breakout force available from operator or hydraulic ram in Newtons.
    pub applied_force_n: f32,
    /// Operator applied prying torque in Newton-meters.
    pub applied_torque_nm: f32,
    /// Maximum scoop or bucket volumetric capacity in cubic meters.
    pub capacity_volume_m3: f32,
}

impl ToolEffector {
    /// Pure mouse click / pointer gaze (informational, non-destructive).
    pub fn mouse_click() -> Self {
        Self {
            kind: ToolKind::ObservationRay,
            blade_width_m: 0.0,
            rake_angle_rad: 1.5708, // 90°
            wedge_angle_rad: 0.0,
            applied_force_n: 0.0, // Zero force
            applied_torque_nm: 0.0,
            capacity_volume_m3: 0.0,
        }
    }

    /// Standard steel trenching spade operated by an adult human with foot-step drive.
    pub fn hand_spade() -> Self {
        Self {
            kind: ToolKind::HandSpade,
            blade_width_m: 0.20,     // 20 cm wide blade
            rake_angle_rad: 1.0472,  // 60° plunge angle
            wedge_angle_rad: 0.3491, // 20° beveled edge
            applied_force_n: 650.0,  // ~66 kg operator driving with boot heel
            applied_torque_nm: 95.0, // Prying leverage on handle
            capacity_volume_m3: 0.005, // 5 liters scoop volume
        }
    }

    /// Pointed pickaxe swing.
    pub fn pickaxe() -> Self {
        Self {
            kind: ToolKind::Pickaxe,
            blade_width_m: 0.025,    // 2.5 cm point tip
            rake_angle_rad: 1.2217,  // 70° impact angle
            wedge_angle_rad: 0.4363, // 25° forged tip
            applied_force_n: 2_400.0, // Dynamic impact force ($F = \Delta p / \Delta t$)
            applied_torque_nm: 40.0,
            capacity_volume_m3: 0.001,
        }
    }

    /// Mini-excavator 5-tonne backhoe bucket.
    pub fn mini_excavator() -> Self {
        Self {
            kind: ToolKind::ExcavatorBucket,
            blade_width_m: 0.60,      // 60 cm bucket with 4 teeth
            rake_angle_rad: 0.7854,   // 45° crowd angle
            wedge_angle_rad: 0.5236,  // 30° tooth wedge
            applied_force_n: 38_000.0, // 38 kN hydraulic breakout force
            applied_torque_nm: 4_500.0,
            capacity_volume_m3: 0.18,  // 180 liters
        }
    }

    /// Wombat forelimb fossorial claws (*Vombatus ursinus*).
    pub fn wombat_claws() -> Self {
        Self {
            kind: ToolKind::FossorialClaw,
            blade_width_m: 0.06,     // 6 cm total 5-claw spread
            rake_angle_rad: 0.8727,  // 50° scratching stroke
            wedge_angle_rad: 0.3840, // 22° sharp keratinous curves
            applied_force_n: 140.0,  // Forelimb muscular force
            applied_torque_nm: 12.0,
            capacity_volume_m3: 0.0015, // 1.5 liters per scratch cycle
        }
    }
}

/// The physical outcome of a tool contact attempt on a soil stratum.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExcavationVerdict {
    /// True if the applied tool force exceeded geotechnical cutting resistance.
    pub severed: bool,
    /// Total calculated cutting force required to displace the target volume (N).
    pub required_force_n: f32,
    /// Net margin force (applied - required). Positive means easy cut; negative means stall.
    pub margin_force_n: f32,
    /// Solid in-situ bank volume removed from terrain (m³).
    pub excavated_bank_volume_m3: f32,
    /// Expanded loose volume ready for tool bucket or spoil heap (m³).
    pub excavated_loose_volume_m3: f32,
    /// True if root mat tensile reinforcement was severed during the pass.
    pub root_mat_severed: bool,
    /// True if stratum was too rigid for the tool (e.g. spade hitting basalt rock).
    pub refusal_hit: bool,
}

/// Evaluates Mckyes-Ali earthmoving cutting resistance and geotechnical failure.
///
/// Returns an `ExcavationVerdict` detailing whether the tool slices, stalls, or refuses.
pub fn evaluate_excavation_pass(
    tool: &ToolEffector,
    target_depth_m: f32,
    soil: &SoilMechanicsProfile,
) -> ExcavationVerdict {
    // 1. Mouse click / observation gaze: strictly 0 force, cannot deform terrain
    if tool.kind == ToolKind::ObservationRay || tool.applied_force_n <= 0.0 {
        return ExcavationVerdict {
            severed: false,
            required_force_n: 0.0,
            margin_force_n: 0.0,
            excavated_bank_volume_m3: 0.0,
            excavated_loose_volume_m3: 0.0,
            root_mat_severed: false,
            refusal_hit: false,
        };
    }

    let d = target_depth_m.max(0.01);
    let w = tool.blade_width_m.max(0.01);

    // 2. Lithic refusal check: if soil compressive strength exceeds tool capability
    let max_penetrable_ucs = match tool.kind {
        ToolKind::ObservationRay => 0.0,
        ToolKind::HandSpade => 400.0,       // Hand spade halts on hardpan / soft rock
        ToolKind::FossorialClaw => 300.0,   // Claws cannot dig solid shale/rock
        ToolKind::Pickaxe => 35_000.0,      // Pickaxe fractures soft to medium rock
        ToolKind::ExcavatorBucket => 12_000.0, // Excavator teeth rip gravel/caliche, not solid granite
    };

    if soil.unconfined_compressive_strength_kpa > max_penetrable_ucs {
        return ExcavationVerdict {
            severed: false,
            required_force_n: soil.unconfined_compressive_strength_kpa * w * d * 100.0,
            margin_force_n: tool.applied_force_n - 100_000.0,
            excavated_bank_volume_m3: 0.0,
            excavated_loose_volume_m3: 0.0,
            root_mat_severed: false,
            refusal_hit: true,
        };
    }

    // 3. Mohr-Coulomb effective cohesion including Wu-Waldron root reinforcement
    let effective_cohesion = soil.cohesion_kpa + soil.root_tensile_reinforcement_kpa;

    // 4. Mckyes-Ali N-coefficients & wedge cutting factors
    let phi = soil.friction_angle_rad.max(0.1);
    let n_gamma = (tool.rake_angle_rad * 1.8).sin() * (1.0 + phi.tan() * 2.2);
    let n_c = (tool.rake_angle_rad).cos().abs() + 2.0 * (1.0 + phi.sin()) / (1.0 - phi.sin());
    let n_ca = 1.0 + (tool.rake_angle_rad).cos();

    // Sharp beveled cutting edges (spades, ripper teeth) cleave rather than displace a massive passive wedge
    let wedge_reduction = (tool.wedge_angle_rad * 0.5).sin().clamp(0.18, 1.0);

    // 5. Total cutting force P = [gamma * d^2 * N_gamma * w_red + c * d * N_c * w_red + c_a * d * N_ca] * w
    let unit_weight = soil.bulk_density_kn_m3;
    let pressure_kpa = unit_weight * d * d * n_gamma * wedge_reduction
        + effective_cohesion * d * n_c * wedge_reduction
        + soil.soil_tool_adhesion_kpa * d * n_ca;
    let required_force_n = (pressure_kpa * w * 1_000.0).max(10.0);

    let severed = tool.applied_force_n >= required_force_n;
    let margin = tool.applied_force_n - required_force_n;

    if severed {
        let bank_cut_volume = (w * d * d * 0.75).min(tool.capacity_volume_m3);
        let loose_volume = bank_cut_volume * soil.bulking_factor;
        let root_severed = soil.root_tensile_reinforcement_kpa > 0.0;

        ExcavationVerdict {
            severed: true,
            required_force_n,
            margin_force_n: margin,
            excavated_bank_volume_m3: bank_cut_volume,
            excavated_loose_volume_m3: loose_volume,
            root_mat_severed: root_severed,
            refusal_hit: false,
        }
    } else {
        ExcavationVerdict {
            severed: false,
            required_force_n,
            margin_force_n: margin,
            excavated_bank_volume_m3: 0.0,
            excavated_loose_volume_m3: 0.0,
            root_mat_severed: false,
            refusal_hit: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mouse_click_never_excavates() {
        let mouse = ToolEffector::mouse_click();
        let loam = SoilMechanicsProfile::loam();
        let verdict = evaluate_excavation_pass(&mouse, 0.15, &loam);
        assert!(!verdict.severed);
        assert_eq!(verdict.excavated_bank_volume_m3, 0.0);
    }

    #[test]
    fn hand_spade_slices_friable_loam() {
        let spade = ToolEffector::hand_spade();
        let loam = SoilMechanicsProfile::loam();
        // 10 cm slice in friable topsoil should yield cleanly
        let verdict = evaluate_excavation_pass(&spade, 0.10, &loam);
        assert!(verdict.severed);
        assert!(verdict.excavated_loose_volume_m3 > verdict.excavated_bank_volume_m3);
        assert!(!verdict.refusal_hit);
    }

    #[test]
    fn hand_spade_refuses_on_solid_bedrock() {
        let spade = ToolEffector::hand_spade();
        let rock = SoilMechanicsProfile::bedrock();
        let verdict = evaluate_excavation_pass(&spade, 0.05, &rock);
        assert!(!verdict.severed);
        assert!(verdict.refusal_hit);
    }

    #[test]
    fn excavator_rips_deep_dense_clay() {
        let spade = ToolEffector::hand_spade();
        let excavator = ToolEffector::mini_excavator();
        let clay = SoilMechanicsProfile::illuvial_clay();

        // Hand spade stalls in deep dense clay (35 cm deep)
        let spade_verdict = evaluate_excavation_pass(&spade, 0.35, &clay);
        assert!(!spade_verdict.severed);

        // Heavy excavator bucket breaks through
        let excavator_verdict = evaluate_excavation_pass(&excavator, 0.35, &clay);
        assert!(excavator_verdict.severed);
        assert!(excavator_verdict.excavated_loose_volume_m3 > 0.05);
    }
}
