//! Regime III: Inverted Cuticular Exoskeleton Solver (`exoskeletal_cuticle.rs`).
//!
//! Models arthropod and crustacean sclerite shells articulated via dual-pin bicondylar
//! (dicondylic) arthrodial hinges, internal flexor/extensor apodeme muscle routing,
//! hydraulic hemolymph extension torque $\tau_{\text{hyd}} = P_{\text{hemo}} A r_{\text{hyd}}$,
//! coelomic volume conservation, and a WGSL compute kernel (`HotZeroHeap`).

use super::{Motor, SkeletalError};

pub const MAX_EXO_SCLERITES: usize = 128;
pub const MAX_EXO_APODEMES: usize = 256;

/// Exterior load-bearing cuticular sclerite shell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CuticleSclerite {
    pub parent: Option<u16>,
    pub thickness_mm: f64,
    /// Chitin-protein cross-linking in `[0.05, 1.0]` (low immediately post-molt).
    pub sclerotization_ratio: f64,
    /// $\text{CaCO}_3$ biomineralization ratio in `[0.0, 1.0]`.
    pub caco3_calcification_ratio: f64,
    pub shell_mass_kg: f64,
    pub effective_inertia_kg_m2: f64,
    pub pose_world: Motor,
}

impl Default for CuticleSclerite {
    fn default() -> Self {
        Self {
            parent: None,
            thickness_mm: 0.4,
            sclerotization_ratio: 0.9,
            caco3_calcification_ratio: 0.1,
            shell_mass_kg: 0.002,
            effective_inertia_kg_m2: 1.5e-5,
            pose_world: Motor::identity(),
        }
    }
}

impl CuticleSclerite {
    /// Effective flexural cuticle stiffness scaling factor from sclerotization and calcification.
    #[inline]
    pub fn shell_rigidity_factor(&self) -> f64 {
        let scl = self.sclerotization_ratio.clamp(0.05, 1.0);
        let calc = self.caco3_calcification_ratio.clamp(0.0, 1.0);
        scl * (1.0 + 2.5 * calc) * (self.thickness_mm / 0.4).clamp(0.1, 10.0)
    }
}

/// Dual-pin dicondylic arthrodial hinge between parent and child sclerite shells.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BicondylarHinge {
    pub condyle_anterior_m: [f64; 3],
    pub condyle_posterior_m: [f64; 3],
    pub theta_rad: f64,
    pub omega_rad_s: f64,
    pub rest_angle_rad: f64,
    pub min_angle_rad: f64,
    pub max_angle_rad: f64,
    pub membrane_stiffness_nm_rad: f64,
    pub membrane_damping_nm_s_rad: f64,
    pub external_torque_nm: f64,
}

impl Default for BicondylarHinge {
    fn default() -> Self {
        Self {
            condyle_anterior_m: [-0.004, 0.0, 0.0],
            condyle_posterior_m: [0.004, 0.0, 0.0],
            theta_rad: 0.0,
            omega_rad_s: 0.0,
            rest_angle_rad: 0.0,
            min_angle_rad: -1.4,
            max_angle_rad: 1.4,
            membrane_stiffness_nm_rad: 0.005,
            membrane_damping_nm_s_rad: 0.0005,
            external_torque_nm: 0.0,
        }
    }
}

impl BicondylarHinge {
    /// Returns `(unit_axis, pivot_midpoint_m, condyle_span_m)` for the dual-pin hinge.
    pub fn axis_and_pivot(&self) -> Result<([f64; 3], [f64; 3], f64), SkeletalError> {
        let diff = [
            self.condyle_posterior_m[0] - self.condyle_anterior_m[0],
            self.condyle_posterior_m[1] - self.condyle_anterior_m[1],
            self.condyle_posterior_m[2] - self.condyle_anterior_m[2],
        ];
        let span = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
        if !span.is_finite() || span < 1e-7 {
            return Err(SkeletalError::InvalidInput);
        }
        let axis = [diff[0] / span, diff[1] / span, diff[2] / span];
        let pivot = [
            0.5 * (self.condyle_anterior_m[0] + self.condyle_posterior_m[0]),
            0.5 * (self.condyle_anterior_m[1] + self.condyle_posterior_m[1]),
            0.5 * (self.condyle_anterior_m[2] + self.condyle_posterior_m[2]),
        ];
        Ok((axis, pivot, span))
    }
}

/// Internal apodeme muscle fiber bundle attached inside the cuticular shell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApodemeActuator {
    pub hinge_idx: u16,
    /// `-1.0` for flexor, `+1.0` for extensor.
    pub action_sign: f64,
    pub lever_arm_m: f64,
    pub pcsa_cm2: f64,
    pub specific_tension_n_per_cm2: f64,
    pub activation: f64,
}

impl Default for ApodemeActuator {
    fn default() -> Self {
        Self {
            hinge_idx: 0,
            action_sign: -1.0,
            lever_arm_m: 0.003,
            pcsa_cm2: 0.04,
            specific_tension_n_per_cm2: 35.0,
            activation: 0.0,
        }
    }
}

/// Internal hemolymph / coelomic hydraulic chamber driving joint extension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HemolymphHydraulics {
    pub pressure_pa: f64,
    pub effective_area_m2: f64,
    pub hydraulic_moment_arm_m: f64,
    pub extension_sign: f64,
    pub chamber_rest_volume_m3: f64,
    pub bulk_modulus_pa: f64,
}

impl Default for HemolymphHydraulics {
    fn default() -> Self {
        Self {
            pressure_pa: 0.0,
            effective_area_m2: 1.2e-5,
            hydraulic_moment_arm_m: 0.0025,
            extension_sign: 1.0,
            chamber_rest_volume_m3: 2.0e-7,
            bulk_modulus_pa: 2.0e6,
        }
    }
}

impl HemolymphHydraulics {
    /// Hydraulic extension torque $\tau_{\text{hyd}} = s \cdot P_{\text{hemo}} A r_{\text{hyd}}$.
    #[inline]
    pub fn extension_torque_nm(&self) -> f64 {
        self.extension_sign
            * self.pressure_pa.max(0.0)
            * self.effective_area_m2.max(0.0)
            * self.hydraulic_moment_arm_m.max(0.0)
    }
}

/// GPU-aligned 64-byte dicondylic hinge + hemolymph state matching `EXOSKELETAL_CUTICLE_WGSL`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExoHingeGpu {
    pub axis_theta: [f32; 4],
    pub omega_bounds_inertia: [f32; 4],
    pub hyd_pressure_area_arm_sign: [f32; 4],
    pub torque_stiff_damp_rest: [f32; 4],
}

const _: [(); 64] = [(); core::mem::size_of::<ExoHingeGpu>()];

pub const EXOSKELETAL_CUTICLE_WGSL: &str = r#"
struct ExoParams {
    hinge_count: u32,
    dt: f32,
    _pad0: u32,
    _pad1: u32,
};

struct ExoHingeGpu {
    axis_theta: vec4<f32>,
    omega_bounds_inertia: vec4<f32>,
    hyd_pressure_area_arm_sign: vec4<f32>,
    torque_stiff_damp_rest: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: ExoParams;
@group(0) @binding(1) var<storage, read_write> hinges: array<ExoHingeGpu>;

@compute @workgroup_size(64)
fn step_exoskeletal_hinges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.hinge_count) {
        return;
    }
    var h = hinges[idx];
    let theta = h.axis_theta.w;
    let omega = h.omega_bounds_inertia.x;
    let min_a = h.omega_bounds_inertia.y;
    let max_a = h.omega_bounds_inertia.z;
    let inv_i = 1.0 / max(h.omega_bounds_inertia.w, 1e-7);

    let tau_hyd = max(h.hyd_pressure_area_arm_sign.x, 0.0)
        * max(h.hyd_pressure_area_arm_sign.y, 0.0)
        * max(h.hyd_pressure_area_arm_sign.z, 0.0)
        * h.hyd_pressure_area_arm_sign.w;
    let tau_mem = -h.torque_stiff_damp_rest.y * (theta - h.torque_stiff_damp_rest.w)
        - h.torque_stiff_damp_rest.z * omega;
    let tau_net = h.torque_stiff_damp_rest.x + tau_hyd + tau_mem;

    var omega_new = omega + tau_net * inv_i * params.dt;
    var theta_new = theta + omega_new * params.dt;
    if (theta_new < min_a) {
        theta_new = min_a;
        omega_new = max(omega_new, 0.0);
    } else if (theta_new > max_a) {
        theta_new = max_a;
        omega_new = min(omega_new, 0.0);
    }
    h.axis_theta.w = theta_new;
    h.omega_bounds_inertia.x = omega_new;
    hinges[idx] = h;
}
"#;

/// Advances Regime III cuticular sclerite hinges under apodeme torques, hydraulic hemolymph
/// extension pressure, and coelomic volume conservation (`HotZeroHeap`).
pub fn step_exoskeletal_cuticle_zero_heap(
    sclerites: &mut [CuticleSclerite],
    hinges: &mut [BicondylarHinge],
    hydraulics: &mut [HemolymphHydraulics],
    apodemes: &[ApodemeActuator],
    dt: f64,
) -> Result<f64, SkeletalError> {
    let n = sclerites.len();
    if n == 0
        || n > MAX_EXO_SCLERITES
        || hinges.len() != n
        || hydraulics.len() != n
        || apodemes.len() > MAX_EXO_APODEMES
        || !dt.is_finite()
        || dt <= 0.0
    {
        return Err(SkeletalError::InvalidInput);
    }

    for (i, s) in sclerites.iter().enumerate() {
        if s.parent.is_some_and(|p| p as usize >= i)
            || !s.effective_inertia_kg_m2.is_finite()
            || s.effective_inertia_kg_m2 <= 0.0
            || !s.sclerotization_ratio.is_finite()
        {
            return Err(SkeletalError::InvalidInput);
        }
    }

    let mut axes = [[1.0f64, 0.0, 0.0]; MAX_EXO_SCLERITES];
    let mut pivots = [[0.0f64; 3]; MAX_EXO_SCLERITES];
    for i in 0..n {
        let h = &hinges[i];
        if !h.theta_rad.is_finite()
            || !h.omega_rad_s.is_finite()
            || !h.min_angle_rad.is_finite()
            || !h.max_angle_rad.is_finite()
            || h.min_angle_rad > h.max_angle_rad
        {
            return Err(SkeletalError::InvalidInput);
        }
        let (ax, piv, _) = h.axis_and_pivot()?;
        axes[i] = ax;
        pivots[i] = piv;
    }

    let mut apodeme_torques = [0.0f64; MAX_EXO_SCLERITES];
    for ap in apodemes {
        let idx = ap.hinge_idx as usize;
        if idx >= n
            || !ap.activation.is_finite()
            || !ap.lever_arm_m.is_finite()
            || !ap.pcsa_cm2.is_finite()
            || !ap.specific_tension_n_per_cm2.is_finite()
        {
            return Err(SkeletalError::InvalidInput);
        }
        let f_max = ap.pcsa_cm2.max(0.0) * ap.specific_tension_n_per_cm2.max(0.0);
        // Compliant post-molt cuticle yields slightly under high muscle force.
        let compliance = sclerites[idx].sclerotization_ratio.clamp(0.15, 1.0);
        apodeme_torques[idx] += ap.action_sign
            * ap.activation.clamp(0.0, 1.0)
            * f_max
            * ap.lever_arm_m.max(0.0)
            * compliance;
    }

    // Coelomic volume conservation across connected joints:
    // flexion squeezes hemolymph into coupled chambers, raising global pressure slightly.
    let mut total_rest_vol = 0.0f64;
    let mut total_curr_vol = 0.0f64;
    for i in 0..n {
        let d_theta = hinges[i].theta_rad - hinges[i].rest_angle_rad;
        let dv = hydraulics[i].extension_sign
            * hydraulics[i].effective_area_m2
            * hydraulics[i].hydraulic_moment_arm_m
            * d_theta;
        total_rest_vol += hydraulics[i].chamber_rest_volume_m3.max(1e-12);
        total_curr_vol += (hydraulics[i].chamber_rest_volume_m3 + dv).max(1e-12);
    }
    let coelomic_strain = (total_rest_vol - total_curr_vol) / total_rest_vol.max(1e-12);

    let mut max_volume_strain = coelomic_strain.abs();
    for i in 0..n {
        let p_eff =
            (hydraulics[i].pressure_pa + hydraulics[i].bulk_modulus_pa * coelomic_strain).max(0.0);
        let tau_hyd = hydraulics[i].extension_sign
            * p_eff
            * hydraulics[i].effective_area_m2.max(0.0)
            * hydraulics[i].hydraulic_moment_arm_m.max(0.0);
        let h = &mut hinges[i];
        let rigidity = sclerites[i].shell_rigidity_factor().clamp(0.1, 10.0);
        let tau_mem = -h.membrane_stiffness_nm_rad * rigidity * (h.theta_rad - h.rest_angle_rad)
            - h.membrane_damping_nm_s_rad * h.omega_rad_s;
        let tau_net = apodeme_torques[i] + tau_hyd + tau_mem + h.external_torque_nm;

        h.omega_rad_s += (tau_net / sclerites[i].effective_inertia_kg_m2) * dt;
        h.theta_rad += h.omega_rad_s * dt;

        if h.theta_rad < h.min_angle_rad {
            h.theta_rad = h.min_angle_rad;
            h.omega_rad_s = h.omega_rad_s.max(0.0);
        } else if h.theta_rad > h.max_angle_rad {
            h.theta_rad = h.max_angle_rad;
            h.omega_rad_s = h.omega_rad_s.min(0.0);
        }
        hydraulics[i].pressure_pa = hydraulics[i].pressure_pa.max(0.0);

        let half = 0.5 * h.theta_rad;
        let (s_half, c_half) = half.sin_cos();
        let ax = axes[i];
        let local_motor = Motor::from_rotation_translation(
            [c_half, ax[0] * s_half, ax[1] * s_half, ax[2] * s_half],
            pivots[i],
        );
        sclerites[i].pose_world = if let Some(p) = sclerites[i].parent {
            sclerites[p as usize]
                .pose_world
                .mul(&local_motor)
                .normalize()
        } else {
            local_motor
        };
        max_volume_strain = max_volume_strain.max(coelomic_strain.abs());
    }

    Ok(max_volume_strain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bicondylar_hinge_and_hemolymph_hydraulic_extension_torque() {
        let mut sclerites = [CuticleSclerite::default()];
        let mut hinges = [BicondylarHinge {
            condyle_anterior_m: [0.0, 0.0, -0.005],
            condyle_posterior_m: [0.0, 0.0, 0.005],
            theta_rad: -0.4,
            min_angle_rad: -1.5,
            max_angle_rad: 1.5,
            ..Default::default()
        }];
        let mut hydraulics = [HemolymphHydraulics {
            pressure_pa: 60_000.0,
            effective_area_m2: 2.0e-5,
            hydraulic_moment_arm_m: 0.004,
            extension_sign: 1.0,
            ..Default::default()
        }];
        // Pure hydraulic extension (zero extensor apodeme activation).
        assert!(hydraulics[0].extension_torque_nm() > 0.0);
        let init_theta = hinges[0].theta_rad;
        for _ in 0..25 {
            step_exoskeletal_cuticle_zero_heap(
                &mut sclerites,
                &mut hinges,
                &mut hydraulics,
                &[],
                0.002,
            )
            .unwrap();
        }
        assert!(hinges[0].theta_rad > init_theta);
    }

    #[test]
    fn flexor_apodeme_counteracts_hemolymph_pressure() {
        let mut sclerites = [CuticleSclerite::default()];
        let mut hinges = [BicondylarHinge {
            theta_rad: 0.0,
            ..Default::default()
        }];
        let mut hydraulics = [HemolymphHydraulics {
            pressure_pa: 15_000.0,
            ..Default::default()
        }];
        let apodemes = [ApodemeActuator {
            hinge_idx: 0,
            action_sign: -1.0,
            lever_arm_m: 0.004,
            pcsa_cm2: 0.08,
            specific_tension_n_per_cm2: 35.0,
            activation: 1.0,
        }];
        for _ in 0..20 {
            step_exoskeletal_cuticle_zero_heap(
                &mut sclerites,
                &mut hinges,
                &mut hydraulics,
                &apodemes,
                0.002,
            )
            .unwrap();
        }
        assert!(hinges[0].theta_rad < 0.0);
    }

    #[test]
    fn exoskeletal_wgsl_shader_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(EXOSKELETAL_CUTICLE_WGSL).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
