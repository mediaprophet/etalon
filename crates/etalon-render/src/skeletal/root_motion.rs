//! Loop-Safe Root Motion Extraction & Ontogenetic Posture Modulation (`root_motion.rs`).

use super::absm_types::{BoneLayerMask, OntogeneticStateHeader};
use super::{Motor, MAX_JOINTS};

/// Frame-to-frame root motion displacement extracted from the skeleton's root joint.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootMotionDelta {
    /// World/character-local horizontal and vertical translation delta `[dx, dy, dz]`.
    pub delta_translation: [f32; 3],
    /// Relative root rotation quaternion `[r_w, r_x, r_y, r_z]`.
    pub delta_rotor: [f32; 4],
}

impl Default for RootMotionDelta {
    fn default() -> Self {
        Self {
            delta_translation: [0.0, 0.0, 0.0],
            delta_rotor: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

/// Extracts continuous root motion from the root bone (eliminating loop-wrap teleport spikes)
/// and locks horizontal `(x, z)` drift on the visual skeleton.
#[derive(Clone, Copy, Debug)]
pub struct RootMotionExtractor {
    pub root_joint_index: usize,
    pub lock_horizontal: bool,
    pub lock_vertical: bool,
    pub prev_root_motor: Motor,
    pub loop_start_motor: Motor,
    pub loop_end_motor: Motor,
    pub initialized: bool,
}

impl Default for RootMotionExtractor {
    fn default() -> Self {
        Self::new(0, Motor::identity(), Motor::identity())
    }
}

impl RootMotionExtractor {
    /// Creates a root motion extractor for `root_joint_index` with authored clip seam poses.
    pub const fn new(
        root_joint_index: usize,
        loop_start_motor: Motor,
        loop_end_motor: Motor,
    ) -> Self {
        Self {
            root_joint_index,
            lock_horizontal: true,
            lock_vertical: false,
            prev_root_motor: loop_start_motor,
            loop_start_motor,
            loop_end_motor,
            initialized: false,
        }
    }

    /// Resets tracking to `current_root`.
    pub fn reset(&mut self, current_root: Motor) {
        self.prev_root_motor = current_root;
        self.initialized = true;
    }

    /// Extracts the frame's `RootMotionDelta` from `local_pose[self.root_joint_index]`
    /// and zeroes out horizontal `(x, z)` root bone drift in-place (`HotZeroHeap`).
    pub fn extract_and_lock(
        &mut self,
        local_pose: &mut [Motor],
        looped_this_frame: bool,
    ) -> RootMotionDelta {
        if self.root_joint_index >= local_pose.len() {
            return RootMotionDelta::default();
        }
        let curr = local_pose[self.root_joint_index];
        if !self.initialized {
            self.prev_root_motor = curr;
            self.initialized = true;
            if self.lock_horizontal {
                let (rot, trans) = curr.to_rotation_translation();
                let (_, start_t) = self.loop_start_motor.to_rotation_translation();
                let locked_y = if self.lock_vertical {
                    start_t[1]
                } else {
                    trans[1]
                };
                local_pose[self.root_joint_index] =
                    Motor::from_rotation_translation(rot, [start_t[0], locked_y, start_t[2]]);
            }
            return RootMotionDelta::default();
        }

        let (_, prev_t) = self.prev_root_motor.to_rotation_translation();
        let (curr_r, curr_t) = curr.to_rotation_translation();

        let delta_t = if !looped_this_frame {
            [
                (curr_t[0] - prev_t[0]) as f32,
                (curr_t[1] - prev_t[1]) as f32,
                (curr_t[2] - prev_t[2]) as f32,
            ]
        } else {
            let (_, start_t) = self.loop_start_motor.to_rotation_translation();
            let (_, end_t) = self.loop_end_motor.to_rotation_translation();
            [
                ((end_t[0] - prev_t[0]) + (curr_t[0] - start_t[0])) as f32,
                ((end_t[1] - prev_t[1]) + (curr_t[1] - start_t[1])) as f32,
                ((end_t[2] - prev_t[2]) + (curr_t[2] - start_t[2])) as f32,
            ]
        };

        let delta_motor = self.prev_root_motor.conjugate().mul(&curr).normalize();
        self.prev_root_motor = curr;

        if self.lock_horizontal {
            let (_, start_t) = self.loop_start_motor.to_rotation_translation();
            let locked_y = if self.lock_vertical {
                start_t[1]
            } else {
                curr_t[1]
            };
            local_pose[self.root_joint_index] =
                Motor::from_rotation_translation(curr_r, [start_t[0], locked_y, start_t[2]]);
        }

        RootMotionDelta {
            delta_translation: delta_t,
            delta_rotor: [
                delta_motor.r_w as f32,
                delta_motor.r_x as f32,
                delta_motor.r_y as f32,
                delta_motor.r_z as f32,
            ],
        }
    }
}

/// Ontogenetic and metabolic posture/cadence modulator (`EONT1` compatible).
#[derive(Clone, Copy, Debug)]
pub struct EontPostureConfig {
    pub enabled: bool,
    /// Mask of spine/neck/posture joints affected by fatigue/age pitch droop.
    pub spine_mask: BoneLayerMask,
    /// Lifecycle stage (`0=SeedOrSpore, 1=Juvenile, 2=Adult, 3=Senescent, 4=Deceased`).
    pub stage: u8,
    pub energy_permill: u16,
    pub damage_permill: u16,
    pub hydration_permill: u16,
    /// Maximum forward pitch droop angle in radians at full exhaustion/senescence.
    pub max_droop_rad: f32,
}

impl Default for EontPostureConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            spine_mask: BoneLayerMask::NONE,
            stage: 2,
            energy_permill: 1000,
            damage_permill: 0,
            hydration_permill: 1000,
            max_droop_rad: 0.35,
        }
    }
}

impl EontPostureConfig {
    /// Updates metabolic/ontogenetic fields directly from an `OntogeneticStateHeader` (`EntityOntologyMiniHeader`).
    #[inline]
    pub fn update_from_header(&mut self, header: &OntogeneticStateHeader) {
        self.stage = header.stage;
        self.energy_permill = header.energy_permill;
        self.damage_permill = header.damage_permill;
        self.hydration_permill = header.hydration_permill;
    }

    /// Computes a cadence speed multiplier in `[0.25, 1.15]` driven by age and metabolic energy.
    #[inline]
    pub fn cadence_multiplier(&self) -> f32 {
        if !self.enabled {
            return 1.0;
        }
        let stage_factor = match self.stage {
            0 => 0.50,
            1 => 1.10, // Juvenile brisk cadence
            2 => 1.00, // Adult baseline
            3 => 0.75, // Senescent slower gait
            _ => 0.25,
        };
        let energy_norm = (self.energy_permill.min(1000) as f32) * 0.001;
        let damage_norm = (self.damage_permill.min(1000) as f32) * 0.001;
        let vitality = (0.5 + 0.5 * energy_norm) * (1.0 - 0.4 * damage_norm);
        (stage_factor * vitality).clamp(0.25, 1.15)
    }

    /// Applies ontogenetic/metabolic posture pitch droop onto joints in `self.spine_mask`.
    pub fn apply_posture_droop(&self, pose: &mut [Motor]) {
        if !self.enabled || self.spine_mask == BoneLayerMask::NONE {
            return;
        }
        let fatigue = 1.0 - (self.energy_permill.min(1000) as f32) * 0.001;
        let damage = (self.damage_permill.min(1000) as f32) * 0.001;
        let dehydration = 1.0 - (self.hydration_permill.min(1000) as f32) * 0.001;
        let age_stoop = if self.stage >= 3 { 0.35 } else { 0.0 };

        let droop_intensity =
            (0.45 * fatigue + 0.35 * damage + 0.20 * dehydration + age_stoop).clamp(0.0, 1.0);
        if droop_intensity <= 1e-4 {
            return;
        }
        let half_angle = (0.5 * droop_intensity * self.max_droop_rad) as f64;
        let droop_rotor = Motor::from_rotation_translation(
            [half_angle.cos(), half_angle.sin(), 0.0, 0.0],
            [0.0, 0.0, 0.0],
        );

        let limit = pose.len().min(MAX_JOINTS);
        for j in 0..limit {
            if self.spine_mask.contains(j) {
                pose[j] = pose[j].mul(&droop_rotor).normalize();
            }
        }
    }
}
