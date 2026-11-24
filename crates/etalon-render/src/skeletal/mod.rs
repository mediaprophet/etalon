//! Zero-Heap Skeletal Animation, ABSM, Blend Spaces, Motor IK & Dual-Quaternion Skinning (`skeletal`).
//!
//! Unifies:
//! - `pga_f64`: `f64` PGA $\mathcal{G}_{3,0,1}$ `Motor` and `MotorBivector` (`log`, `exp`, `sclerp`)
//!   with bidirectional conversion to `crate::spatial_math::pga::Motor` (`f32`).
//! - `pose`, `skin`, `gpu_skin`, `gpu_morph`, `ik_motor`: caller-buffered pose sampling,
//!   4-influence CPU/GPU dual-quaternion skinning, ontogenetic morph targets, and analytical/FABRIK IK.
//! - `blend_space_1d`, `blend_space_2d` (re-exported under `blend_space`): 1D & 2D Bowyer–Watson
//!   Delaunay blend spaces.
//! - `absm_types`, `absm_machine`, `root_motion` (re-exported under `absm`): multi-layer bone-masked
//!   Animation Blending State Machine with timeline signals and ontogenetic posture modulation.
//! - `transform_hierarchy`, `clip_state_machine` (also accessible under `inertial_skeleton`):
//!   `Transform3D`, `ZeroHeapSkeleton`, `AnimationStateMachine`, and direct dual-quaternion GPU
//!   palette export.

pub mod absm_machine;
pub mod absm_types;
pub mod activity;
pub mod authoring;
pub mod blend_space_1d;
pub mod blend_space_2d;
pub mod clip_state_machine;
pub mod cosserat_rod;
pub mod dca_multibody;
pub mod exoskeletal_cuticle;
pub mod gpu_morph;
pub mod gpu_skin;
pub mod ik_motor;
pub mod mobility_risk_eval;
pub mod pga_f64;
pub mod player;
pub mod pose;
pub mod root_motion;
pub mod skin;
pub mod time_policy;
pub mod transform_hierarchy;
pub mod vbd_soft_tissue;

/// Submodule namespace preserving `skeletal::absm::*` import paths.
pub mod absm {
    pub use super::absm_machine::*;
    pub use super::absm_types::*;
    pub use super::root_motion::*;
}

/// Submodule namespace preserving `skeletal::blend_space::*` import paths.
pub mod blend_space {
    pub use super::blend_space_1d::*;
    pub use super::blend_space_2d::*;
}

/// Submodule namespace grouping `transform_hierarchy` and `clip_state_machine`.
pub mod inertial_skeleton {
    pub use super::clip_state_machine::*;
    pub use super::transform_hierarchy::*;
}

pub use absm::{
    AbsmLayer, AbsmMachine, AbsmParameter, AbsmStateNode, AbsmStateSource, AbsmTransition,
    AnimationSignal, BoneLayerMask, EontPostureConfig, LayerBlendMode, LeafCondition,
    RootMotionDelta, RootMotionExtractor, SignalQueue, SignalTrigger, TransitionCondition,
    MAX_ABSM_PARAMS, MAX_SIGNALS_PER_STATE,
};
pub use blend_space::{
    sample_blend_space_1d_zero_heap, sample_blend_space_2d_zero_heap, BlendPoint1D, BlendPoint2D,
    BlendSample2D, BlendSpace1D, BlendSpace2D, BlendTriangle,
};
pub use clip_state_machine::{
    blend_skeleton_layer_zero_heap, sample_pose_2d_zero_heap, AnimationStateMachine, BoneMask64,
    LayerBlendOp, SimAnimationSignal, SimBlendPoint2D, SimBlendSpace2D, SimRootMotionDelta,
    SimRootMotionExtractor, SimSignalQueue, TransitionMode,
};
pub use cosserat_rod::{
    compute_darboux_vector, hydrostatic_cavity_axial_force, step_cosserat_rod_zero_heap,
    CosseratElement, CosseratElementGpu, CosseratEnvironment, CosseratNode, COSSERAT_ROD_WGSL,
    MAX_COSSERAT_NODES,
};
pub use dca_multibody::{
    resolve_loop_closures_zero_heap, solve_aba_zero_heap, solve_dca_parallel_chain_zero_heap,
    BodyDensitometryInertia, DcaAssemblyNodeGpu, DcaBodyGpu, DcaBodyNode, LoopClosureConstraint,
    SpatialMat6, SpatialVec6, DCA_MULTIBODY_WGSL, MAX_DCA_BODIES, MAX_DCA_LOOPS,
};
pub use exoskeletal_cuticle::{
    step_exoskeletal_cuticle_zero_heap, ApodemeActuator, BicondylarHinge, CuticleSclerite,
    ExoHingeGpu, HemolymphHydraulics, EXOSKELETAL_CUTICLE_WGSL, MAX_EXO_APODEMES,
    MAX_EXO_SCLERITES,
};
pub use gpu_morph::{
    compute_ontogenetic_morph_weights_from_header_zero_heap,
    compute_ontogenetic_morph_weights_zero_heap, evaluate_morph_deltas_zero_heap,
    ontogenetic_stage_growth_factor, EntityOntologyMiniHeader, GpuInstanceMorphWeights,
    GpuMorphDelta, OntogeneticStateHeader, MAX_GPU_MORPH_TARGETS,
};
pub use gpu_skin::{
    evaluate_skinning_motor_palette_zero_heap, gpu_shader_skin_vertex_reference,
    hemisphere_align_motor, skin_vertices_gpu_reference_into, GpuJointMotorPair, GpuSkinnedVertex,
    MAX_GPU_RIG_JOINTS, MAX_GPU_SKELETON_INSTANCES,
};
pub use ik_motor::{
    solve_constrained_fabrik_motor_ik_zero_heap, solve_two_bone_motor_ik_zero_heap,
    JointConstraint, TwoBoneIkResult, MAX_IK_CHAIN,
};
pub use mobility_risk_eval::{
    evaluate_bone_fracture_load, evaluate_coupled_mobility_and_fracture_risk,
    evaluate_gait_stability, evaluate_joint_mobility, sideways_fall_trochanteric_impact_force_n,
    AnatomicBoneSite, BiologicalSexBasis, BoneFractureLoadEvaluation, BoneFractureLoadInput,
    CoupledMobilityFractureSummary, FractureCallusStage, GaitStabilityEvaluation,
    GaitStabilityInput, JointMobilityEvaluation, JointMobilityInput,
};
pub use pga_f64::{Motor, MotorBivector};
pub use player::{
    clip_view, BodyFitSource, BodyFitSpec, ClipRecord, PlaybackReceipt, PlayerError,
    SkeletalCounts, SkeletalPlayer, SkeletalRigDescriptor, SkeletalSectionSource,
};
pub use pose::{compose_palette_into, sample_pose_into};
pub use skin::{skin_vertices_into, SkinWeight};
pub use time_policy::PoseTimePolicy;
pub use transform_hierarchy::{
    AnimationClip, AnimationEvent, BoneTrack, Keyframe3D, Transform3D, ZeroHeapSkeleton, MAX_BONES,
};
pub use vbd_soft_tissue::{
    step_vbd_soft_tissue_zero_heap, VbdTetElement, VbdVertex, VbdVertexGpu, MAX_VBD_COLORS,
    MAX_VBD_TETS, MAX_VBD_VERTICES, VBD_SOFT_TISSUE_WGSL,
};

pub const MAX_JOINTS: usize = 128;
pub const MAX_KEYS: usize = 4096;
pub const MAX_VERTICES: usize = 262_144;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkeletalError {
    InvalidInput,
    OutputTooSmall,
    BudgetExceeded,
}

#[derive(Debug, Clone, Copy)]
pub struct Joint {
    /// `None` for a root; otherwise strictly less than this joint's index.
    pub parent: Option<u16>,
    pub rest_local: Motor,
    pub inverse_bind: Motor,
}

#[derive(Debug, Clone, Copy)]
pub struct Key {
    pub time: f64,
    pub local: Motor,
}

#[derive(Debug, Clone, Copy)]
pub struct Track {
    pub joint: u16,
    pub first_key: u32,
    pub key_count: u16,
}

#[derive(Debug, Clone, Copy)]
pub struct Clip<'a> {
    pub duration: f64,
    pub looping: bool,
    pub tracks: &'a [Track],
    pub keys: &'a [Key],
}

/// Returns `true` when `m` is a finite, unit-norm, Plücker-orthogonal PGA `Motor`.
pub fn valid_motor(m: &Motor) -> bool {
    let real = [m.r_w, m.r_x, m.r_y, m.r_z];
    let dual = [m.d_w, m.d_x, m.d_y, m.d_z];
    real.iter()
        .chain(dual.iter())
        .all(|v| v.is_finite() && v.abs() < 1e8)
        && (real.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-5
        && real.iter().zip(dual).map(|(r, d)| r * d).sum::<f64>().abs() < 1e-5
}

/// Validates topological parent ordering and motor invariants for `joints`.
pub fn validate_joints(joints: &[Joint]) -> Result<(), SkeletalError> {
    if joints.len() > MAX_JOINTS {
        return Err(SkeletalError::BudgetExceeded);
    }
    if joints.is_empty()
        || joints.iter().enumerate().any(|(i, j)| {
            j.parent.is_some_and(|p| p as usize >= i)
                || !valid_motor(&j.rest_local)
                || !valid_motor(&j.inverse_bind)
        })
    {
        return Err(SkeletalError::InvalidInput);
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn assert_zero_alloc<F: FnOnce() -> R, R>(_label: &str, f: F) -> R {
    f()
}

#[cfg(test)]
mod absm_tests;
#[cfg(test)]
mod gpu_skin_tests;
#[cfg(test)]
mod tests;
