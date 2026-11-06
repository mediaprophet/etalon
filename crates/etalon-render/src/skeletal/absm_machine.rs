//! Zero-Heap Multi-Layer Animation Blending State Machine Evaluator (`absm_machine.rs`).

use super::absm_types::{
    AbsmParameter, AbsmStateNode, AbsmStateSource, AbsmTransition, BoneLayerMask, LayerBlendMode,
    SignalQueue, SignalTrigger, TransitionCondition, MAX_ABSM_PARAMS, MAX_SIGNALS_PER_STATE,
};
use super::blend_space_1d::{sample_blend_space_1d_zero_heap, BlendSpace1D};
use super::blend_space_2d::{sample_blend_space_2d_zero_heap, BlendSpace2D};
use super::root_motion::{EontPostureConfig, RootMotionDelta, RootMotionExtractor};
use super::{Motor, SkeletalError, MAX_JOINTS};

/// Single animation layer within an [`AbsmMachine`].
#[derive(Clone, Copy, Debug)]
pub struct AbsmLayer<const MAX_STATES: usize = 16, const MAX_TRANSITIONS: usize = 32> {
    pub enabled: bool,
    pub mask: BoneLayerMask,
    pub blend_mode: LayerBlendMode,
    pub weight: f32,
    pub states: [AbsmStateNode; MAX_STATES],
    pub state_count: usize,
    pub transitions: [AbsmTransition; MAX_TRANSITIONS],
    pub transition_count: usize,
    pub current_state: u8,
    pub target_state: Option<u8>,
    pub state_time: f32,
    pub target_state_time: f32,
    pub transition_elapsed: f32,
    pub transition_duration: f32,
    pub looped_last_step: bool,
}

impl<const MAX_STATES: usize, const MAX_TRANSITIONS: usize> Default
    for AbsmLayer<MAX_STATES, MAX_TRANSITIONS>
{
    fn default() -> Self {
        Self::new(BoneLayerMask::ALL, LayerBlendMode::Override, 1.0)
    }
}

impl<const MAX_STATES: usize, const MAX_TRANSITIONS: usize> AbsmLayer<MAX_STATES, MAX_TRANSITIONS> {
    pub const fn new(mask: BoneLayerMask, blend_mode: LayerBlendMode, weight: f32) -> Self {
        Self {
            enabled: false,
            mask,
            blend_mode,
            weight,
            states: [AbsmStateNode {
                active: false,
                source: AbsmStateSource::SingleClip { clip_index: 0 },
                duration: 1.0,
                speed: 1.0,
                looping: true,
                triggers: [SignalTrigger {
                    active: false,
                    signal_id: 0,
                    normalized_time: 0.0,
                    sonic_archetype: 0,
                    payload_f32: 0.0,
                }; MAX_SIGNALS_PER_STATE],
            }; MAX_STATES],
            state_count: 0,
            transitions: [AbsmTransition {
                active: false,
                from_state: 0,
                to_state: 0,
                duration: 0.2,
                condition: TransitionCondition::StateTimeFinished,
            }; MAX_TRANSITIONS],
            transition_count: 0,
            current_state: 0,
            target_state: None,
            state_time: 0.0,
            target_state_time: 0.0,
            transition_elapsed: 0.0,
            transition_duration: 0.0,
            looped_last_step: false,
        }
    }

    pub fn add_state(&mut self, mut node: AbsmStateNode) -> Option<u8> {
        if self.state_count >= MAX_STATES {
            return None;
        }
        node.active = true;
        let idx = self.state_count;
        self.states[idx] = node;
        self.state_count += 1;
        self.enabled = true;
        Some(idx as u8)
    }

    pub fn add_transition(
        &mut self,
        from_state: u8,
        to_state: u8,
        duration: f32,
        condition: TransitionCondition,
    ) -> bool {
        if self.transition_count >= MAX_TRANSITIONS
            || (from_state as usize) >= self.state_count
            || (to_state as usize) >= self.state_count
        {
            return false;
        }
        self.transitions[self.transition_count] = AbsmTransition {
            active: true,
            from_state,
            to_state,
            duration: duration.max(0.0),
            condition,
        };
        self.transition_count += 1;
        true
    }
}

/// Multi-layer zero-heap Animation Blending State Machine (`AbsmMachine`).
#[derive(Clone, Copy, Debug)]
pub struct AbsmMachine<
    const MAX_LAYERS: usize = 4,
    const MAX_STATES: usize = 16,
    const MAX_TRANSITIONS: usize = 32,
> {
    pub layers: [AbsmLayer<MAX_STATES, MAX_TRANSITIONS>; MAX_LAYERS],
    pub layer_count: usize,
    pub params: [AbsmParameter; MAX_ABSM_PARAMS],
    pub blend_spaces_1d: [BlendSpace1D<8>; 4],
    pub blend_spaces_2d: [BlendSpace2D<16, 32>; 4],
    pub eont_posture: EontPostureConfig,
    pub root_motion: Option<RootMotionExtractor>,
}

impl<const MAX_LAYERS: usize, const MAX_STATES: usize, const MAX_TRANSITIONS: usize> Default
    for AbsmMachine<MAX_LAYERS, MAX_STATES, MAX_TRANSITIONS>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_LAYERS: usize, const MAX_STATES: usize, const MAX_TRANSITIONS: usize>
    AbsmMachine<MAX_LAYERS, MAX_STATES, MAX_TRANSITIONS>
{
    /// Creates an empty multi-layer ABSM.
    pub fn new() -> Self {
        Self {
            layers: [AbsmLayer::default(); MAX_LAYERS],
            layer_count: 0,
            params: [AbsmParameter::Float(0.0); MAX_ABSM_PARAMS],
            blend_spaces_1d: [BlendSpace1D::new(); 4],
            blend_spaces_2d: [BlendSpace2D::new(); 4],
            eont_posture: EontPostureConfig::default(),
            root_motion: None,
        }
    }

    #[inline]
    pub fn set_parameter(&mut self, idx: usize, value: AbsmParameter) {
        if idx < MAX_ABSM_PARAMS {
            self.params[idx] = value;
        }
    }

    #[inline]
    pub fn add_layer(&mut self, layer: AbsmLayer<MAX_STATES, MAX_TRANSITIONS>) -> Option<usize> {
        if self.layer_count >= MAX_LAYERS {
            return None;
        }
        let idx = self.layer_count;
        self.layers[idx] = layer;
        self.layers[idx].enabled = true;
        self.layer_count += 1;
        Some(idx)
    }

    fn evaluate_state_source(
        &self,
        source: AbsmStateSource,
        clip_poses: &[&[Motor]],
        rest_pose: &[Motor],
        out_pose: &mut [Motor],
    ) {
        let joint_count = rest_pose.len().min(out_pose.len()).min(MAX_JOINTS);
        match source {
            AbsmStateSource::SingleClip { clip_index } => {
                if let Some(clip) = clip_poses.get(clip_index as usize) {
                    let n = joint_count.min(clip.len());
                    out_pose[..n].copy_from_slice(&clip[..n]);
                } else {
                    out_pose[..joint_count].copy_from_slice(&rest_pose[..joint_count]);
                }
            }
            AbsmStateSource::Blend1D {
                space_index,
                param_idx,
            } => {
                let p = match self.params.get(param_idx as usize) {
                    Some(AbsmParameter::Float(v)) => *v,
                    _ => 0.0,
                };
                if let Some(space) = self.blend_spaces_1d.get(space_index as usize) {
                    if sample_blend_space_1d_zero_heap(
                        space,
                        p,
                        clip_poses,
                        &mut out_pose[..joint_count],
                    )
                    .is_err()
                    {
                        out_pose[..joint_count].copy_from_slice(&rest_pose[..joint_count]);
                    }
                } else {
                    out_pose[..joint_count].copy_from_slice(&rest_pose[..joint_count]);
                }
            }
            AbsmStateSource::Blend2D {
                space_index,
                param_idx,
            } => {
                let q = match self.params.get(param_idx as usize) {
                    Some(AbsmParameter::Vec2(v)) => *v,
                    _ => [0.0, 0.0],
                };
                if let Some(space) = self.blend_spaces_2d.get(space_index as usize) {
                    if sample_blend_space_2d_zero_heap(
                        space,
                        q,
                        clip_poses,
                        &mut out_pose[..joint_count],
                    )
                    .is_err()
                    {
                        out_pose[..joint_count].copy_from_slice(&rest_pose[..joint_count]);
                    }
                } else {
                    out_pose[..joint_count].copy_from_slice(&rest_pose[..joint_count]);
                }
            }
        }
    }

    /// Advances all active layers by `dt` seconds, evaluates state transitions, fires timeline
    /// [`super::absm_types::AnimationSignal`]s into `signals`, blends `Override` and `Additive`
    /// bone-masked layers, applies ontogenetic/metabolic posture modulation, and extracts
    /// [`RootMotionDelta`] — all with zero heap allocations (`HotZeroHeap`).
    pub fn step_and_evaluate_zero_heap<const SIG_CAP: usize>(
        &mut self,
        dt: f32,
        rest_pose: &[Motor],
        clip_poses: &[&[Motor]],
        out_pose: &mut [Motor],
        signals: &mut SignalQueue<SIG_CAP>,
    ) -> Result<RootMotionDelta, SkeletalError> {
        let joint_count = rest_pose.len().min(MAX_JOINTS);
        if joint_count == 0 || !dt.is_finite() || dt < 0.0 {
            return Err(SkeletalError::InvalidInput);
        }
        if out_pose.len() < joint_count {
            return Err(SkeletalError::OutputTooSmall);
        }

        out_pose[..joint_count].copy_from_slice(&rest_pose[..joint_count]);
        let cadence = self.eont_posture.cadence_multiplier();
        let eff_dt = dt * cadence;

        let mut layer_pose = [Motor::identity(); MAX_JOINTS];
        let mut target_pose = [Motor::identity(); MAX_JOINTS];

        for l_idx in 0..self.layer_count.min(MAX_LAYERS) {
            if !self.layers[l_idx].enabled
                || self.layers[l_idx].state_count == 0
                || self.layers[l_idx].weight <= 1e-6
            {
                continue;
            }

            let mut curr_idx = self.layers[l_idx].current_state as usize;
            if curr_idx >= self.layers[l_idx].state_count {
                continue;
            }

            if self.layers[l_idx].target_state.is_none() {
                let init_dur = self.layers[l_idx].states[curr_idx].duration.max(1e-4);
                let init_norm = (self.layers[l_idx].state_time / init_dur).clamp(0.0, 1.0);
                let t_count = self.layers[l_idx].transition_count.min(MAX_TRANSITIONS);
                for tr_idx in 0..t_count {
                    let tr = self.layers[l_idx].transitions[tr_idx];
                    if tr.active
                        && tr.from_state == self.layers[l_idx].current_state
                        && tr.condition.evaluate(&self.params, init_norm)
                    {
                        if tr.duration <= 1e-4 {
                            self.layers[l_idx].current_state = tr.to_state;
                            self.layers[l_idx].state_time = 0.0;
                            curr_idx = tr.to_state as usize;
                        } else {
                            self.layers[l_idx].target_state = Some(tr.to_state);
                            self.layers[l_idx].target_state_time = 0.0;
                            self.layers[l_idx].transition_elapsed = 0.0;
                            self.layers[l_idx].transition_duration = tr.duration;
                        }
                        break;
                    }
                }
            }

            let state_node = self.layers[l_idx].states[curr_idx];
            let dur = state_node.duration.max(1e-4);
            let prev_norm = (self.layers[l_idx].state_time / dur).clamp(0.0, 1.0);

            self.layers[l_idx].state_time += eff_dt * state_node.speed;
            let mut looped = false;
            if state_node.looping {
                if self.layers[l_idx].state_time >= dur {
                    self.layers[l_idx].state_time = self.layers[l_idx].state_time.rem_euclid(dur);
                    looped = true;
                }
            } else if self.layers[l_idx].state_time > dur {
                self.layers[l_idx].state_time = dur;
            }
            self.layers[l_idx].looped_last_step = looped;
            let curr_norm = (self.layers[l_idx].state_time / dur).clamp(0.0, 1.0);

            signals.emit_crossed_signals(
                &state_node.triggers,
                l_idx as u8,
                curr_idx as u8,
                prev_norm,
                curr_norm,
                looped,
            );

            if self.layers[l_idx].target_state.is_none() {
                let t_count = self.layers[l_idx].transition_count.min(MAX_TRANSITIONS);
                for tr_idx in 0..t_count {
                    let tr = self.layers[l_idx].transitions[tr_idx];
                    if tr.active
                        && tr.from_state == self.layers[l_idx].current_state
                        && tr.condition.evaluate(&self.params, curr_norm)
                    {
                        if tr.duration <= 1e-4 {
                            self.layers[l_idx].current_state = tr.to_state;
                            self.layers[l_idx].state_time = 0.0;
                        } else {
                            self.layers[l_idx].target_state = Some(tr.to_state);
                            self.layers[l_idx].target_state_time = 0.0;
                            self.layers[l_idx].transition_elapsed = 0.0;
                            self.layers[l_idx].transition_duration = tr.duration;
                        }
                        break;
                    }
                }
            }

            let active_source =
                self.layers[l_idx].states[self.layers[l_idx].current_state as usize].source;
            self.evaluate_state_source(active_source, clip_poses, rest_pose, &mut layer_pose);

            if let Some(target_idx) = self.layers[l_idx].target_state {
                self.layers[l_idx].transition_elapsed += eff_dt;
                let fade_dur = self.layers[l_idx].transition_duration.max(1e-4);
                let alpha = (self.layers[l_idx].transition_elapsed / fade_dur).clamp(0.0, 1.0);
                let t_source = self.layers[l_idx].states[target_idx as usize].source;
                self.evaluate_state_source(t_source, clip_poses, rest_pose, &mut target_pose);

                if alpha >= 1.0 {
                    self.layers[l_idx].current_state = target_idx;
                    self.layers[l_idx].state_time = self.layers[l_idx].target_state_time;
                    self.layers[l_idx].target_state = None;
                    layer_pose[..joint_count].copy_from_slice(&target_pose[..joint_count]);
                } else {
                    let alpha_f64 = alpha as f64;
                    for j in 0..joint_count {
                        layer_pose[j] = Motor::sclerp(&layer_pose[j], &target_pose[j], alpha_f64);
                    }
                }
            }

            let layer_w = self.layers[l_idx].weight.clamp(0.0, 1.0) as f64;
            let mask = self.layers[l_idx].mask;
            let mode = self.layers[l_idx].blend_mode;

            for j in 0..joint_count {
                if !mask.contains(j) {
                    continue;
                }
                match mode {
                    LayerBlendMode::Override => {
                        out_pose[j] = if layer_w >= 1.0 - 1e-6 {
                            layer_pose[j]
                        } else {
                            Motor::sclerp(&out_pose[j], &layer_pose[j], layer_w)
                        };
                    }
                    LayerBlendMode::Additive => {
                        let delta = rest_pose[j].conjugate().mul(&layer_pose[j]).normalize();
                        let weighted_delta = if layer_w >= 1.0 - 1e-6 {
                            delta
                        } else {
                            Motor::sclerp(&Motor::identity(), &delta, layer_w)
                        };
                        out_pose[j] = out_pose[j].mul(&weighted_delta).normalize();
                    }
                }
            }
        }

        self.eont_posture
            .apply_posture_droop(&mut out_pose[..joint_count]);

        let base_looped = self.layer_count > 0 && self.layers[0].looped_last_step;
        let root_delta = if let Some(ref mut rm) = self.root_motion {
            rm.extract_and_lock(&mut out_pose[..joint_count], base_looped)
        } else {
            RootMotionDelta::default()
        };

        Ok(root_delta)
    }
}
