//! Zero-Heap Animation Blending State Machine Types (`absm_types.rs`).
//!
//! Defines `BoneLayerMask`, `LayerBlendMode`, `AbsmParameter`, `LeafCondition`,
//! `TransitionCondition`, `AnimationSignal`, `SignalTrigger`, `SignalQueue`,
//! `AbsmStateSource`, `AbsmStateNode`, and `AbsmTransition`.

pub use super::gpu_morph::{EntityOntologyMiniHeader, OntogeneticStateHeader};
use super::{Joint, MAX_JOINTS};

/// Maximum parameter slots in an [`super::absm_machine::AbsmMachine`].
pub const MAX_ABSM_PARAMS: usize = 16;

/// Maximum signal triggers per animation state.
pub const MAX_SIGNALS_PER_STATE: usize = 4;

/// 128-bit bone mask (1 bit per joint up to `MAX_JOINTS = 128`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BoneLayerMask(pub u128);

impl Default for BoneLayerMask {
    fn default() -> Self {
        Self::ALL
    }
}

impl BoneLayerMask {
    /// All 128 joints enabled.
    pub const ALL: Self = Self(u128::MAX);
    /// No joints enabled.
    pub const NONE: Self = Self(0);

    /// Builds a `BoneLayerMask` covering `root_joint` and all of its descendants
    /// from a parent index slice (`parent < 0` indicates a root bone).
    pub fn from_joint_subtree(parent_indices: &[i32], root_joint: usize) -> Self {
        let count = parent_indices.len().min(MAX_JOINTS);
        if root_joint >= count {
            return Self::NONE;
        }
        let mut mask = 1u128 << root_joint;
        for _ in 0..count {
            let mut changed = false;
            for j in 0..count {
                if (mask & (1u128 << j)) != 0 {
                    continue;
                }
                let p = parent_indices[j];
                if p >= 0 && (p as usize) < count && (mask & (1u128 << (p as usize))) != 0 {
                    mask |= 1u128 << j;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Self(mask)
    }

    /// Builds a `BoneLayerMask` covering `root_joint` and all of its descendants from `&[Joint]`.
    pub fn from_joints(joints: &[Joint], root_joint: usize) -> Self {
        let count = joints.len().min(MAX_JOINTS);
        if root_joint >= count {
            return Self::NONE;
        }
        let mut mask = 1u128 << root_joint;
        for j in (root_joint + 1)..count {
            if let Some(p) = joints[j].parent {
                if (p as usize) < count && (mask & (1u128 << (p as usize))) != 0 {
                    mask |= 1u128 << j;
                }
            }
        }
        Self(mask)
    }

    /// Returns `true` if `joint_idx` (`< 128`) is enabled in this mask.
    #[inline]
    pub const fn contains(self, joint_idx: usize) -> bool {
        if joint_idx >= MAX_JOINTS {
            return false;
        }
        (self.0 & (1u128 << joint_idx)) != 0
    }

    /// Enables `joint_idx` in this mask.
    #[inline]
    pub fn insert(&mut self, joint_idx: usize) {
        if joint_idx < MAX_JOINTS {
            self.0 |= 1u128 << joint_idx;
        }
    }

    /// Disables `joint_idx` in this mask.
    #[inline]
    pub fn remove(&mut self, joint_idx: usize) {
        if joint_idx < MAX_JOINTS {
            self.0 &= !(1u128 << joint_idx);
        }
    }
}

/// How an ABSM layer combines its output pose with lower layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LayerBlendMode {
    /// Geodesic `Motor::sclerp` override of masked joints by `layer.weight`.
    #[default]
    Override,
    /// Multiplies relative delta motor $M_{\text{base}} \circ (M_{\text{rest}}^{-1} \circ M_{\text{layer}})^w$
    /// onto masked joints.
    Additive,
}

/// Typed parameter value driving ABSM transitions and blend spaces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AbsmParameter {
    Float(f32),
    Bool(bool),
    Int(i32),
    Vec2([f32; 2]),
}

impl Default for AbsmParameter {
    fn default() -> Self {
        Self::Float(0.0)
    }
}

/// Atomic predicate for transition evaluation (stack-resident, `Copy`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LeafCondition {
    Always,
    ParamGt { param_idx: u8, threshold: f32 },
    ParamLt { param_idx: u8, threshold: f32 },
    ParamEqBool { param_idx: u8, expected: bool },
    ParamEqInt { param_idx: u8, expected: i32 },
    StateTimeFinished,
}

impl LeafCondition {
    #[inline]
    pub fn evaluate(self, params: &[AbsmParameter], normalized_state_time: f32) -> bool {
        match self {
            Self::Always => true,
            Self::ParamGt {
                param_idx,
                threshold,
            } => matches!(
                params.get(param_idx as usize),
                Some(AbsmParameter::Float(v)) if *v > threshold
            ),
            Self::ParamLt {
                param_idx,
                threshold,
            } => matches!(
                params.get(param_idx as usize),
                Some(AbsmParameter::Float(v)) if *v < threshold
            ),
            Self::ParamEqBool {
                param_idx,
                expected,
            } => matches!(
                params.get(param_idx as usize),
                Some(AbsmParameter::Bool(v)) if *v == expected
            ),
            Self::ParamEqInt {
                param_idx,
                expected,
            } => matches!(
                params.get(param_idx as usize),
                Some(AbsmParameter::Int(v)) if *v == expected
            ),
            Self::StateTimeFinished => normalized_state_time >= 1.0 - 1e-5,
        }
    }
}

/// Zero-heap transition firing condition supporting single predicates and `And`/`Or` combinators.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TransitionCondition {
    ParamGt { param_idx: u8, threshold: f32 },
    ParamLt { param_idx: u8, threshold: f32 },
    ParamEqBool { param_idx: u8, expected: bool },
    StateTimeFinished,
    And(LeafCondition, LeafCondition),
    Or(LeafCondition, LeafCondition),
}

impl TransitionCondition {
    /// Evaluates this transition condition against `params` and `normalized_state_time`.
    #[inline]
    pub fn evaluate(self, params: &[AbsmParameter], normalized_state_time: f32) -> bool {
        match self {
            Self::ParamGt {
                param_idx,
                threshold,
            } => LeafCondition::ParamGt {
                param_idx,
                threshold,
            }
            .evaluate(params, normalized_state_time),
            Self::ParamLt {
                param_idx,
                threshold,
            } => LeafCondition::ParamLt {
                param_idx,
                threshold,
            }
            .evaluate(params, normalized_state_time),
            Self::ParamEqBool {
                param_idx,
                expected,
            } => LeafCondition::ParamEqBool {
                param_idx,
                expected,
            }
            .evaluate(params, normalized_state_time),
            Self::StateTimeFinished => {
                LeafCondition::StateTimeFinished.evaluate(params, normalized_state_time)
            }
            Self::And(a, b) => {
                a.evaluate(params, normalized_state_time)
                    && b.evaluate(params, normalized_state_time)
            }
            Self::Or(a, b) => {
                a.evaluate(params, normalized_state_time)
                    || b.evaluate(params, normalized_state_time)
            }
        }
    }
}

/// Timeline event emitted when an animation playhead crosses a normalized trigger timestamp.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AnimationSignal {
    pub signal_id: u32,
    pub layer_index: u8,
    pub state_index: u8,
    pub normalized_time: f32,
    pub sonic_archetype: u8,
    pub payload_f32: f32,
}

/// Authored signal trigger attached to an animation state.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SignalTrigger {
    pub active: bool,
    pub signal_id: u32,
    pub normalized_time: f32,
    pub sonic_archetype: u8,
    pub payload_f32: f32,
}

/// Fixed-capacity FIFO ring buffer for [`AnimationSignal`] events (`HotZeroHeap`).
#[derive(Clone, Copy, Debug)]
pub struct SignalQueue<const CAP: usize = 16> {
    buffer: [AnimationSignal; CAP],
    head: usize,
    tail: usize,
    len: usize,
}

impl<const CAP: usize> Default for SignalQueue<CAP> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAP: usize> SignalQueue<CAP> {
    /// Creates an empty signal queue.
    pub const fn new() -> Self {
        Self {
            buffer: [AnimationSignal {
                signal_id: 0,
                layer_index: 0,
                state_index: 0,
                normalized_time: 0.0,
                sonic_archetype: 0,
                payload_f32: 0.0,
            }; CAP],
            head: 0,
            tail: 0,
            len: 0,
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
        self.len = 0;
    }

    /// Pushes a signal into the ring buffer. Returns `false` if full.
    pub fn push(&mut self, sig: AnimationSignal) -> bool {
        if self.len >= CAP || CAP == 0 {
            return false;
        }
        self.buffer[self.tail] = sig;
        self.tail = (self.tail + 1) % CAP;
        self.len += 1;
        true
    }

    /// Pops the oldest signal from the ring buffer.
    pub fn pop(&mut self) -> Option<AnimationSignal> {
        if self.len == 0 || CAP == 0 {
            return None;
        }
        let item = self.buffer[self.head];
        self.head = (self.head + 1) % CAP;
        self.len -= 1;
        Some(item)
    }

    /// Checks `triggers` against the playhead advancement `(prev_norm_t, curr_norm_t]`
    /// (handling loop wrap-around cleanly) and pushes crossed events.
    pub fn emit_crossed_signals(
        &mut self,
        triggers: &[SignalTrigger],
        layer_index: u8,
        state_index: u8,
        prev_norm_t: f32,
        curr_norm_t: f32,
        looped: bool,
    ) -> usize {
        let mut fired = 0usize;
        for trig in triggers {
            if !trig.active {
                continue;
            }
            let t = trig.normalized_time;
            let crossed = if !looped {
                t > prev_norm_t && t <= curr_norm_t
            } else {
                t > prev_norm_t || t <= curr_norm_t
            };
            if crossed
                && self.push(AnimationSignal {
                    signal_id: trig.signal_id,
                    layer_index,
                    state_index,
                    normalized_time: t,
                    sonic_archetype: trig.sonic_archetype,
                    payload_f32: trig.payload_f32,
                })
            {
                fired += 1;
            }
        }
        fired
    }
}

/// Motion generator source for a state inside an [`super::absm_machine::AbsmLayer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AbsmStateSource {
    /// Plays a single clip from the clip pose table.
    SingleClip { clip_index: u16 },
    /// Samples a [`super::blend_space_1d::BlendSpace1D`] driven by `Float` parameter `param_idx`.
    Blend1D { space_index: u8, param_idx: u8 },
    /// Samples a [`super::blend_space_2d::BlendSpace2D`] driven by `Vec2` parameter `param_idx`.
    Blend2D { space_index: u8, param_idx: u8 },
}

impl Default for AbsmStateSource {
    fn default() -> Self {
        Self::SingleClip { clip_index: 0 }
    }
}

/// A state node inside an [`super::absm_machine::AbsmLayer`].
#[derive(Clone, Copy, Debug)]
pub struct AbsmStateNode {
    pub active: bool,
    pub source: AbsmStateSource,
    pub duration: f32,
    pub speed: f32,
    pub looping: bool,
    pub triggers: [SignalTrigger; MAX_SIGNALS_PER_STATE],
}

impl Default for AbsmStateNode {
    fn default() -> Self {
        Self {
            active: false,
            source: AbsmStateSource::default(),
            duration: 1.0,
            speed: 1.0,
            looping: true,
            triggers: [SignalTrigger::default(); MAX_SIGNALS_PER_STATE],
        }
    }
}

/// Directed transition edge between two states in an [`super::absm_machine::AbsmLayer`].
#[derive(Clone, Copy, Debug)]
pub struct AbsmTransition {
    pub active: bool,
    pub from_state: u8,
    pub to_state: u8,
    pub duration: f32,
    pub condition: TransitionCondition,
}

impl Default for AbsmTransition {
    fn default() -> Self {
        Self {
            active: false,
            from_state: 0,
            to_state: 0,
            duration: 0.2,
            condition: TransitionCondition::StateTimeFinished,
        }
    }
}
