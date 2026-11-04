//! Optional, renderer-owned GPU temporal resolve.
//!
//! This module owns the resolve, current linear-depth, and history targets. It
//! produces linear depth from the renderer's authoritative depth attachment,
//! but it does not invent motion-vector or reactive-mask data. A frame can
//! enter temporal accumulation only when those remaining real producer views
//! are supplied by the caller.

use std::num::NonZeroU64;

const DEPTH_HISTORY_FORMAT: etalon::TextureFormat = etalon::TextureFormat::Rgba16Float;
const DEPTH_DISOCCLUSION_THRESHOLD: f32 = 0.01;
const MAX_MOTION_FOR_HISTORY: f32 = 2.0;
const MAX_HISTORY_WEIGHT: f32 = 0.9;
const PARAMS_BYTES: u64 = 32;
const OUTPUT_PARAMS_BYTES: u64 = 32;
const DEPTH_PRODUCER_PARAMS_BYTES: u64 = 16;
const LINEAR_DEPTH_FORMAT: etalon::TextureFormat = etalon::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TemporalParams {
    viewport: [f32; 4],
    controls: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct OutputParams {
    exposure: f32,
    surface_is_srgb: f32,
    _padding0: [f32; 2],
    white_balance_gains: [f32; 3],
    _padding1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DepthProducerParams {
    near_plane: f32,
    far_plane: f32,
    inv_width: f32,
    inv_height: f32,
}

/// Real producer views required to admit a temporal resolve.
///
/// `motion_vectors` are UV-space vectors pointing from the current pixel to its
/// previous-frame location. `reactive_mask.r` is normalized to `[0, 1]`, where
/// one rejects history. All views must have the helper's extent and one sample.
pub struct TemporalResolveInputs<'scene, 'producer> {
    pub extent: (u32, u32),
    pub current_scene: &'scene etalon::TextureView,
    pub current_linear_depth: &'producer etalon::TextureView,
    pub motion_vectors: &'producer etalon::TextureView,
    pub reactive_mask: &'producer etalon::TextureView,
}

/// Producer views supplied by a host for one temporal frame.
///
/// Scene colour and linear depth are owned by `PortalGpu` by default; a host may still provide a
/// compatible linear-depth view. Motion vectors and reactive masks remain explicit host producer
/// views. Omitting either of those views keeps temporal output disabled.
#[derive(Clone, Copy, Default)]
pub struct TemporalProducerViews<'a> {
    pub extent: (u32, u32),
    pub current_linear_depth: Option<&'a etalon::TextureView>,
    pub motion_vectors: Option<&'a etalon::TextureView>,
    pub reactive_mask: Option<&'a etalon::TextureView>,
}

/// Explicit availability report for the renderer/host temporal handoff.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TemporalProducerAvailability {
    pub scene_color: bool,
    pub linear_depth: bool,
    pub motion_vectors: bool,
    pub reactive_mask: bool,
}

impl TemporalProducerAvailability {
    /// `TemporalProducerAvailability::host_producers_available`.
    pub const fn host_producers_available(self) -> bool {
        self.motion_vectors && self.reactive_mask
    }

    /// `TemporalProducerAvailability::all_available`.
    pub const fn all_available(self) -> bool {
        self.scene_color && self.linear_depth && self.host_producers_available()
    }
}

impl<'a> TemporalProducerViews<'a> {
    /// `TemporalProducerViews::availability`.
    pub const fn availability(self, scene_color: bool) -> TemporalProducerAvailability {
        TemporalProducerAvailability {
            scene_color,
            linear_depth: self.current_linear_depth.is_some(),
            motion_vectors: self.motion_vectors.is_some(),
            reactive_mask: self.reactive_mask.is_some(),
        }
    }

    /// Whether the host has supplied both producer attachments that the renderer cannot derive.
    pub const fn has_required_host_producers(self) -> bool {
        self.availability(false).host_producers_available()
    }
}

/// Command-recording lifecycle for one renderer-owned temporal submission.
///
/// This is deliberately a state machine rather than a collection of inferred booleans. It makes
/// the ordering visible to the renderer adapter: a producer pass may be recorded first, resolve
/// must follow it, history publication is a separate step, and the final output transform is the
/// last step. The state describes recorded work in the current command encoder; history validity
/// remains separately tracked because it spans frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TemporalSubmissionState {
    #[default]
    Idle,
    LinearDepthReady,
    Resolved,
    HistoryPublished,
    OutputReady,
}
/// GPU temporal resolve and explicit history publication owner.
pub struct TemporalResolveGpu<R> {
    bind_layout: etalon::BindGroupLayout,
    depth_producer_layout: etalon::BindGroupLayout,
    publish_depth_layout: etalon::BindGroupLayout,
    output_layout: etalon::BindGroupLayout,
    resolve_pipeline: etalon::RenderPipeline,
    depth_producer_pipeline: etalon::RenderPipeline,
    publish_depth_pipeline: etalon::RenderPipeline,
    output_pipeline: etalon::RenderPipeline,
    params_buf: etalon::Buffer,
    depth_producer_params_buf: etalon::Buffer,
    depth_sampler: etalon::Sampler,
    output_params_buf: etalon::Buffer,
    history_texture: etalon::Texture,
    history_view: etalon::TextureView,
    resolved_texture: etalon::Texture,
    resolved_view: etalon::TextureView,
    _linear_depth_texture: etalon::Texture,
    linear_depth_view: etalon::TextureView,
    _history_depth_texture: etalon::Texture,
    history_depth_view: etalon::TextureView,
    width: u32,
    height: u32,
    history_valid: bool,
    resolved_ready: bool,
    submission_state: TemporalSubmissionState,
    _reservation: R,
    exposure_scale: fn(f32) -> Option<f32>,
}

pub const TEMPORAL_RESOLVE_WGSL: &str = include_str!("../shaders/temporal_resolve.wgsl");
pub const TEMPORAL_DEPTH_WGSL: &str = include_str!("../shaders/temporal_depth.wgsl");
pub const TEMPORAL_MOTION_WGSL: &str = include_str!("../shaders/temporal_motion.wgsl");
pub const TEMPORAL_REACTIVE_WGSL: &str = include_str!("../shaders/temporal_reactive.wgsl");
pub const TEMPORAL_OUTPUT_WGSL: &str = concat!(
    include_str!("../shaders/output_transform.wgsl"),
    include_str!("../shaders/temporal_output.wgsl")
);

/// Ordered temporal handoff selected while compiling a frame graph.
///
/// `TemporalResolveGpu` consumes this contract rather than guessing whether a host has supplied
/// the required motion-vector and reactive-mask views. Linear depth is produced by the renderer
/// from its authoritative scene-depth attachment. A disabled contract must leave the existing
/// output path untouched.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TemporalOutputSchedule {
    pub enabled: bool,
    pub reset_history: bool,
    pub reads_history: bool,
    pub publish_history: bool,
    pub final_output: bool,
}

impl TemporalOutputSchedule {
    /// `TemporalOutputSchedule::disabled`.
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            reset_history: false,
            reads_history: false,
            publish_history: false,
            final_output: false,
        }
    }

    /// `TemporalOutputSchedule::fail_closed`.
    pub const fn fail_closed() -> Self {
        Self::disabled()
    }

    /// Whether a backend may record the complete resolve/publication/output handoff.
    ///
    /// Reset frames never read history, and a temporal schedule must publish before its final
    /// output. The pass-order part is checked by `CompiledSchedule::temporal_order_is_valid`.
    pub const fn is_valid(self) -> bool {
        self.enabled
            && self.publish_history
            && self.final_output
            && !(self.reset_history && self.reads_history)
    }

    /// `true` when complete.
    pub const fn is_complete(self) -> bool {
        self.is_valid()
    }

    /// Builder: `reset-history`.
    pub const fn with_reset_history(mut self) -> Self {
        self.reset_history = true;
        self.reads_history = false;
        self
    }
}

mod construct;
mod execute;
mod helpers;
pub use helpers::target_bytes;
use helpers::*;
