//! Webizen-Render core primitives and contracts migrated to `etalon-render`.

pub mod audio_contract;
#[cfg(feature = "gpu")]
pub mod bind_groups;
pub mod buffer_alignment;
pub mod motor_encoder;
pub mod scene;
pub mod scene_contract;
pub mod shaders;
pub mod telemetry;
#[cfg(feature = "gpu")]
pub mod temporal_contract;
pub mod zero_copy_views;

/// PGA motor and 64-byte buffer alignment math module matching `webizen_render::math`.
pub mod math {
    pub use super::buffer_alignment;
    pub use super::buffer_alignment::{AlignedBufferF32, RenderQuin};
    pub use super::motor_encoder;
    pub use super::motor_encoder::{Motor, MotorEncoder};
}

pub use audio_contract::{
    AudioFrameContract, AudioMode, AudioScene, AudioSpectralSheet, AudioTrack,
    GenerativeAudioSheet, PCMAudioSheet, PCMBufferDescriptor, SpectralParams, TensorBufferView,
};
#[cfg(feature = "gpu")]
pub use bind_groups::{BindGroupManager, EpistemicParams, RenderBindGroups};
pub use buffer_alignment::{AlignedBufferF32, RenderQuin};
pub use motor_encoder::{Motor, MotorEncoder};
pub use scene_contract::{
    EpistemicState, RenderScene, SceneCamera, SceneEdge, SceneFace, SceneNode, ScenePoint,
    Tensor10DProjection, TransitionState,
};
pub use shaders::{
    AMBIENT_FRAGMENT_WGSL, AMBIENT_VERTEX_WGSL, AMBIENT_WGSL, EPISTEMIC_WGSL, PROJECTOR_WGSL,
    SCREEN_WGSL,
};
pub use telemetry::SystemTelemetry;
#[cfg(feature = "gpu")]
pub use temporal_contract::{
    admit_capabilities, FrameExtent, HistoryPublication, LinearDepthView, MotionVectorsView,
    ReactiveMaskView, TemporalAdmission, TemporalCapabilityRefusal, TemporalContractError,
    TemporalFrameContract, TemporalHistoryContract, TemporalProducerCapabilities,
    TemporalProducerContract, TemporalResource,
};
pub use zero_copy_views::{
    compute_std140_layout, std140_layout, CameraUniform, ModelUniform, Std140Field, Std140Layout,
    Std140Type, Std140TypeSimple, TimeUniform,
};
#[cfg(target_arch = "wasm32")]
pub use zero_copy_views::{view_f32, view_u32, view_u8, view_uniform_bytes};
