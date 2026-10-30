//! Decoupled volumetric, HUD, and EMF rendering substrate.

pub mod emf;
pub mod hud;
pub mod scene_contract;
pub mod volumetric;

pub use emf::{
    checked_field_bytes, checked_grid_cell_count, EmfFieldCell, EmfPipeline, EmfPipelineDesc,
    EmfSliceUniform, EMF_VOLUMETRIC_WGSL,
};
pub use hud::{
    AlignItems, ComputedUiRect, Edges, FlexDirection, HudVertex, JustifyContent, Margin, Padding,
    SizeRule, UiLayoutNode, UiLayoutStyle, UiLayoutTree, MAX_UI_NODES, NO_UI_NODE,
};
pub use scene_contract::{FrameCameraPod, ManifoldProjectionPod};
pub use volumetric::{
    CameraUniform, HolographicNodeGpu, ObserverStandpointUniform, VolumetricPipeline,
    HOLOGRAPHIC_WGSL,
};
