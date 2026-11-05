//! Decoupled volumetric, HUD, and EMF rendering substrate.

#[cfg(feature = "gpu")]
pub mod emf;
pub mod hud;
pub mod scene_contract;
#[cfg(feature = "gpu")]
pub mod volumetric;

#[cfg(feature = "gpu")]
pub use emf::{
    checked_field_bytes, checked_grid_cell_count, EmfFieldCell, EmfPipeline, EmfPipelineDesc,
    EmfSliceUniform, EMF_VOLUMETRIC_WGSL,
};
pub use hud::{
    AlignItems, ComputedUiRect, Edges, FlexDirection, HudVertex, JustifyContent, Margin, Padding,
    SizeRule, UiLayoutNode, UiLayoutStyle, UiLayoutTree, MAX_UI_NODES, NO_UI_NODE,
};
pub use scene_contract::{FrameCameraPod, ManifoldProjectionPod};
#[cfg(feature = "gpu")]
pub use volumetric::{
    CameraUniform, HolographicNodeGpu, ObserverStandpointUniform, VolumetricPipeline,
    HOLOGRAPHIC_WGSL,
};

#[cfg(feature = "gpu")]
pub mod mesh_normals;
#[cfg(feature = "gpu")]
pub mod shadow_math;
#[cfg(feature = "gpu")]
pub mod targets;
#[cfg(feature = "gpu")]
pub mod texture_mips;

#[cfg(feature = "gpu")]
pub mod holographic;
#[cfg(feature = "gpu")]
pub mod picking;
#[cfg(feature = "gpu")]
pub mod texture_layout;

#[cfg(feature = "gpu")]
pub mod temporal;
#[cfg(feature = "gpu")]
pub mod temporal_producers;

#[cfg(feature = "gpu")]
pub mod bloom;
#[cfg(feature = "gpu")]
pub mod output_pass;

#[cfg(feature = "gpu")]
pub mod ao;
#[cfg(feature = "gpu")]
pub mod shadows;

#[cfg(feature = "gpu")]
pub mod picking_readback;

#[cfg(feature = "gpu")]
pub mod readback;

#[cfg(feature = "gpu")]
pub mod compute_readback;

#[cfg(feature = "gpu")]
pub mod indirect_cull;
pub mod instance_culling;
#[cfg(feature = "gpu")]
pub mod instance_stream;
#[cfg(feature = "gpu")]
pub mod mesh_slab;

#[cfg(feature = "gpu")]
pub mod materials;

#[cfg(feature = "gpu")]
pub mod mesh;

#[cfg(feature = "gpu")]
pub mod texture_residency;

#[cfg(feature = "gpu")]
pub mod lighting;

#[cfg(feature = "gpu")]
pub mod texture_precomputed;

pub mod control;
pub mod frame_lock;
pub mod frame_quality;
pub mod geo_viewport;
pub mod ibl_cubemap;
pub mod naga_webgl2;
pub mod shaders;
pub mod spatial_math;
pub mod spectral;
pub mod surface_frames;
pub mod texture_codec;
pub mod ui_canvas;
pub mod vegetation;
pub mod water_surface;
pub mod webizen_core;
