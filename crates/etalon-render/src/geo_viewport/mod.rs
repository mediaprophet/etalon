// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 QualiaDB & Webizen Contributors (adapted from EarthCraft v0.2.0, Nerthus v2.3.1, and RustGIS v0.4.0)
//! Planetary-to-Millimeter Geospatial Viewport (`geo_viewport`):
//! Capped-Angle Polyline & Earcut Tessellation, CDLOD Quadtree & Terrain Skirts,
//! MVT Style & Curved River/Road Labels, Screen-Space Collision Solver,
//! Rayleigh/Mie Spherical Atmosphere Limb, Gerstner Water Waves, Hypsometric Contours,
//! GPU Picking, and Ambient Vegetation/Particle Effects.

pub mod curved_labels;
pub mod sky_limb;
pub mod tessellation;
pub mod vegetation_effects;

/// Module aliases preserving backwards-compatible `webizen-render` submodule paths.
pub use self::curved_labels as mvt_pmtiles_curved_labels;
pub use self::sky_limb as sky_atmosphere_limb;
pub use self::tessellation as geo_tessellation_capped;

pub use curved_labels::{
    evaluate_style_zoom_interpolated, layout_glyphs_along_centerline,
    solve_screen_space_label_collisions, CurvedGlyphQuad, GeoLabelError, LabelCandidate,
    PlacedLabel, RTreeNode, RTreeScreenIndex, StyleZoomStop, VectorTileLayerStyle, OFFSET_B,
    OFFSET_BL, OFFSET_BR, OFFSET_CENTER, OFFSET_L, OFFSET_R, OFFSET_T, OFFSET_TL, OFFSET_TR,
};
pub use sky_limb::{
    evaluate_sky_limb_and_aerial_perspective, evaluate_water_wave_and_foam,
    resolve_gpu_picking_id_aperture, shade_hypsometric_and_contours_into, AtmosphereParams,
    GeoAtmosphereError, PickingHitRecord, WaterSurfaceSample,
};
pub use tessellation::{
    build_terrain_patch_with_skirts, select_cdlod_quadtree_nodes, tessellate_polygon_earcut,
    tessellate_polyline_stroke_capped, CdlodNodeSpec, GeoRenderVertex, GeoTessError,
    StrokeJoinCapStyle,
};
pub use vegetation_effects::{
    select_mode, stable_seed, EffectEvent, EffectPool, EffectsMode, GpuParticle, SpawnOutcome,
    SystemTelemetry, WindField, WorldFrame, DEFAULT_WORLD_SEED, MAX_AMBIENT_PARTICLES,
    MAX_EFFECT_EVENTS,
};
