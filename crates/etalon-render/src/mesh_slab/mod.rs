//! Unified Multi-Modal GPU Mesh Slab & `.10d` Living Container Residency (`MeshSlabArenaGpu`).
//!
//! Implements Epic 2 of the Superior Game Engine Master Plan:
//! - Zero-heap `SlabRangeAllocator<MAX_RANGES>` sub-allocator for GPU vertex/index buffers.
//! - Multi-modal `ResidentAssetEntry` tracking 4 LOD tiers across B-Rep meshes, Holographic
//!   Proxy-Lens SDFs, `.10d` `MAGIC_SURF` Gaussian surfels, `SGGX` microflake canopies, and
//!   `.10d` `MAGIC_IMPO` hemi-octahedral impostors.
//! - Direct GPU upload of `.10d` `MAGIC_PROC` `ProceduralSynthesisHeader` (`P`,
//!   64 bytes) for GPU-evaluated Voronoi micro-relief, smooth-minimum SDFs, Huxley allometry,
//!   logarithmic spirals, and PGA motor harmonic oscillations.

use bytemuck::{Pod, Zeroable};

/// Representation kind for a `SubmeshSlice` LOD slot (`0..=4`).
pub const REP_KIND_MESH_BREP: u32 = 0;
pub const REP_KIND_HOLOGRAPHIC_SDF_PROXY: u32 = 1;
pub const REP_KIND_SURFEL_CLOUD: u32 = 2;
pub const REP_KIND_SGGX_CANOPY: u32 = 3;
pub const REP_KIND_OCTAHEDRAL_IMPOSTOR: u32 = 4;

/// Sentinel index for an absent `.10d` `PROC1` or `EONT1` header slot.
pub const INVALID_HEADER_SLOT: u16 = 0xFFFF;

/// Maximum resident assets tracked by the zero-heap slab table in portable mode.
pub const MAX_SLAB_RESIDENT_ASSETS: usize = 512;
/// Maximum resident `.10d` `PROC1` synthesis headers in the GPU storage buffer.
pub const MAX_SLAB_PROC_HEADERS: usize = 256;

/// Typed representation archetype for a LOD tier in `ResidentAssetEntry`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssetRepresentationKind {
    MeshBRep = REP_KIND_MESH_BREP,
    HolographicSdfProxy = REP_KIND_HOLOGRAPHIC_SDF_PROXY,
    SurfelCloud = REP_KIND_SURFEL_CLOUD,
    SggxCanopy = REP_KIND_SGGX_CANOPY,
    OctahedralImpostor = REP_KIND_OCTAHEDRAL_IMPOSTOR,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct SubmeshSlice {
    pub first_index: u32,
    pub index_count: u32,
    pub base_vertex: i32,
    /// `0 = MeshBRep`, `1 = HolographicSdfProxy`, `2 = SurfelCloud`, `3 = SggxCanopy`, `4 = OctahedralImpostor`.
    pub rep_kind: u32,
}

const _: [(); 16] = [(); std::mem::size_of::<SubmeshSlice>()];

/// 112-byte multi-modal resident asset entry tracking 4 LOD tiers and `.10d` `PROC`/`EONT`/`SURF`/`IMPO` bindings.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct ResidentAssetEntry {
    pub asset_id: u32,
    /// `.10d` `MAGIC_PROC` `ProceduralSynthesisHeader` slot (`0xFFFF` if none).
    pub proc_header_index: u16,
    /// `.10d` `MAGIC_EONT` `EntityOntologyMiniHeader` slot (`0xFFFF` if none).
    pub eont_header_index: u16,
    /// Offset into the `.10d` `MAGIC_SURF` surfel storage buffer.
    pub surfel_slice_offset: u32,
    /// Number of Gaussian covariance surfels for this asset.
    pub surfel_count: u32,
    /// `.10d` `MAGIC_IMPO` hemi-octahedral impostor atlas layer.
    pub impostor_atlas_layer: u16,
    /// Bit flags (bit 0 = active, bit 1 = casts shadow, bit 2 = skinned).
    pub flags: u16,
    /// Vertex count allocated in `vertex_buffer` for cleanup on `free_asset`.
    pub total_vertex_offset: u32,
    pub total_vertex_count: u32,
    pub reserved_u32: u32,
    /// Local bounding sphere `[center_x, center_y, center_z, radius]`.
    pub bounding_sphere: [f32; 4],
    /// 4-tier LOD slices (`LOD0` near .. `LOD3` far).
    pub lod_slices: [SubmeshSlice; 4],
}

const _: [(); 112] = [(); std::mem::size_of::<ResidentAssetEntry>()];

impl Default for ResidentAssetEntry {
    fn default() -> Self {
        Self {
            asset_id: 0,
            proc_header_index: INVALID_HEADER_SLOT,
            eont_header_index: INVALID_HEADER_SLOT,
            surfel_slice_offset: 0,
            surfel_count: 0,
            impostor_atlas_layer: 0,
            flags: 0,
            total_vertex_offset: 0,
            total_vertex_count: 0,
            reserved_u32: 0,
            bounding_sphere: [0.0, 0.0, 0.0, 1.0],
            lod_slices: [SubmeshSlice::default(); 4],
        }
    }
}

/// 64-byte interleaved vertex record inside `MeshSlabArenaGpu::vertex_buffer`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct MeshSlabVertex {
    pub position: [f32; 3],
    pub uv0_x: f32,
    pub normal: [f32; 3],
    pub uv0_y: f32,
    pub tangent: [f32; 4],
    pub color: [f32; 4],
}

const _: [(); 64] = [(); std::mem::size_of::<MeshSlabVertex>()];

/// 24-byte skeletal skinning vertex attribute record (`JOINTS_0` + `WEIGHTS_0`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct SkinAttrVertex {
    pub joints: [u16; 4],
    pub weights: [f32; 4],
}

const _: [(); 24] = [(); std::mem::size_of::<SkinAttrVertex>()];

/// Capacity configuration for `MeshSlabArenaGpu`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshSlabArenaDescriptor {
    pub max_vertices: u32,
    pub max_indices: u32,
    pub max_assets: u32,
    pub max_proc_headers: u32,
    pub max_indirect_commands: u32,
}

impl Default for MeshSlabArenaDescriptor {
    fn default() -> Self {
        Self {
            max_vertices: 65_536,
            max_indices: 196_608,
            max_assets: MAX_SLAB_RESIDENT_ASSETS as u32,
            max_proc_headers: MAX_SLAB_PROC_HEADERS as u32,
            max_indirect_commands: 2_048,
        }
    }
}

mod range;
pub use range::*;
mod residency;
pub use residency::*;
