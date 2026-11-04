//! GPU-Driven Multi-Draw Indirect Culling, 2-Phase HZB Occlusion & 4-Tier Hybrid LOD.
//!
//! Pairs with [`super::mesh_slab::MeshSlabArenaGpu`] and `INSTANCE_CULL_INDIRECT_WGSL` to
//! evaluate per-instance PGA motors, 6-plane frustum culling, 2-phase Hierarchical Z-Buffer
//! (HZB) occlusion culling, continuous screen-space projected radius LOD selection (`LOD0..LOD3`)
//! with hysteresis and epistemic salience weighting, and stream compaction into:
//! - **Stream 0 (Mesh Indirect Stream)**: `DrawIndexedIndirectCommand` for `LOD0..LOD2` (`q == 0`)
//! - **Stream 1 (Holographic SDF Proxy Stream)**: instance indices with `q > 0` or `HolographicSdfProxy`
//! - **Stream 2 (Surfel / Octahedral Impostor Stream)**: instance indices at `LOD3` or `Surfel`/`Impostor`

use bytemuck::{Pod, Zeroable};
use etalon as wgpu;

use super::mesh_slab::{ResidentAssetEntry, SubmeshSlice};
use crate::instance_culling::{extract_frustum_planes, FrustumPlane};

/// Default screen-space projected radius thresholds in pixels (`[LOD0, LOD1, LOD2]`).
/// - `LOD0`: `>= 128.0` px (full procedural/CAD/skinned mesh)
/// - `LOD1`: `[48.0, 128.0)` px (simplified mesh, ~40% triangles)
/// - `LOD2`: `[16.0, 48.0)` px (coarse mesh, ~12% triangles)
/// - `LOD3`: `< 16.0` px (Gaussian Surfel Cloud / Octahedral Impostor proxy)
pub const DEFAULT_LOD_THRESHOLDS_PX: [f32; 3] = [128.0, 48.0, 16.0];

/// Default minimum projected radius in pixels below which an instance is sub-pixel culled.
pub const DEFAULT_MIN_CULL_RADIUS_PX: f32 = 1.5;

/// Default fractional hysteresis deadband around LOD thresholds (`10%`).
pub const DEFAULT_LOD_HYSTERESIS: f32 = 0.10;

/// Multi-modal representation policy encoded in `InstanceCullDescriptor::rep_policy`.
#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepresentationPolicy {
    /// Automatic hybrid transitions based on screen-space radius and tensor valence `(q, v)`.
    Auto = 0,
    /// Force explicit triangle mesh LOD (`LOD0..LOD2`).
    ExplicitMeshLod = 1,
    /// Route directly to the depth-interleaved Holographic CSG/SDF Proxy stream.
    HolographicSdfProxy = 2,
    /// Route directly to the Gaussian Surfel Cloud stream.
    GaussianSurfelCloud = 3,
    /// Route directly to the Octahedral Impostor stream.
    OctahedralImpostor = 4,
}

impl RepresentationPolicy {
    #[inline]
    pub const fn from_u16(v: u16) -> Self {
        match v {
            1 => Self::ExplicitMeshLod,
            2 => Self::HolographicSdfProxy,
            3 => Self::GaussianSurfelCloud,
            4 => Self::OctahedralImpostor,
            _ => Self::Auto,
        }
    }
}

/// WebGPU-standard 20-byte `DrawIndexedIndirect` command payload.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

const _: [(); 20] = [(); std::mem::size_of::<DrawIndexedIndirectCommand>()];

/// GPU-aligned per-instance culling & multi-modal routing descriptor (80 bytes, `5 x vec4`).
///
/// Matches `InstanceCullDescriptorGpu` in `instance_cull_indirect.wgsl` byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct InstanceCullDescriptor {
    /// PGA unit motor rotor part $R = (e_{023}, e_{031}, e_{012}, 1)$.
    pub r: [f32; 4],
    /// PGA unit motor dual part $D = (e_{01}, e_{02}, e_{03}, e_{0123})$.
    pub d: [f32; 4],
    /// `xyz` = non-uniform instance scale, `w` = local bounding sphere radius.
    pub scale_radius: [f32; 4],
    /// Index into the resident `ResidentAssetEntry` table in `MeshSlabArenaGpu`.
    pub asset_id: u32,
    /// Material slot identifier (`0..65535`).
    pub material_id: u16,
    /// Multi-modal representation policy (`RepresentationPolicy as u16`).
    pub rep_policy: u16,
    /// Base joint offset in `skin_motors_buf` (`0` if unrigged).
    pub rig_offset: u32,
    /// Index into `proc_headers_buf` (`0` if non-procedural).
    pub proc_header_idx: u16,
    /// Packed `.10d` tensor valence: low byte = `q` (quantum/field degree), high byte = `v`.
    pub tensor_q_v: u16,
    /// Epistemic salience $\alpha \in [0.05, 4.0]$ modulating screen-space LOD priority.
    pub alpha_salience: f32,
    /// Spectral uncertainty $\sigma \ge 0.0$ modulating coarse LOD fallback.
    pub sigma_spectral: f32,
    /// Previous frame's LOD tier (`0.0..=3.0`) used for hysteresis stability.
    pub prev_lod_tier: f32,
    pub _pad: f32,
}

const _: [(); 80] = [(); std::mem::size_of::<InstanceCullDescriptor>()];

impl Default for InstanceCullDescriptor {
    fn default() -> Self {
        Self {
            r: [0.0, 0.0, 0.0, 1.0],
            d: [0.0, 0.0, 0.0, 0.0],
            scale_radius: [1.0, 1.0, 1.0, 1.0],
            asset_id: 0,
            material_id: 0,
            rep_policy: RepresentationPolicy::Auto as u16,
            rig_offset: 0,
            proc_header_idx: 0,
            tensor_q_v: 0,
            alpha_salience: 1.0,
            sigma_spectral: 0.0,
            prev_lod_tier: 0.0,
            _pad: 0.0,
        }
    }
}

impl InstanceCullDescriptor {
    /// Packs `(q, v)` tensor valence into `tensor_q_v`.
    #[inline]
    pub const fn pack_tensor_q_v(q: u8, v: u8) -> u16 {
        (q as u16) | ((v as u16) << 8)
    }

    /// Unpacks the quantum/holographic field degree `q` from `tensor_q_v`.
    #[inline]
    pub const fn tensor_q(&self) -> u8 {
        (self.tensor_q_v & 0xFF) as u8
    }

    /// Unpacks the contravariant/covariant valence `v` from `tensor_q_v`.
    #[inline]
    pub const fn tensor_v(&self) -> u8 {
        ((self.tensor_q_v >> 8) & 0xFF) as u8
    }

    /// Evaluates the world-space center of this instance's bounding sphere from its PGA motor $(R, D)$:
    /// $$\mathbf{p}_{\text{world}} = 2 (r_w \mathbf{d}_v - d_w \mathbf{r}_v + \mathbf{d}_v \times \mathbf{r}_v)$$
    #[inline]
    pub fn world_center(&self) -> [f32; 3] {
        let rv = [self.r[0], self.r[1], self.r[2]];
        let rw = self.r[3];
        let dv = [self.d[0], self.d[1], self.d[2]];
        let dw = self.d[3];
        let cross = [
            dv[1] * rv[2] - dv[2] * rv[1],
            dv[2] * rv[0] - dv[0] * rv[2],
            dv[0] * rv[1] - dv[1] * rv[0],
        ];
        [
            2.0 * (rw * dv[0] - dw * rv[0] + cross[0]),
            2.0 * (rw * dv[1] - dw * rv[1] + cross[1]),
            2.0 * (rw * dv[2] - dw * rv[2] + cross[2]),
        ]
    }

    /// Builds a pure-translation PGA motor from a world-space position `pos`.
    /// For translation by $\mathbf{t}$ with identity rotation $R = (0, 0, 0, 1)$,
    /// $D = \frac{1}{2}(t_x, t_y, t_z, 0)$.
    #[inline]
    pub fn with_translation(mut self, pos: [f32; 3], radius: f32) -> Self {
        self.r = [0.0, 0.0, 0.0, 1.0];
        self.d = [0.5 * pos[0], 0.5 * pos[1], 0.5 * pos[2], 0.0];
        self.scale_radius[3] = radius;
        self
    }

    /// Returns the world-space bounding sphere radius scaled by the maximum axis scale.
    #[inline]
    pub fn world_radius(&self) -> f32 {
        let max_s = self.scale_radius[0]
            .max(self.scale_radius[1])
            .max(self.scale_radius[2]);
        (self.scale_radius[3] * max_s).max(0.001)
    }
}

/// Uniform buffer layout for `instance_cull_indirect.wgsl` (288 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CullUniformsGpu {
    pub view_proj: [[f32; 4]; 4],
    pub view_matrix: [[f32; 4]; 4],
    pub frustum_planes: [[f32; 4]; 6],
    /// `xyz` = camera world position, `w` = `cot(fov_y * 0.5) * 0.5 * viewport_height`.
    pub camera_pos_and_focal_px: [f32; 4],
    /// `xy` = viewport `(w, h)`, `zw` = HZB mip 0 `(w, h)`.
    pub viewport_and_hzb_dims: [f32; 4],
    /// `x` = LOD0 (128px), `y` = LOD1 (48px), `z` = LOD2 (16px), `w` = min cull px (1.5px).
    pub lod_thresholds_px: [f32; 4],
    /// `x` = instance_count, `y` = hzb_mip_count, `z` = enable_hzb (0/1), `w` = reserved.
    pub counts_and_flags: [u32; 4],
}

const _: [(); 288] = [(); std::mem::size_of::<CullUniformsGpu>()];

/// Output stream counts produced by [`cull_and_select_lod_indirect_zero_heap`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct IndirectCullStreamCounts {
    /// Number of `DrawIndexedIndirectCommand` entries written to the Mesh Indirect Stream (`LOD0..LOD2`).
    pub mesh_draw_count: u32,
    /// Number of instance indices routed to the Holographic SDF Proxy Stream (`q > 0` or `HolographicSdfProxy`).
    pub holo_proxy_count: u32,
    /// Number of instance indices routed to the Surfel / Octahedral Impostor Stream (`LOD3` or `Surfel`/`Impostor`).
    pub surfel_impostor_count: u32,
    /// Number of frustum-visible instances rejected by the 2-phase HZB depth pyramid.
    pub occluded_count: u32,
}

/// Lightweight zero-heap view over a 2-phase Hierarchical Z-Buffer (HZB) mip chain.
///
/// Each mip stores conservative **farthest** depth values in `[0.0, 1.0]` (`max` reduction
/// for standard `[0, 1]` depth where `0.0` is near and `1.0` is far).
#[derive(Clone, Copy, Debug)]
pub struct HzbPyramidView<'a> {
    /// Base mip 0 dimensions `(width, height)`.
    pub base_width: u32,
    pub base_height: u32,
    /// Packed mip texels in row-major order starting from mip 0.
    pub mip_texels: &'a [f32],
    /// `(offset, width, height)` for up to 8 mip levels.
    pub mip_slices: [(usize, u32, u32); 8],
    pub mip_count: usize,
}

impl<'a> HzbPyramidView<'a> {
    /// Samples the conservative farthest depth at `(uv_x, uv_y)` in `[0, 1]^2` at `mip_level`.
    #[inline]
    pub fn sample_farthest_depth(&self, uv: [f32; 2], mip_level: usize) -> f32 {
        if self.mip_count == 0 || self.mip_texels.is_empty() {
            return 1.0;
        }
        let mip = mip_level.min(self.mip_count - 1);
        let (offset, w, h) = self.mip_slices[mip];
        if w == 0 || h == 0 {
            return 1.0;
        }
        let tx = ((uv[0].clamp(0.0, 0.999_999) * w as f32) as u32).min(w - 1);
        let ty = ((uv[1].clamp(0.0, 0.999_999) * h as f32) as u32).min(h - 1);
        let idx = offset + (ty * w + tx) as usize;
        self.mip_texels.get(idx).copied().unwrap_or(1.0)
    }
}

/// Builds an HZB mip chain (`max`-depth reduction) from `mip0` into `out_texels` with zero heap allocation.
///
mod selection;
pub use selection::*;
mod pipeline;
pub use pipeline::*;
#[cfg(test)]
mod tests;
