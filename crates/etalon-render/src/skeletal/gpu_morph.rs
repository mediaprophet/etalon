//! GPU Morph Target (Blend Shape) & Ontogenetic Deformation Bridge (`gpu_morph.rs`).
//!
//! Packs morph target position and normal deltas into 32-byte `GpuMorphDelta` records
//! and combines explicit facial/pose blend-shape weights with `OntogeneticStateHeader`
//! (`EntityOntologyMiniHeader`) metabolic state (`damage_permill`, `hydration_permill`, `stage`)
//! inside a 48-byte `GpuInstanceMorphWeights` uniform/storage record.
//!
//! Honors `AGENTS.md` §0 & §0-A: all evaluation functions are `HotZeroHeap`.

use super::SkeletalError;

/// Domain-neutral 40-byte ontogenetic and metabolic state header (`EONT1` layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OntogeneticStateHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub stage: u8,
    pub reserved_u8: u8,
    pub entity_did_hash: u64,
    pub taxon_iri_hash: u64,
    pub energy_permill: u16,
    pub damage_permill: u16,
    pub hydration_permill: u16,
    pub wear_permill: u16,
    pub affordance_count: u32,
    pub reserved_u32: u32,
}

const _: [(); 40] = [(); core::mem::size_of::<OntogeneticStateHeader>()];

/// Backwards-compatible alias for `OntogeneticStateHeader`.
pub type EntityOntologyMiniHeader = OntogeneticStateHeader;

/// Maximum morph targets evaluated simultaneously per instance on the GPU.
pub const MAX_GPU_MORPH_TARGETS: usize = 8;

/// 32-byte GPU storage record holding a single morph target's per-vertex position and normal delta.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMorphDelta {
    /// Position displacement `[dx, dy, dz]` in authored bind space.
    pub pos_delta: [f32; 3],
    pub _pad0: f32,
    /// Surface normal displacement `[dnx, dny, dnz]` in authored bind space.
    pub normal_delta: [f32; 3],
    pub _pad1: f32,
}

const _: [(); 32] = [(); core::mem::size_of::<GpuMorphDelta>()];

impl GpuMorphDelta {
    /// Creates a new `GpuMorphDelta` from position and normal delta vectors.
    #[inline]
    pub const fn new(pos_delta: [f32; 3], normal_delta: [f32; 3]) -> Self {
        Self {
            pos_delta,
            _pad0: 0.0,
            normal_delta,
            _pad1: 0.0,
        }
    }
}

/// 48-byte per-instance morph weight uniform/storage record uploaded to the GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuInstanceMorphWeights {
    /// Normalized morph target weights (`0..8`).
    pub weights: [f32; MAX_GPU_MORPH_TARGETS],
    /// Number of active morph targets (`<= 8`).
    pub target_count: u32,
    /// Vertex count stride between consecutive morph target slabs in the delta buffer.
    pub vertex_stride: u32,
    /// Structural/tissue damage in permill (`0..=1000`).
    pub eont_damage_permill: u32,
    /// Hydration / turgor pressure in permill (`0..=1000`).
    pub eont_hydration_permill: u32,
}

const _: [(); 48] = [(); core::mem::size_of::<GpuInstanceMorphWeights>()];

/// Maps a `DevelopmentalStage` discriminant (`0..=4`) to a normalized
/// ontogenetic growth factor in `[0.0, 1.0]`.
#[inline]
pub fn ontogenetic_stage_growth_factor(stage: u8) -> f32 {
    match stage {
        0 => 0.10, // SeedOrSpore
        1 => 0.50, // Juvenile
        2 => 1.00, // Adult
        3 => 0.85, // Senescent
        _ => 0.60, // Deceased / withered
    }
}

/// Computes `GpuInstanceMorphWeights` with zero heap allocations (`HotZeroHeap`),
/// combining explicit pose/facial blend-shape weights with ontogenetic/metabolic state
/// (`damage_permill`, `hydration_permill`, and `stage` ontogenetic factor).
pub fn compute_ontogenetic_morph_weights_zero_heap(
    explicit_weights: &[f32],
    target_count: u32,
    vertex_stride: u32,
    stage: u8,
    damage_permill: u16,
    hydration_permill: u16,
) -> GpuInstanceMorphWeights {
    let mut weights = [0.0f32; MAX_GPU_MORPH_TARGETS];
    let copy_len = explicit_weights.len().min(MAX_GPU_MORPH_TARGETS);
    for i in 0..copy_len {
        let w = explicit_weights[i];
        weights[i] = if w.is_finite() {
            w.clamp(0.0, 1.0)
        } else {
            0.0
        };
    }

    let clamped_damage = damage_permill.min(1000);
    let clamped_hydration = hydration_permill.min(1000);

    let growth_factor = ontogenetic_stage_growth_factor(stage);
    let damage_weight = (clamped_damage as f32) * 0.001;
    let wilt_weight = 1.0 - (clamped_hydration as f32) * 0.001;

    weights[5] = weights[5].max(growth_factor);
    weights[6] = weights[6].max(damage_weight);
    weights[7] = weights[7].max(wilt_weight);

    GpuInstanceMorphWeights {
        weights,
        target_count: target_count.min(MAX_GPU_MORPH_TARGETS as u32),
        vertex_stride,
        eont_damage_permill: clamped_damage as u32,
        eont_hydration_permill: clamped_hydration as u32,
    }
}

/// Convenience wrapper extracting `stage`, `damage_permill`, and `hydration_permill`
/// directly from an `OntogeneticStateHeader` (`EntityOntologyMiniHeader`).
#[inline]
pub fn compute_ontogenetic_morph_weights_from_header_zero_heap(
    explicit_weights: &[f32],
    target_count: u32,
    vertex_stride: u32,
    header: &OntogeneticStateHeader,
) -> GpuInstanceMorphWeights {
    compute_ontogenetic_morph_weights_zero_heap(
        explicit_weights,
        target_count,
        vertex_stride,
        header.stage,
        header.damage_permill,
        header.hydration_permill,
    )
}

/// Evaluates linear blend-shape morph deltas into caller-supplied vertex position and normal
/// buffers with zero heap allocations (`HotZeroHeap`).
pub fn evaluate_morph_deltas_zero_heap(
    base_positions: &[[f32; 3]],
    base_normals: &[[f32; 3]],
    morph_deltas: &[GpuMorphDelta],
    instance_weights: &GpuInstanceMorphWeights,
    out_positions: &mut [[f32; 3]],
    out_normals: &mut [[f32; 3]],
) -> Result<usize, SkeletalError> {
    let vertex_count = base_positions.len();
    if vertex_count == 0 || base_normals.len() != vertex_count {
        return Err(SkeletalError::InvalidInput);
    }
    if out_positions.len() < vertex_count || out_normals.len() < vertex_count {
        return Err(SkeletalError::OutputTooSmall);
    }

    let target_count = (instance_weights.target_count as usize).min(MAX_GPU_MORPH_TARGETS);
    let stride = instance_weights.vertex_stride as usize;
    if target_count > 0 {
        if stride < vertex_count {
            return Err(SkeletalError::InvalidInput);
        }
        let needed_deltas = (target_count - 1)
            .checked_mul(stride)
            .and_then(|base| base.checked_add(vertex_count))
            .ok_or(SkeletalError::InvalidInput)?;
        if morph_deltas.len() < needed_deltas {
            return Err(SkeletalError::InvalidInput);
        }
    }

    for v in 0..vertex_count {
        let mut px = base_positions[v][0];
        let mut py = base_positions[v][1];
        let mut pz = base_positions[v][2];

        let mut nx = base_normals[v][0];
        let mut ny = base_normals[v][1];
        let mut nz = base_normals[v][2];

        for t in 0..target_count {
            let w = instance_weights.weights[t];
            if w.abs() <= 1e-7 || !w.is_finite() {
                continue;
            }
            let delta = &morph_deltas[t * stride + v];
            px += w * delta.pos_delta[0];
            py += w * delta.pos_delta[1];
            pz += w * delta.pos_delta[2];

            nx += w * delta.normal_delta[0];
            ny += w * delta.normal_delta[1];
            nz += w * delta.normal_delta[2];
        }

        out_positions[v] = [px, py, pz];

        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        out_normals[v] = if len > 1e-6 {
            [nx / len, ny / len, nz / len]
        } else {
            base_normals[v]
        };
    }

    Ok(vertex_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeletal::assert_zero_alloc;

    #[test]
    fn ontogenetic_morph_weights_and_delta_evaluation() {
        let base_pos = [[1.0f32, 2.0, 3.0], [0.0, 1.0, 0.0]];
        let base_nrm = [[0.0f32, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let mut deltas = [GpuMorphDelta::default(); 16];
        deltas[0 * 2 + 0] = GpuMorphDelta::new([0.0, 0.5, 0.0], [0.0, 0.0, 0.0]);
        deltas[6 * 2 + 0] = GpuMorphDelta::new([-0.2, 0.0, 0.0], [0.5, 0.0, 0.0]);
        deltas[7 * 2 + 1] = GpuMorphDelta::new([0.0, -0.4, 0.0], [0.0, 0.0, 0.0]);

        let weights = compute_ontogenetic_morph_weights_zero_heap(&[0.5], 8, 2, 2, 500, 250);
        assert!((weights.weights[0] - 0.5).abs() < 1e-6);
        assert!((weights.weights[5] - 1.0).abs() < 1e-6);
        assert!((weights.weights[6] - 0.5).abs() < 1e-6);
        assert!((weights.weights[7] - 0.75).abs() < 1e-6);
        assert_eq!(weights.eont_damage_permill, 500);
        assert_eq!(weights.eont_hydration_permill, 250);

        let mut out_pos = [[0.0f32; 3]; 2];
        let mut out_nrm = [[0.0f32; 3]; 2];

        assert_zero_alloc("evaluate_morph_deltas", || {
            let count = evaluate_morph_deltas_zero_heap(
                &base_pos,
                &base_nrm,
                &deltas,
                &weights,
                &mut out_pos,
                &mut out_nrm,
            )
            .unwrap();
            assert_eq!(count, 2);
        });

        assert!((out_pos[0][0] - 0.9).abs() < 1e-5);
        assert!((out_pos[0][1] - 2.25).abs() < 1e-5);
        assert!((out_pos[1][1] - 0.7).abs() < 1e-5);
        let n0_len: f32 = out_nrm[0].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n0_len - 1.0).abs() < 1e-5);
    }
}
