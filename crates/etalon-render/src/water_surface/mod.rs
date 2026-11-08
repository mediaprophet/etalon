//! Caller-buffered shoreline attributes, Gerstner spectra, buoyancy, optics, and ripples for open water.

pub mod buoyancy;
pub mod gerstner;
pub mod optics;
pub mod ripple;
pub mod shoreline;
pub mod spatial;

pub use buoyancy::*;
pub use gerstner::*;
pub use optics::*;
pub use ripple::*;
pub use shoreline::{
    shore_distance_into, shore_topology_into, ShoreError, ShoreReceipt, ShoreTopology,
};
pub use spatial::{
    build_bvh_recursive, point_segment_distance_sq_2d, shore_distance_accelerated_into,
    sort_by_morton_3d, spatial_working_bytes, Aabb, BvhError, BvhNode, Point2, Point3,
    ShoreSpatialScratch, SpatialOrderError, BVH_NODE_SIZE, MAX_BVH_DEPTH,
};

/// Allocation classification for water surface geometry passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationClass {
    /// Mandatory zero-heap hot-path pass.
    HotZeroHeap,
    /// Bounded cold construction pass.
    ColdBounded,
}

pub const ALLOCATION_CLASS: AllocationClass = AllocationClass::HotZeroHeap;

/// The optional spatial builder delegates to the native bounded cold BVH
/// lifecycle; nearest-edge traversal itself stays allocation-free.
pub const SPATIAL_BUILD_ALLOCATION_CLASS: AllocationClass = AllocationClass::ColdBounded;

/// Combines an analytical Gerstner wave spectrum with a local dynamic interactive ripple buffer.
///
/// Returns the combined surface elevation and normalized surface normal `(height, [nx, ny, nz])`.
pub fn sample_combined_water_surface(
    spectrum: &SimdGerstnerSpectrum,
    ripple: &InteractiveRippleBuffer,
    x: f32,
    z: f32,
    t: f32,
) -> (f32, [f32; 3]) {
    let (g_h, g_n) = spectrum.sample_surface(x, z, t);
    let r_h = ripple.sample_height(x, z);
    let r_n = ripple.sample_normal(x, z);

    let height = g_h + r_h;

    let nx = g_n[0] + r_n[0];
    let ny = (g_n[1] + r_n[1]) * 0.5;
    let nz = g_n[2] + r_n[2];
    let len_sq = nx * nx + ny * ny + nz * nz;
    let inv_len = if len_sq > 1e-8 {
        1.0 / len_sq.sqrt()
    } else {
        1.0
    };

    (height, [nx * inv_len, ny * inv_len, nz * inv_len])
}

#[cfg(test)]
mod combined_tests {
    use super::*;

    #[test]
    fn test_sample_combined_water_surface() {
        let spectrum = SimdGerstnerSpectrum::coastal_preset(0.0);
        let mut ripple = InteractiveRippleBuffer::new(0.0, 0.0, 16.0, 2.5, 0.985);
        ripple.inject_impulse(0.0, 0.0, 0.5, 1.0);

        let (h, normal) = sample_combined_water_surface(&spectrum, &ripple, 0.0, 0.0, 0.0);
        assert!(h.is_finite());
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-4);
    }
}

#[cfg(test)]
mod spatial_tests;
#[cfg(test)]
mod tests;
