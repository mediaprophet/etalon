//! Caller-buffered shoreline attributes for native open water meshes.

pub mod shoreline;
pub mod spatial;

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

#[cfg(test)]
mod spatial_tests;
#[cfg(test)]
mod tests;
