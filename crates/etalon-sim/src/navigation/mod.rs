//! Unified 2D Grid & 2.5D Polygon Navigation Engine (`navigation`).
//!
//! Provides:
//! - 2D integer grid A* pathfinding and group formations (`grid2d`),
//! - Flocking / boids crowd separation steering forces (`compute_boids_separation`),
//! - 2.5D convex polygon NavMesh graph A* pathfinding with string-pulling (`navmesh`),
//! - Continuous world-space waypoint projection (`project_grid_path_to_waypoints`).

pub mod grid2d;
pub mod navmesh;

pub use grid2d::*;
pub use navmesh::*;

/// Projects a 2D integer grid path into continuous world-space waypoints (`Point2D` on XZ plane),
/// centering each cell coordinate and scaling by `cell_size_m`.
///
/// Returns the number of waypoints written into `out`.
pub fn project_grid_path_to_waypoints(
    grid_path: &[(i32, i32)],
    cell_size_m: f32,
    origin_x: f32,
    origin_z: f32,
    out: &mut [Point2D],
) -> usize {
    let count = grid_path.len().min(out.len());
    for i in 0..count {
        let (gx, gy) = grid_path[i];
        out[i] = [
            origin_x + (gx as f32 + 0.5) * cell_size_m,
            origin_z + (gy as f32 + 0.5) * cell_size_m,
        ];
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_grid_path_to_waypoints() {
        let grid_path = [(0, 0), (1, 0), (2, 1)];
        let mut waypoints = [[0.0, 0.0]; 4];
        let written = project_grid_path_to_waypoints(&grid_path, 2.0, 10.0, 20.0, &mut waypoints);

        assert_eq!(written, 3);
        // (0 + 0.5) * 2.0 + 10.0 = 11.0, (0 + 0.5) * 2.0 + 20.0 = 21.0
        assert_eq!(waypoints[0], [11.0, 21.0]);
        // (1 + 0.5) * 2.0 + 10.0 = 13.0, (0 + 0.5) * 2.0 + 20.0 = 21.0
        assert_eq!(waypoints[1], [13.0, 21.0]);
        // (2 + 0.5) * 2.0 + 10.0 = 15.0, (1 + 0.5) * 2.0 + 20.0 = 23.0
        assert_eq!(waypoints[2], [15.0, 23.0]);
    }
}
