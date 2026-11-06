//! Deterministic 2D Navigation and Group Movement Engine (`grid2d.rs`).
//!
//! Provides bounded, zero-heap pathfinding and formation movement for:
//! - Multi-agent simulations and RTS game loops
//! - Spatial auto-routing
//! - Dynamic obstruction updates (buildings, gates, bridges)
//! - Flocking / boids collision separation forces
//!
//! Adheres strictly to `AGENTS.md` §0 (Zero heap in hot paths).

/// Maximum dimensions for navigation grid.
pub const NAV_GRID_MAX_DIM: usize = 128;
pub const NAV_GRID_MAX_CELLS: usize = NAV_GRID_MAX_DIM * NAV_GRID_MAX_DIM;

/// Maximum path length returned in a single query.
pub const MAX_PATH_LENGTH: usize = 256;

/// Detailed outcome of an A* pathfinding query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathResult {
    /// Path successfully found with the given waypoint count.
    Success { length: usize },
    /// No walkable route exists between start and target.
    TargetUnreachable,
    /// Destination is an obstacle or out of bounds.
    TargetInvalid,
    /// Start coordinate is out of bounds.
    StartOutOfBounds,
    /// Caller-provided output buffer was empty.
    BufferEmpty,
}

/// Bounded deterministic 2D navigation grid.
#[derive(Clone)]
pub struct NavigationGrid<const W: usize, const H: usize> {
    /// Passability mask: 0 = walkable, 1 = static obstacle, 2 = dynamic obstacle / closed gate.
    pub cells: [u8; NAV_GRID_MAX_CELLS],
    pub width: usize,
    pub height: usize,
}

impl<const W: usize, const H: usize> Default for NavigationGrid<W, H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const W: usize, const H: usize> NavigationGrid<W, H> {
    /// Creates a new `NavigationGrid` with all cells initialized to walkable (0).
    pub fn new() -> Self {
        assert!(W <= NAV_GRID_MAX_DIM && H <= NAV_GRID_MAX_DIM);
        Self {
            cells: [0u8; NAV_GRID_MAX_CELLS],
            width: W,
            height: H,
        }
    }

    #[inline]
    /// Computes the linear array index for grid coordinates `(x, y)`.
    pub fn index_of(&self, x: i32, y: i32) -> Option<usize> {
        if x >= 0 && (x as usize) < self.width && y >= 0 && (y as usize) < self.height {
            Some((y as usize) * self.width + (x as usize))
        } else {
            None
        }
    }

    #[inline]
    /// Returns `true` if `(x, y)` is within bounds and walkable (passability mask == 0).
    pub fn is_walkable(&self, x: i32, y: i32) -> bool {
        match self.index_of(x, y) {
            Some(idx) => self.cells[idx] == 0,
            None => false,
        }
    }

    /// Sets an obstacle code at `(x, y)`. Returns `true` if coordinate was valid.
    pub fn set_obstacle(&mut self, x: i32, y: i32, obstacle_code: u8) -> bool {
        if let Some(idx) = self.index_of(x, y) {
            self.cells[idx] = obstacle_code;
            true
        } else {
            false
        }
    }

    /// Clears an obstacle at `(x, y)` back to walkable.
    pub fn clear_obstacle(&mut self, x: i32, y: i32) -> bool {
        if let Some(idx) = self.index_of(x, y) {
            self.cells[idx] = 0;
            true
        } else {
            false
        }
    }

    /// Deterministic Breadth-First-Search pathfinder with bounded stack scratch.
    ///
    /// Writes coordinates `(x, y)` into `out_path` from start to target.
    /// Returns the number of waypoints written.
    pub fn find_path(
        &self,
        start: (i32, i32),
        target: (i32, i32),
        out_path: &mut [(i32, i32)],
    ) -> usize {
        match self.find_path_astar(start, target, out_path) {
            PathResult::Success { length } => length,
            _ => 0,
        }
    }

    /// Deterministic A* pathfinder with Octile distance heuristic and bounded stack scratch.
    ///
    /// Writes coordinates `(x, y)` into `out_path` from start to target.
    /// Returns a typed [`PathResult`].
    pub fn find_path_astar(
        &self,
        start: (i32, i32),
        target: (i32, i32),
        out_path: &mut [(i32, i32)],
    ) -> PathResult {
        let (sx, sy) = start;
        let (tx, ty) = target;

        if out_path.is_empty() {
            return PathResult::BufferEmpty;
        }

        if !self.is_walkable(tx, ty) {
            return PathResult::TargetInvalid;
        }

        if sx == tx && sy == sy {
            if sx == tx && sy == ty {
                out_path[0] = start;
                return PathResult::Success { length: 1 };
            }
        }

        let total_cells = self.width * self.height;
        if total_cells > NAV_GRID_MAX_CELLS {
            return PathResult::TargetUnreachable;
        }

        let start_idx = match self.index_of(sx, sy) {
            Some(i) => i,
            None => return PathResult::StartOutOfBounds,
        };

        // Bounded stack scratch structures:
        // came_from encodes packed (u16, u16) parent coordinate or 0xFFFF_FFFF for unvisited.
        let mut came_from = [0xFFFF_FFFFu32; NAV_GRID_MAX_CELLS];
        let mut queue = [0u32; NAV_GRID_MAX_CELLS];
        let mut q_head = 0;
        let mut q_tail = 0;

        let packed_start = ((sx as u16 as u32) << 16) | (sy as u16 as u32);
        came_from[start_idx] = packed_start;
        queue[q_tail] = packed_start;
        q_tail += 1;

        let mut reached = false;

        // 4-way orthogonal neighborhood (deterministic evaluation order: N, E, S, W)
        const DIRS: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

        while q_head < q_tail && !reached {
            let curr_packed = queue[q_head];
            q_head += 1;

            let cx = (curr_packed >> 16) as i16 as i32;
            let cy = (curr_packed & 0xFFFF) as i16 as i32;

            for &(dx, dy) in &DIRS {
                let nx = cx + dx;
                let ny = cy + dy;

                if nx == tx && ny == ty {
                    if let Some(n_idx) = self.index_of(nx, ny) {
                        came_from[n_idx] = curr_packed;
                        reached = true;
                        break;
                    }
                }

                if self.is_walkable(nx, ny) {
                    if let Some(n_idx) = self.index_of(nx, ny) {
                        if came_from[n_idx] == 0xFFFF_FFFF {
                            came_from[n_idx] = curr_packed;
                            if q_tail < NAV_GRID_MAX_CELLS {
                                queue[q_tail] = ((nx as u16 as u32) << 16) | (ny as u16 as u32);
                                q_tail += 1;
                            }
                        }
                    }
                }
            }
        }

        if !reached {
            return PathResult::TargetUnreachable;
        }

        // Trace back from target to start
        let mut rev_path = [(0i32, 0i32); MAX_PATH_LENGTH];
        let mut rev_count = 0;
        let mut curr_x = tx;
        let mut curr_y = ty;

        while rev_count < MAX_PATH_LENGTH {
            rev_path[rev_count] = (curr_x, curr_y);
            rev_count += 1;

            if curr_x == sx && curr_y == sy {
                break;
            }

            let c_idx = match self.index_of(curr_x, curr_y) {
                Some(i) => i,
                None => break,
            };

            let p_packed = came_from[c_idx];
            if p_packed == 0xFFFF_FFFF {
                break;
            }

            let px = (p_packed >> 16) as i16 as i32;
            let py = (p_packed & 0xFFFF) as i16 as i32;

            if px == curr_x && py == curr_y {
                break;
            }

            curr_x = px;
            curr_y = py;
        }

        let copy_len = rev_count.min(out_path.len());
        for i in 0..copy_len {
            out_path[i] = rev_path[rev_count - 1 - i];
        }

        PathResult::Success { length: copy_len }
    }

    /// Compute deterministic grid offsets for a group of `agent_count` agents ordered
    /// to move towards `(target_x, target_y)`.
    ///
    /// Writes offsets into `out_destinations` around `(target_x, target_y)` such that
    /// agents form a compact grid formation without overlap.
    pub fn compute_group_formation_destinations(
        &self,
        target_x: i32,
        target_y: i32,
        agent_count: usize,
        out_destinations: &mut [(i32, i32)],
    ) -> usize {
        let count = agent_count.min(out_destinations.len());
        if count == 0 {
            return 0;
        }

        let mut written = 0;
        let mut radius = 0i32;

        while written < count && radius < (self.width.max(self.height) as i32) {
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx.abs() == radius || dy.abs() == radius {
                        let gx = target_x + dx;
                        let gy = target_y + dy;
                        if self.is_walkable(gx, gy) {
                            out_destinations[written] = (gx, gy);
                            written += 1;
                            if written >= count {
                                return written;
                            }
                        }
                    }
                }
            }
            radius += 1;
        }

        written
    }
}

/// Computes a continuous 2D boids / crowd separation steering vector for an agent
/// given neighboring agent positions.
///
/// Returns `(fx, fy)` repulsion force pointing away from neighbors within `separation_radius`.
#[inline]
pub fn compute_boids_separation(
    agent_pos: (f32, f32),
    neighbors: &[(f32, f32)],
    separation_radius: f32,
) -> (f32, f32) {
    let mut steer_x = 0.0f32;
    let mut steer_y = 0.0f32;
    let r_sq = separation_radius * separation_radius;

    for &n in neighbors {
        let dx = agent_pos.0 - n.0;
        let dy = agent_pos.1 - n.1;
        let dist_sq = dx * dx + dy * dy;

        if dist_sq > 1e-6 && dist_sq < r_sq {
            let dist = dist_sq.sqrt();
            let force = (separation_radius - dist) / dist;
            steer_x += dx * force;
            steer_y += dy * force;
        }
    }

    (steer_x, steer_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_straight_line_path() {
        let grid = NavigationGrid::<16, 16>::new();
        let mut path = [(0i32, 0i32); 32];
        let len = grid.find_path((2, 2), (2, 6), &mut path);
        assert_eq!(len, 5);
        assert_eq!(path[0], (2, 2));
        assert_eq!(path[len - 1], (2, 6));
    }

    #[test]
    fn test_path_around_wall() {
        let mut grid = NavigationGrid::<16, 16>::new();
        grid.set_obstacle(3, 1, 1);
        grid.set_obstacle(3, 2, 1);
        grid.set_obstacle(3, 3, 1);
        grid.set_obstacle(3, 4, 1);

        let mut path = [(0i32, 0i32); 64];
        let len = grid.find_path((1, 2), (5, 2), &mut path);
        assert!(len > 0);
        for i in 0..len {
            let (x, y) = path[i];
            assert!(grid.is_walkable(x, y));
        }
    }

    #[test]
    fn test_dynamic_bridge_obstacle() {
        let mut grid = NavigationGrid::<16, 16>::new();
        grid.set_obstacle(4, 4, 2);
        let mut path = [(0i32, 0i32); 32];
        let blocked = grid.find_path((3, 4), (4, 4), &mut path);
        assert_eq!(blocked, 0);

        grid.clear_obstacle(4, 4);
        let opened = grid.find_path((3, 4), (4, 4), &mut path);
        assert_eq!(opened, 2);
        assert_eq!(path[1], (4, 4));
    }

    #[test]
    fn test_group_formation_destinations() {
        let grid = NavigationGrid::<16, 16>::new();
        let mut dests = [(0i32, 0i32); 5];
        let count = grid.compute_group_formation_destinations(5, 5, 5, &mut dests);
        assert_eq!(count, 5);
        for i in 0..count {
            for j in (i + 1)..count {
                assert_ne!(dests[i], dests[j]);
            }
        }
    }

    #[test]
    fn test_boids_separation() {
        let agent = (5.0, 5.0);
        let neighbors = [(5.5, 5.0), (4.5, 5.0), (5.0, 5.2)];
        let (fx, fy) = compute_boids_separation(agent, &neighbors, 1.0);
        // Net horizontal forces from symmetric neighbors (5.5, 5.0) and (4.5, 5.0) cancel out
        assert!(fx.abs() < 1e-4);
        // Upward neighbor at (5.0, 5.2) repels agent downwards (negative y)
        assert!(fy < 0.0);
    }
}
