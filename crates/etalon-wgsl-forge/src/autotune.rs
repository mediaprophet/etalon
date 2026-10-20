//! Pareto-frontier schedule tuning and benchmarking.

use crate::schedule::{Schedule, TileConfig, WorkgroupSize};

pub fn generate_search_candidates() -> Vec<Schedule> {
    vec![
        Schedule::new(WorkgroupSize::new(128, 1, 1), TileConfig::new(8, 8, 8), 4),
        Schedule::new(WorkgroupSize::new(256, 1, 1), TileConfig::new(16, 16, 16), 4),
        Schedule::new(WorkgroupSize::new(256, 1, 1), TileConfig::new(16, 16, 32), 4),
    ]
}
