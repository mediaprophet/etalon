//! Unified Multi-Rate Time & Fixed-Tick Engine (`time`).
//!
//! Provides:
//! - 64 Hz binary-exact continuous simulation accumulator (`FixedTimestepAccumulator`),
//! - Seeded deterministic xorshift64* pseudo-random number generator (`DeterministicPrng`),
//! - 20 Hz deterministic lockstep multi-agent world-state kernel (`FixedTickWorld`),
//! - Multi-rate clock bridging high-rate continuous physics to discrete lockstep ticks (`MultiRateSimClock`).

pub mod accumulator;
pub mod fixed_tick;
pub mod prng;

#[cfg(test)]
mod tests;

pub use accumulator::*;
pub use fixed_tick::*;
pub use prng::*;

/// Multi-rate simulation clock coordinating 64 Hz continuous physics substeps
/// with deterministic lockstep world-state ticks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiRateSimClock {
    /// 64 Hz fixed timestep accumulator for high-rate continuous physics and character stepping.
    pub physics_accumulator: FixedTimestepAccumulator,
    /// Number of physics substeps required to trigger one lockstep tick (typically 3 for ~21.3 Hz ≈ 20 Hz).
    pub lockstep_substep_ratio: u32,
    /// Accumulated physics substeps since the last lockstep tick.
    pub substeps_since_lockstep: u32,
    /// Total deterministic lockstep ticks executed.
    pub total_lockstep_ticks: u64,
}

impl Default for MultiRateSimClock {
    fn default() -> Self {
        Self::new(3) // 64 Hz / 3 = 21.33 Hz lockstep rate
    }
}

impl MultiRateSimClock {
    /// Creates a new `MultiRateSimClock` with the given substep-to-tick ratio.
    pub const fn new(lockstep_substep_ratio: u32) -> Self {
        Self {
            physics_accumulator: FixedTimestepAccumulator::new(),
            lockstep_substep_ratio: if lockstep_substep_ratio == 0 {
                1
            } else {
                lockstep_substep_ratio
            },
            substeps_since_lockstep: 0,
            total_lockstep_ticks: 0,
        }
    }

    /// Advances the clock by real frame delta time in seconds.
    ///
    /// Returns `(physics_substeps, lockstep_ticks)` to execute during this frame.
    pub fn advance(&mut self, raw_delta_seconds: f64) -> (usize, usize) {
        let physics_steps = self.physics_accumulator.advance(raw_delta_seconds);
        let mut lockstep_ticks = 0;

        for _ in 0..physics_steps {
            self.substeps_since_lockstep += 1;
            if self.substeps_since_lockstep >= self.lockstep_substep_ratio {
                self.substeps_since_lockstep = 0;
                self.total_lockstep_ticks += 1;
                lockstep_ticks += 1;
            }
        }

        (physics_steps, lockstep_ticks)
    }

    /// Returns fractional blend factor $\alpha \in [0, 1)$ between the previous
    /// and current physics states for renderer interpolation.
    #[inline]
    pub fn interpolation_alpha(&self) -> f32 {
        self.physics_accumulator.interpolation_alpha()
    }

    /// Resets all counters and accumulator.
    pub fn reset(&mut self) {
        self.physics_accumulator.reset();
        self.substeps_since_lockstep = 0;
        self.total_lockstep_ticks = 0;
    }
}

#[cfg(test)]
mod multi_rate_tests {
    use super::*;

    #[test]
    fn test_multi_rate_sim_clock_coupling() {
        let mut clock = MultiRateSimClock::new(3);

        // 16 frames of 0.0625s = 1.0s total: 64 physics steps, 64 / 3 = 21 lockstep ticks, 1 remaining
        let mut physics_steps = 0;
        let mut lockstep_ticks = 0;
        for _ in 0..16 {
            let (p, l) = clock.advance(0.0625);
            physics_steps += p;
            lockstep_ticks += l;
        }
        assert_eq!(physics_steps, 64);
        assert_eq!(lockstep_ticks, 21);
        assert_eq!(clock.substeps_since_lockstep, 1);
        assert_eq!(clock.total_lockstep_ticks, 21);

        // Advance two more 64 Hz steps (2 * 0.015625 = 0.03125s)
        let (p2, l2) = clock.advance(0.03125);
        assert_eq!(p2, 2);
        // substeps_since was 1 + 2 = 3 -> triggers 1 lockstep tick!
        assert_eq!(l2, 1);
        assert_eq!(clock.substeps_since_lockstep, 0);
        assert_eq!(clock.total_lockstep_ticks, 22);
    }
}
