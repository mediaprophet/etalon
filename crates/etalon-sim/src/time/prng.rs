//! Deterministic PRNG Kernel (xorshift64*).
//!
//! Provides reproducible pseudo-random integer streams for fixed-tick simulation
//! state machines and ambient entity updates without runtime heap allocations.

/// Deterministic 64-bit pseudo-random number generator (xorshift64*).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeterministicPrng {
    state: u64,
}

impl Default for DeterministicPrng {
    fn default() -> Self {
        Self::new(0x5EED_5EED_CAFE_BABE)
    }
}

impl DeterministicPrng {
    /// Creates a new `DeterministicPrng` with the given non-zero seed.
    pub const fn new(seed: u64) -> Self {
        // Ensure non-zero initial state
        let s = if seed == 0 {
            0x5EED_5EED_CAFE_BABE
        } else {
            seed
        };
        Self { state: s }
    }

    /// Advances the generator state and returns the next pseudo-random `u64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Returns a pseudo-random integer in the half-open range `[0, bound)`.
    #[inline]
    pub fn next_bounded(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            0
        } else {
            self.next_u64() % bound
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_prng_reproducibility() {
        let mut prng1 = DeterministicPrng::new(12345);
        let mut prng2 = DeterministicPrng::new(12345);

        for _ in 0..100 {
            assert_eq!(prng1.next_u64(), prng2.next_u64());
            assert_eq!(prng1.next_bounded(10), prng2.next_bounded(10));
        }
    }

    #[test]
    fn test_deterministic_prng_zero_seed_fallback() {
        let mut prng = DeterministicPrng::new(0);
        assert_ne!(prng.next_u64(), 0);
    }
}
