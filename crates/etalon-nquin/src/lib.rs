//! 48-byte universal semantic datum, warp-scan sieve, and stable GPU LSD radix sort.

pub mod layout;
pub mod sieve;
pub mod radix;

pub use layout::NQuinWire;
pub use sieve::{sieve_eq_cpu, QuinField};
pub use radix::{radix_sort_u64_cpu, RadixSortOutcome};

pub const NQUIN_BYTES: usize = 48;
pub const NQUIN_ALIGN: usize = 16;
