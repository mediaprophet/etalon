//! 48-byte universal semantic datum, warp-scan sieve, and stable GPU LSD radix sort.

pub mod layout;
pub mod radix;
pub mod sieve;

pub use layout::NQuinWire;
pub use radix::{
    cpu_prefix_sum_256, radix_sort_u64_cpu, PrefixParams, RadixParams, RadixSortOutcome,
};
pub use sieve::{sieve_eq_cpu, QuinField, SieveParams, NQUIN_WARP_SCAN_WGSL};

#[cfg(feature = "gpu")]
pub use radix::GpuRadixSorter;
#[cfg(feature = "gpu")]
pub use sieve::{GpuNQuinSieve, GpuQuinSieve};

pub const NQUIN_BYTES: usize = 48;
pub const NQUIN_ALIGN: usize = 16;
