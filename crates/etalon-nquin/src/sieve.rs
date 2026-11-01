//! Equality sieve, Warp-Transposed `uvec4` GPU scan pipeline, and CPU verification oracle.

use bytemuck::{Pod, Zeroable};

use super::layout::NQuinWire;

/// Raw WGSL source for the bank-conflict-free warp-transposed `uvec4` NQuin scan shader.
pub const NQUIN_WARP_SCAN_WGSL: &str = include_str!("shaders/nquin_warp_scan.wgsl");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum QuinField {
    Subject = 0,
    Predicate = 1,
    Object = 2,
    Context = 3,
}

impl QuinField {
    #[inline(always)]
    pub fn extract(&self, quin: &NQuinWire) -> u64 {
        match self {
            Self::Subject => quin.subject,
            Self::Predicate => quin.predicate,
            Self::Object => quin.object,
            Self::Context => quin.context,
        }
    }

    #[inline(always)]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct SieveParams {
    pub n: u32,
    pub field: u32,
    pub needle_lo: u32,
    pub needle_hi: u32,
}

/// Compact Quins whose `field == needle` into `out`. Returns count written.
pub fn sieve_eq_cpu(
    quins: &[NQuinWire],
    field: QuinField,
    needle: u64,
    out: &mut [NQuinWire],
) -> usize {
    let mut written = 0;
    for q in quins {
        if field.extract(q) == needle {
            if written < out.len() {
                out[written] = *q;
                written += 1;
            } else {
                break;
            }
        }
    }
    written
}

#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "gpu")]
pub use gpu::{GpuNQuinSieve, GpuQuinSieve};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nquin_warp_scan_wgsl_parses_and_validates_in_naga() {
        let module = naga::front::wgsl::parse_str(NQUIN_WARP_SCAN_WGSL)
            .expect("nquin_warp_scan.wgsl must parse cleanly");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator
            .validate(&module)
            .expect("nquin_warp_scan.wgsl must validate cleanly through naga");
    }

    #[test]
    fn sieve_eq_cpu_filters_exact_field() {
        let q0 = NQuinWire::new(10, 0x100, 20, 1, 0, 10 ^ 0x100 ^ 20 ^ 1);
        let q1 = NQuinWire::new(11, 0x200, 21, 1, 0, 11 ^ 0x200 ^ 21 ^ 1);
        let q2 = NQuinWire::new(12, 0x100, 22, 2, 0, 12 ^ 0x100 ^ 22 ^ 2);
        let quins = [q0, q1, q2];
        let mut out = [NQuinWire::new(0, 0, 0, 0, 0, 0); 3];
        let count = sieve_eq_cpu(&quins, QuinField::Predicate, 0x100, &mut out);
        assert_eq!(count, 2);
        assert_eq!(out[0], q0);
        assert_eq!(out[1], q2);
    }
}
