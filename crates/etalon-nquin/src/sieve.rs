//! Equality sieve and CPU verification oracle.

use super::layout::NQuinWire;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuinField {
    Subject,
    Predicate,
    Object,
    Context,
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
}

/// Compact Quins whose `field == needle` into `out`. Returns count written.
pub fn sieve_eq_cpu(quins: &[NQuinWire], field: QuinField, needle: u64, out: &mut [NQuinWire]) -> usize {
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
