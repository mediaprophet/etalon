//! 48-byte universal wire layout and compile-time ABI assertions.

/// 48-byte Super-Quin wire layout.
/// Six 64-bit integer fields aligned to 16 bytes.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NQuinWire {
    pub subject: u64,
    pub predicate: u64,
    pub object: u64,
    pub context: u64,
    pub metadata: u64,
    pub parity: u64,
}

impl NQuinWire {
    pub const fn new(
        subject: u64,
        predicate: u64,
        object: u64,
        context: u64,
        metadata: u64,
        parity: u64,
    ) -> Self {
        Self {
            subject,
            predicate,
            object,
            context,
            metadata,
            parity,
        }
    }

    /// Calculate parity check (subject ^ predicate ^ object ^ context).
    #[inline(always)]
    pub fn compute_parity(&self) -> u64 {
        self.subject ^ self.predicate ^ self.object ^ self.context
    }

    #[inline(always)]
    pub fn is_parity_valid(&self) -> bool {
        self.compute_parity() == self.parity
    }
}

const _: () = {
    assert!(core::mem::size_of::<NQuinWire>() == 48);
    assert!(core::mem::align_of::<NQuinWire>() == 16);
};
