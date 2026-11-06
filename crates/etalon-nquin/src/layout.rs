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

    /// Construct a 48-byte `NQuinWire` with canonical 5-field XOR parity.
    #[inline(always)]
    pub const fn from_parts(
        subject: u64,
        predicate: u64,
        object: u64,
        context: u64,
        metadata: u64,
    ) -> Self {
        Self {
            subject,
            predicate,
            object,
            context,
            metadata,
            parity: subject ^ predicate ^ object ^ context ^ metadata,
        }
    }

    /// Calculate canonical 5-field parity check (`subject ^ predicate ^ object ^ context ^ metadata`).
    #[inline(always)]
    pub const fn compute_parity(&self) -> u64 {
        self.subject ^ self.predicate ^ self.object ^ self.context ^ self.metadata
    }

    #[inline(always)]
    pub const fn is_parity_valid(&self) -> bool {
        self.compute_parity() == self.parity
    }
}

const _: () = {
    assert!(core::mem::size_of::<NQuinWire>() == 48);
    assert!(core::mem::align_of::<NQuinWire>() == 16);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_field_xor_parity_includes_metadata() {
        let q = NQuinWire::from_parts(0x11, 0x22, 0x33, 0x44, 0x5500_0000_0001);
        assert!(q.is_parity_valid());
        assert_eq!(
            q.parity,
            0x11 ^ 0x22 ^ 0x33 ^ 0x44 ^ 0x5500_0000_0001
        );
        // 4-field parity omitting non-zero metadata must fail verification
        let q_bad = NQuinWire::new(
            0x11,
            0x22,
            0x33,
            0x44,
            0x5500_0000_0001,
            0x11 ^ 0x22 ^ 0x33 ^ 0x44,
        );
        assert!(!q_bad.is_parity_valid());
    }
}
