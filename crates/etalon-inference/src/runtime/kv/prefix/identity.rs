#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PrefixIdentity {
    pub words: [u64; 4],
}
