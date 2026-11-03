use super::layout::P64TensorEntry;

#[derive(Clone, Copy, Debug)]
pub struct P64TensorView<'a> {
    pub(super) entry: P64TensorEntry,
    pub(super) name: &'a [u8],
    pub(super) blob: &'a [u8],
}

impl<'a> P64TensorView<'a> {
    pub fn entry(&self) -> P64TensorEntry {
        self.entry
    }
    pub fn name_bytes(&self) -> &'a [u8] {
        self.name
    }
    pub fn blob(&self) -> &'a [u8] {
        self.blob
    }
}

/// Raw manifold bytes. No semantic or finite-value interpretation is imposed.
#[derive(Clone, Copy, Debug)]
pub struct P64ManifoldRecord<'a> {
    pub(super) bytes: &'a [u8; 64],
}

impl<'a> P64ManifoldRecord<'a> {
    pub fn bytes(&self) -> &'a [u8; 64] {
        self.bytes
    }

    pub fn values_le(&self) -> [f32; 10] {
        core::array::from_fn(|index| {
            let start = index * 4;
            f32::from_le_bytes(self.bytes[start..start + 4].try_into().unwrap())
        })
    }
}
