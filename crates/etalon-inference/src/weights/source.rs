use etalon_model_io::lease::ModelLease;
use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    sync::Arc,
};

pub(crate) struct LeaseBytes(pub Arc<ModelLease>);
impl AsRef<[u8]> for LeaseBytes {
    fn as_ref(&self) -> &[u8] {
        self.0.bytes()
    }
}
pub(crate) enum WeightCursor {
    File(File),
    Lease(io::Cursor<LeaseBytes>),
}
impl Read for WeightCursor {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::File(f) => f.read(out),
            Self::Lease(c) => c.read(out),
        }
    }
}
impl Seek for WeightCursor {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match self {
            Self::File(f) => f.seek(pos),
            Self::Lease(c) => c.seek(pos),
        }
    }
}
pub(crate) enum WeightMap {
    File(memmap2::Mmap),
    Lease(Arc<ModelLease>),
}
impl std::ops::Deref for WeightMap {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::File(m) => m,
            Self::Lease(l) => l.bytes(),
        }
    }
}
