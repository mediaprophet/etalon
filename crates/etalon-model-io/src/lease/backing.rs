#[cfg(windows)]
use sha2::{Digest, Sha256};
#[cfg(windows)]
use std::sync::atomic::Ordering;
use std::{path::Path, sync::atomic::AtomicBool};

#[derive(Debug)]
pub enum LeaseError {
    UnsupportedPlatform,
    UnsupportedBacking,
    InvalidIdentity,
    InvalidLength,
    Budget,
    PathBudget,
    NotFile,
    Cancelled,
    ContentMismatch,
    InvalidBundle,
    Io(std::io::Error),
    #[cfg(test)]
    Injected,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeaseIdentity {
    pub content_sha256: [u8; 32],
    pub byte_len: u64,
}
#[derive(Clone, Copy, Debug)]
pub struct LeaseLimits {
    pub max_source_bytes: u64,
    pub max_mapped_bytes: u64,
    /// Admission ledger for source bytes + mapped address range; not RSS/commit.
    pub max_total_bytes: u64,
}
impl LeaseLimits {
    #[cfg(windows)]
    pub(crate) fn admit(self, length: u64) -> Result<usize, LeaseError> {
        let size = usize::try_from(length).map_err(|_| LeaseError::InvalidLength)?;
        if size == 0 || size > isize::MAX as usize {
            return Err(LeaseError::InvalidLength);
        }
        let charged = length.checked_add(length).ok_or(LeaseError::Budget)?;
        if length > self.max_source_bytes
            || length > self.max_mapped_bytes
            || charged > self.max_total_bytes
        {
            return Err(LeaseError::Budget);
        }
        Ok(size)
    }
}
/// Private immutable mapping followed by its exclusion handle in drop order.
/// No File/Mmap accessor, mutation method or independently cloneable mmap exists.
#[derive(Debug)]
pub struct ModelLease {
    #[cfg(windows)]
    mapping: memmap2::Mmap,
    #[cfg(windows)]
    _exclusion: std::fs::File,
    identity: LeaseIdentity,
}
impl ModelLease {
    /// `expected` is the independently selected full-file SHA-256, never a path,
    /// header-only digest or newly trusted metadata claim. This function verifies
    /// content equality, not the provenance/authorization of that expected value.
    /// Caller controls aggregate admission for concurrent leases and Arc storage.
    pub fn open(
        path: &Path,
        expected: [u8; 32],
        limits: LeaseLimits,
        cancelled: &AtomicBool,
    ) -> Result<Self, LeaseError> {
        #[cfg(not(windows))]
        {
            let _ = (path, expected, limits, cancelled);
            Err(LeaseError::UnsupportedPlatform)
        }
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            if expected == [0; 32] {
                return Err(LeaseError::InvalidIdentity);
            }
            if cancelled.load(Ordering::Acquire) {
                return Err(LeaseError::Cancelled);
            }
            // Bound the cold platform path encoding allocation before File::open.
            if path.as_os_str().encode_wide().take(32_768).count() >= 32_768 {
                return Err(LeaseError::PathBudget);
            }
            // FILE_SHARE_READ only: excludes existing/new write/delete handles
            // and existing writable file mappings, including closed-File views.
            let file = super::platform::open_supported(path)?;
            #[cfg(test)]
            fault(1, cancelled)?;
            let metadata = file.metadata().map_err(LeaseError::Io)?;
            if !metadata.is_file() {
                return Err(LeaseError::NotFile);
            }
            let length = metadata.len();
            let len = limits.admit(length)?;
            if cancelled.load(Ordering::Acquire) {
                return Err(LeaseError::Cancelled);
            }
            // SAFETY: exclusion was acquired before reading length and is held
            // until after this read-only mapping drops. Ordinary mutation/delete
            // handles and writable mappings cannot coexist with this handle.
            // The File is private/read-only, never duplicated or returned.
            let mapping = unsafe { memmap2::MmapOptions::new().len(len).map(&file) }
                .map_err(LeaseError::Io)?;
            #[cfg(test)]
            fault(2, cancelled)?;
            let mut hash = Sha256::new();
            for chunk in mapping.chunks(64 * 1024) {
                #[cfg(test)]
                fault(3, cancelled)?;
                if cancelled.load(Ordering::Acquire) {
                    return Err(LeaseError::Cancelled);
                }
                hash.update(chunk);
            }
            if cancelled.load(Ordering::Acquire) {
                return Err(LeaseError::Cancelled);
            }
            let digest: [u8; 32] = hash.finalize().into();
            if digest != expected {
                return Err(LeaseError::ContentMismatch);
            }
            Ok(Self {
                mapping,
                _exclusion: file,
                identity: LeaseIdentity {
                    content_sha256: digest,
                    byte_len: length,
                },
            })
        }
    }
    pub fn identity(&self) -> LeaseIdentity {
        self.identity
    }
    pub fn bytes(&self) -> &[u8] {
        #[cfg(windows)]
        {
            &self.mapping
        }
        #[cfg(not(windows))]
        {
            &[]
        }
    }
}
#[cfg(all(test, windows))]
thread_local! { pub(crate) static FAULT:std::cell::Cell<u8>=const {std::cell::Cell::new(0)}; }
#[cfg(all(test, windows))]
fn fault(stage: u8, cancelled: &AtomicBool) -> Result<(), LeaseError> {
    let active = FAULT.get();
    if (active == 1 && stage == 1) || (active == 2 && stage == 2) {
        return Err(LeaseError::Injected);
    }
    if active == 3 && stage == 2 {
        panic!("injected mapping unwind");
    }
    if active == 4 && stage == 3 {
        cancelled.store(true, Ordering::Release);
    }
    Ok(())
}
