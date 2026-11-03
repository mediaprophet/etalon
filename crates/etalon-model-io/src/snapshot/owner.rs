use super::{SnapshotError, SnapshotLimits};
use crate::lease::{LeaseIdentity, ModelLease};
use std::{io::Read, path::Path, sync::atomic::AtomicBool};

/// Immutable bytes and their owned scratch directory share one lifetime.
/// Field order releases the mapping/exclusion handle before directory cleanup.
pub struct Snapshot {
    pub(crate) lease: ModelLease,
    #[cfg(windows)]
    pub(crate) directory: tempfile::TempDir,
}
impl Snapshot {
    pub fn copy_from<R: Read + ?Sized>(
        source: &mut R,
        local_parent: &Path,
        expected: LeaseIdentity,
        limits: SnapshotLimits,
        cancelled: &AtomicBool,
    ) -> Result<Self, SnapshotError> {
        super::copy::copy_from(source, local_parent, expected, limits, cancelled)
    }
    pub fn bytes(&self) -> &[u8] {
        self.lease.bytes()
    }
    pub fn identity(&self) -> LeaseIdentity {
        self.lease.identity()
    }
    /// Consumes the owner; an Arc must have unique ownership before closing.
    pub fn close(self) -> Result<(), CleanupPending> {
        drop(self.lease);
        #[cfg(windows)]
        {
            CleanupPending::new(self.directory).retry()
        }
        #[cfg(not(windows))]
        {
            Ok(())
        }
    }
}
impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Snapshot")
            .field("identity", &self.identity())
            .finish_non_exhaustive()
    }
}

/// Owns only cleanup resources after all writer/mapping handles are released.
/// Dropping this value retries through TempDir's best-effort cleanup.
pub struct CleanupPending {
    #[cfg(windows)]
    directory: tempfile::TempDir,
    last_error: std::io::ErrorKind,
}
impl CleanupPending {
    #[cfg(windows)]
    pub(crate) fn new(directory: tempfile::TempDir) -> Self {
        Self {
            directory,
            last_error: std::io::ErrorKind::Other,
        }
    }
    pub fn last_error(&self) -> std::io::ErrorKind {
        self.last_error
    }
    pub fn retry(self) -> Result<(), Self> {
        #[cfg(windows)]
        {
            let mut owned = self;
            match remove_owned(&owned.directory) {
                Ok(()) => {
                    owned.removed();
                    Ok(())
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    owned.removed();
                    Ok(())
                }
                Err(error) => {
                    owned.last_error = error.kind();
                    Err(owned)
                }
            }
        }
        #[cfg(not(windows))]
        {
            Ok(())
        }
    }
    #[cfg(windows)]
    fn removed(&mut self) {
        // The owned pathname has been removed; never delete a later replacement.
        self.directory.disable_cleanup(true);
        #[cfg(test)]
        if RECREATE.get() {
            std::fs::create_dir(self.directory.path()).unwrap();
            std::fs::write(
                self.directory.path().join("replacement"),
                b"not the snapshot",
            )
            .unwrap();
        }
    }
}
impl std::fmt::Debug for CleanupPending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CleanupPending")
            .field("last_error", &self.last_error)
            .finish_non_exhaustive()
    }
}
#[cfg(windows)]
fn remove_owned(directory: &tempfile::TempDir) -> std::io::Result<()> {
    #[cfg(test)]
    if FAIL_REMOVE.get() {
        return Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied));
    }
    std::fs::remove_dir_all(directory.path())
}
#[cfg(all(test, windows))]
thread_local! { pub(super) static FAIL_REMOVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
#[cfg(all(test, windows))]
thread_local! { pub(super) static RECREATE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
