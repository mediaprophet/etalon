use super::CleanupPending;
use crate::lease::LeaseError;

#[derive(Debug)]
pub enum SnapshotErrorKind {
    UnsupportedPlatform,
    InvalidIdentity,
    InvalidLength,
    PathBudget,
    Budget,
    ReadCallBudget,
    WriteCallBudget,
    InvalidReadCount,
    ShortSource,
    LongSource,
    ContentMismatch,
    Cancelled,
    Io(std::io::Error),
    Lease(LeaseError),
}

#[derive(Debug)]
pub struct SnapshotError {
    pub(crate) kind: SnapshotErrorKind,
    pub(crate) cleanup: Option<CleanupPending>,
}
impl SnapshotError {
    pub fn kind(&self) -> &SnapshotErrorKind {
        &self.kind
    }
    pub fn take_cleanup(&mut self) -> Option<CleanupPending> {
        self.cleanup.take()
    }
    pub fn cleanup_pending(&self) -> bool {
        self.cleanup.is_some()
    }
}
impl From<SnapshotErrorKind> for SnapshotError {
    fn from(kind: SnapshotErrorKind) -> Self {
        Self {
            kind,
            cleanup: None,
        }
    }
}
impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "model snapshot failed: {:?}; cleanup pending: {}",
            self.kind,
            self.cleanup_pending()
        )
    }
}
impl std::error::Error for SnapshotError {}
