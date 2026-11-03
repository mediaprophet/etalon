//! Explicit verified local snapshots of caller-owned readers.
//!
//! A trusted expected length and complete digest identify the copied bytes; a
//! reader does not establish remote writer exclusion, coherent source versions,
//! or source authority. There is no implicit fallback from a direct lease.
//! The caller supplies an existing trusted local parent. Output backing is
//! checked by opened handle before payload writes, then leased and rehashed.
//! Cancellation is checked between calls, not inside a blocking reader or OS
//! operation. Reader allocations and OS metadata/cache costs are not budgeted.
//! Explicit close/errors retain failed cleanup for retry; Drop/unwind cleanup
//! is best effort under OS refusal, and process abort is outside this contract.
#![forbid(unsafe_code)]
mod copy;
mod error;
mod limits;
mod owner;

pub use error::{SnapshotError, SnapshotErrorKind};
pub use limits::{SnapshotLimits, COPY_BUFFER_BYTES};
pub use owner::{CleanupPending, Snapshot};

#[cfg(test)]
mod tests;
