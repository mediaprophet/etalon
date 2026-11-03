//! Local-NTFS Windows model-byte lease; parsing, authority and loader adoption are separate.
//! Local disk/NTFS support is verified from the opened exclusion handle; remote,
//! redirected, unknown filesystem or ambiguous query results fail before mapping.
//! A retained read handle excludes ordinary writers/delete access before reading
//! length, mapping or hashing. All consumers must retain Arc<ModelLease> or borrow
//! its bytes; returning a bare Arc<Mmap> would break this lifetime guarantee.
//!
//! Source-byte plus mapped-range accounting is not a measured resident-memory
//! ceiling: hashing touches all pages, and OS working set/page-cache costs remain.
//! No scratch model copy is made. No snapshot fallback or platform emulation is
//! implicit. Privileged raw-volume writes and hardware faults are outside scope.
//! A secondary hardlink name may be unlinked/replaced without mutating the opened
//! file object. Namespace stability is not promised: loader/cache lookup must use
//! the retained lease identity, never re-resolve a pathname to regain authority.
mod backing;
mod identity;
#[cfg(windows)]
pub(crate) mod platform;
pub use backing::{LeaseError, LeaseIdentity, LeaseLimits, ModelLease};
pub use identity::{bundle_digest, BundleLimits, ModelPart, ModelPartRole};
#[cfg(test)]
mod tests;

#[cfg(all(test, not(windows)))]
mod unsupported_tests;
