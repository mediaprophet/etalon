//! Local artifact integrity admission, separate from loading or attestation.
//!
//! The host authenticates the manifest and protects the root and its ancestors
//! throughout validation, runtime loading and use. Final-component checks do
//! not establish locality of mapped volumes, mounts or ancestor redirections.
//! ORT may reuse a previously loaded process-global library: these checks do
//! not prove which library ORT uses. Initialize before any other ORT use.
#![forbid(unsafe_code)]

mod paths;
mod verify;
pub mod fastrpc;
pub mod rtlsdr;
pub use verify::validate_deployment;

use std::path::Path;

pub const MAX_HASH_SCRATCH_BYTES: usize = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderKind { OpenVinoNpu, QnnHtp }

#[derive(Clone, Copy, Debug)]
pub struct ArtifactSpec<'a> {
    pub path: &'a Path,
    pub bytes: u64,
    pub sha256: [u8; 32],
}

/// Paths, hashes, platform and deployment IDs are host declarations. Hashes
/// must come from an authenticated manifest, not the files being admitted.
pub struct DeploymentManifest<'a> {
    pub deployment_id: [u8; 32],
    pub target_id: u64,
    pub provider_kind: ProviderKind,
    pub protected_root: &'a Path,
    pub runtime: ArtifactSpec<'a>,
    /// May be absent for a provider linked into the admitted ORT build.
    /// QNN HTP requires its separately admitted backend library.
    pub provider: Option<ArtifactSpec<'a>>,
}

pub trait DeploymentAuthority {
    /// Authenticate every declaration, enforce root/ACL/network isolation and
    /// hold an immutable-artifact/policy lease through validation and all use.
    fn admit(&self, manifest: &DeploymentManifest<'_>) -> bool;
}

#[derive(Debug)]
pub enum DeploymentError {
    InvalidManifest, InvalidPath, InvalidBudget, NotAuthorized, NotRegular,
    LengthMismatch, DigestMismatch, UnsupportedPlatform, Io(std::io::Error),
}
impl From<std::io::Error> for DeploymentError {
    fn from(error: std::io::Error) -> Self { Self::Io(error) }
}

/// Exact artifacts matched the declared lengths and hashes during this check.
/// This is not a loaded-library identity, signature, hardware or driver proof.
pub struct AdmittedDeployment<'manifest, 'path, 'authority> {
    manifest: &'manifest DeploymentManifest<'path>,
    authority: &'authority dyn DeploymentAuthority,
    total_bytes: u64,
}

impl AdmittedDeployment<'_, '_, '_> {
    pub fn manifest(&self) -> &DeploymentManifest<'_> { self.manifest }
    pub fn runtime_path(&self) -> &Path { self.manifest.runtime.path }
    pub fn provider_path(&self) -> Option<&Path> { self.manifest.provider.map(|a| a.path) }
    pub fn provider_kind(&self) -> ProviderKind { self.manifest.provider_kind }
    pub fn total_verified_bytes(&self) -> u64 { self.total_bytes }
    /// Recheck host authority only. The host's artifact lease must prevent
    /// replacement; this does not rehash files or attest global ORT state.
    pub fn revalidate_authority(&self) -> Result<(), DeploymentError> {
        if self.authority.admit(self.manifest) { Ok(()) } else { Err(DeploymentError::NotAuthorized) }
    }
}
