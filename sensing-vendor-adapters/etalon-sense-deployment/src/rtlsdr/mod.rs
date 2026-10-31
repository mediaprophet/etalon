//! Explicit local RTL-SDR runtime admission, separate from device permission.

mod validate;
pub use validate::validate_rtl_sdr_deployment;

use crate::{ArtifactSpec, DeploymentError};
use std::path::Path;

pub const MAX_RTLSDR_DEPENDENCIES: usize = 64;
pub const MAX_RTLSDR_PAIRS: u32 = 1_048_576;

pub struct RtlSdrManifest<'a> {
    pub deployment_id: [u8; 32],
    pub target_id: u64,
    /// Fresh host-instance generation; not a physical device serial number.
    pub generation: u32,
    /// Explicit enumeration position. Native handle identity needs its own gate.
    pub device_index: u32,
    pub max_pairs: u32,
    pub protected_root: &'a Path,
    pub runtime: ArtifactSpec<'a>,
    pub dependencies: &'a [ArtifactSpec<'a>],
}

pub trait RtlSdrAuthority {
    /// Authenticate every manifest field and the complete native dependency
    /// closure. Protect roots, resolution and loaded code with an immutable
    /// no-egress lease. Authorize target/index/generation/capacity and platform
    /// ABI/driver compatibility; never reuse a generation for another instance.
    /// Enumeration position and USB strings are not authenticated identities.
    /// Device selection/configuration/acquisition require separate policy gates.
    fn admit(&self, manifest: &RtlSdrManifest<'_>) -> bool;
}

pub struct AdmittedRtlSdrDeployment<'manifest, 'path, 'authority> {
    manifest: &'manifest RtlSdrManifest<'path>,
    authority: &'authority dyn RtlSdrAuthority,
    total_bytes: u64,
}

impl AdmittedRtlSdrDeployment<'_, '_, '_> {
    pub fn manifest(&self) -> &RtlSdrManifest<'_> {
        self.manifest
    }
    pub fn runtime_path(&self) -> &Path {
        self.manifest.runtime.path
    }
    pub fn target_id(&self) -> u64 {
        self.manifest.target_id
    }
    pub fn generation(&self) -> u32 {
        self.manifest.generation
    }
    pub fn device_index(&self) -> u32 {
        self.manifest.device_index
    }
    pub fn max_pairs(&self) -> u32 {
        self.manifest.max_pairs
    }
    pub fn total_verified_bytes(&self) -> u64 {
        self.total_bytes
    }
    /// Rechecks the authority lease, not file hashes or physical device state.
    pub fn revalidate_authority(&self) -> Result<(), DeploymentError> {
        if self.authority.admit(self.manifest) {
            Ok(())
        } else {
            Err(DeploymentError::NotAuthorized)
        }
    }
}
