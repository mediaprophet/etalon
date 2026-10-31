//! Local FastRPC artifact admission. This does not load or attest DSP firmware.

mod validate;
pub use validate::validate_fast_rpc_deployment;

use crate::{ArtifactSpec, DeploymentError};
use std::{ffi::CStr, path::Path};

pub const MAX_FASTRPC_CLOSURE_ARTIFACTS: usize = 64;
pub const MAX_FASTRPC_MODULE_URI_BYTES: usize = 1024;
pub const MAX_FASTRPC_SERVICE_PAIRS: u32 = 1_048_576;

/// All IDs and the exact URI are declarations authenticated by the authority.
pub struct FastRpcManifest<'a> {
    pub deployment_id: [u8; 32],
    pub target_id: u64,
    /// Admitted session generation carried by the exact authenticated URI.
    /// Never reuse a generation after close, reset or module restart.
    pub provider_generation: u32,
    /// Exact immutable service image IQ-pair capacity.
    pub service_max_pairs: u32,
    pub protected_root: &'a Path,
    pub runtime: ArtifactSpec<'a>,
    pub module: ArtifactSpec<'a>,
    /// Remaining native libraries, firmware and DSP-module dependencies.
    pub firmware_and_dependencies: &'a [ArtifactSpec<'a>],
    pub module_uri: &'a CStr,
    /// Physical DSP domain assertion; not proof of the execution processor.
    pub domain: u32,
    /// Session policy assertion; admission does not reserve or configure it.
    pub session: u32,
    pub service_schema_id: u64,
    pub service_schema_sha256: [u8; 32],
}

pub trait FastRpcAuthority {
    /// Authenticate the whole declaration against trusted signatures, target,
    /// domain/session and schema policy. Reject incomplete dependency/firmware
    /// closures or any URI whose resolution is not exactly the admitted module.
    /// Bind target ID and capacity to the signed image, and generation to the
    /// canonical forwarded URI parameter. Admit only one active session for
    /// this generation, never reusing it after close, reset or module restart.
    /// Protect roots, ancestors, runtime search paths and loaded modules with an
    /// immutable no-egress lease through validation, loading and every RPC.
    fn admit(&self, manifest: &FastRpcManifest<'_>) -> bool;
}

/// Exact declared bytes matched once. Authority owns signature verification,
/// target compatibility and protection against later replacement or redirection.
pub struct AdmittedFastRpcDeployment<'manifest, 'path, 'authority> {
    manifest: &'manifest FastRpcManifest<'path>,
    authority: &'authority dyn FastRpcAuthority,
    total_bytes: u64,
}

impl AdmittedFastRpcDeployment<'_, '_, '_> {
    pub fn manifest(&self) -> &FastRpcManifest<'_> {
        self.manifest
    }
    pub fn runtime_path(&self) -> &Path {
        self.manifest.runtime.path
    }
    pub fn module_path(&self) -> &Path {
        self.manifest.module.path
    }
    pub fn module_uri(&self) -> &CStr {
        self.manifest.module_uri
    }
    pub fn target_id(&self) -> u64 {
        self.manifest.target_id
    }
    pub fn provider_generation(&self) -> u32 {
        self.manifest.provider_generation
    }
    pub fn service_max_pairs(&self) -> u32 {
        self.manifest.service_max_pairs
    }
    pub fn domain(&self) -> u32 {
        self.manifest.domain
    }
    pub fn session(&self) -> u32 {
        self.manifest.session
    }
    pub fn service_schema_id(&self) -> u64 {
        self.manifest.service_schema_id
    }
    pub fn service_schema_sha256(&self) -> [u8; 32] {
        self.manifest.service_schema_sha256
    }
    pub fn total_verified_bytes(&self) -> u64 {
        self.total_bytes
    }
    /// The lease must preserve the artifacts; this only rechecks authority.
    pub fn revalidate_authority(&self) -> Result<(), DeploymentError> {
        if self.authority.admit(self.manifest) {
            Ok(())
        } else {
            Err(DeploymentError::NotAuthorized)
        }
    }
}
