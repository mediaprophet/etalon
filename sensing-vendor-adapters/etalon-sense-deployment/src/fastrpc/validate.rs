use super::{
    AdmittedFastRpcDeployment, FastRpcAuthority, FastRpcManifest, MAX_FASTRPC_CLOSURE_ARTIFACTS,
    MAX_FASTRPC_MODULE_URI_BYTES, MAX_FASTRPC_SERVICE_PAIRS,
};
use crate::{
    paths::admit_path, verify::verify, ArtifactSpec, DeploymentError, MAX_HASH_SCRATCH_BYTES,
};

/// Hash admitted local files without loading, installing or executing them.
/// Checks declared aggregate bytes before I/O; probes one extra byte per file.
pub fn validate_fast_rpc_deployment<'m, 'p, 'a>(
    manifest: &'m FastRpcManifest<'p>,
    byte_budget: u64,
    scratch: &mut [u8],
    authority: &'a dyn FastRpcAuthority,
) -> Result<AdmittedFastRpcDeployment<'m, 'p, 'a>, DeploymentError> {
    scratch.fill(0);
    let scratch = ClearScratch(scratch);
    let total_bytes = validate_manifest(manifest)?;
    if byte_budget == 0 || total_bytes > byte_budget || scratch.0.is_empty() {
        return Err(DeploymentError::InvalidBudget);
    }
    if !authority.admit(manifest) {
        return Err(DeploymentError::NotAuthorized);
    }
    let capacity = scratch.0.len().min(MAX_HASH_SCRATCH_BYTES);
    for artifact in artifacts(manifest) {
        verify(artifact, &mut scratch.0[..capacity])?;
    }
    if !authority.admit(manifest) {
        return Err(DeploymentError::NotAuthorized);
    }
    Ok(AdmittedFastRpcDeployment {
        manifest,
        authority,
        total_bytes,
    })
}

fn validate_manifest(manifest: &FastRpcManifest<'_>) -> Result<u64, DeploymentError> {
    admit_path(manifest.protected_root)?;
    let uri = manifest.module_uri.to_bytes();
    if manifest.target_id == 0
        || manifest.deployment_id == [0; 32]
        || manifest.provider_generation == 0
        || manifest.service_max_pairs == 0
        || manifest.service_max_pairs > MAX_FASTRPC_SERVICE_PAIRS
        || manifest.service_schema_id == 0
        || manifest.service_schema_sha256 == [0; 32]
        || manifest.firmware_and_dependencies.len() > MAX_FASTRPC_CLOSURE_ARTIFACTS
        || uri.is_empty()
        || uri.len() > MAX_FASTRPC_MODULE_URI_BYTES
        || uri.iter().any(|byte| !byte.is_ascii_graphic())
    {
        return Err(DeploymentError::InvalidManifest);
    }
    let mut total = 0u64;
    for (index, artifact) in artifacts(manifest).enumerate() {
        admit_path(artifact.path)?;
        if artifact.bytes == 0
            || artifact.sha256 == [0; 32]
            || artifact.path == manifest.protected_root
            || !artifact.path.starts_with(manifest.protected_root)
            || artifacts(manifest)
                .take(index)
                .any(|prior| prior.path == artifact.path)
        {
            return Err(DeploymentError::InvalidManifest);
        }
        total = total
            .checked_add(artifact.bytes)
            .ok_or(DeploymentError::InvalidBudget)?;
    }
    Ok(total)
}

fn artifacts<'a, 'p>(
    manifest: &'a FastRpcManifest<'p>,
) -> impl Iterator<Item = ArtifactSpec<'p>> + 'a
where
    'p: 'a,
{
    [manifest.runtime, manifest.module]
        .into_iter()
        .chain(manifest.firmware_and_dependencies.iter().copied())
}

struct ClearScratch<'a>(&'a mut [u8]);
impl Drop for ClearScratch<'_> {
    fn drop(&mut self) {
        self.0.fill(0);
    }
}

#[cfg(test)]
mod tests;
