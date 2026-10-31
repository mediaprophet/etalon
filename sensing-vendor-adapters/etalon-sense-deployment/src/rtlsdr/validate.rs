use super::{
    AdmittedRtlSdrDeployment, RtlSdrAuthority, RtlSdrManifest, MAX_RTLSDR_DEPENDENCIES,
    MAX_RTLSDR_PAIRS,
};
use crate::{
    paths::admit_path, verify::verify, ArtifactSpec, DeploymentError, MAX_HASH_SCRATCH_BYTES,
};

/// Verify local files within a declared artifact-byte budget without loading.
pub fn validate_rtl_sdr_deployment<'m, 'p, 'a>(
    manifest: &'m RtlSdrManifest<'p>,
    byte_budget: u64,
    scratch: &mut [u8],
    authority: &'a dyn RtlSdrAuthority,
) -> Result<AdmittedRtlSdrDeployment<'m, 'p, 'a>, DeploymentError> {
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
    Ok(AdmittedRtlSdrDeployment {
        manifest,
        authority,
        total_bytes,
    })
}

fn validate_manifest(manifest: &RtlSdrManifest<'_>) -> Result<u64, DeploymentError> {
    admit_path(manifest.protected_root)?;
    if manifest.deployment_id == [0; 32]
        || manifest.target_id == 0
        || manifest.generation == 0
        || manifest.max_pairs == 0
        || manifest.max_pairs > MAX_RTLSDR_PAIRS
        || manifest.dependencies.len() > MAX_RTLSDR_DEPENDENCIES
    {
        return Err(DeploymentError::InvalidManifest);
    }
    let mut bytes = 0u64;
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
        bytes = bytes
            .checked_add(artifact.bytes)
            .ok_or(DeploymentError::InvalidBudget)?;
    }
    Ok(bytes)
}

fn artifacts<'a, 'p>(
    manifest: &'a RtlSdrManifest<'p>,
) -> impl Iterator<Item = ArtifactSpec<'p>> + 'a
where
    'p: 'a,
{
    core::iter::once(manifest.runtime).chain(manifest.dependencies.iter().copied())
}

struct ClearScratch<'a>(&'a mut [u8]);
impl Drop for ClearScratch<'_> {
    fn drop(&mut self) {
        self.0.fill(0);
    }
}

#[cfg(test)]
mod tests;
