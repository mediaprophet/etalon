use crate::{
    paths::{admit_path, open_regular},
    AdmittedDeployment, ArtifactSpec, DeploymentAuthority, DeploymentError, DeploymentManifest,
    ProviderKind, MAX_HASH_SCRATCH_BYTES,
};
use sha2::{Digest, Sha256};
use std::io::Read;

/// Validate exactly one ORT library and optional provider library. The budget
/// bounds artifact bytes hashed; one extra byte per file detects growth.
/// Scratch is caller-owned and is wiped on success, error and unwind.
pub fn validate_deployment<'m, 'p, 'a>(
    manifest: &'m DeploymentManifest<'p>,
    byte_budget: u64,
    scratch: &mut [u8],
    authority: &'a dyn DeploymentAuthority,
) -> Result<AdmittedDeployment<'m, 'p, 'a>, DeploymentError> {
    scratch.fill(0);
    let mut scratch = ClearScratch(scratch);
    admit_path(manifest.protected_root)?;
    admit_path(manifest.runtime.path)?;
    if manifest.target_id == 0
        || manifest.deployment_id.iter().all(|b| *b == 0)
        || manifest.runtime.bytes == 0
        || manifest.runtime.path == manifest.protected_root
        || !manifest.runtime.path.starts_with(manifest.protected_root)
        || (manifest.provider_kind == ProviderKind::QnnHtp && manifest.provider.is_none())
    {
        return Err(DeploymentError::InvalidManifest);
    }
    let total = match manifest.provider {
        Some(provider) => {
            admit_path(provider.path)?;
            if provider.bytes == 0
                || provider.path == manifest.runtime.path
                || provider.path == manifest.protected_root
                || !provider.path.starts_with(manifest.protected_root)
            {
                return Err(DeploymentError::InvalidManifest);
            }
            manifest
                .runtime
                .bytes
                .checked_add(provider.bytes)
                .ok_or(DeploymentError::InvalidBudget)?
        }
        None => manifest.runtime.bytes,
    };
    if byte_budget == 0 || total > byte_budget || scratch.0.is_empty() {
        return Err(DeploymentError::InvalidBudget);
    }
    if !authority.admit(manifest) {
        return Err(DeploymentError::NotAuthorized);
    }
    let capacity = scratch.0.len().min(MAX_HASH_SCRATCH_BYTES);
    verify(manifest.runtime, &mut scratch.0[..capacity])?;
    if let Some(provider) = manifest.provider {
        verify(provider, &mut scratch.0[..capacity])?;
    }
    if !authority.admit(manifest) {
        return Err(DeploymentError::NotAuthorized);
    }
    Ok(AdmittedDeployment {
        manifest,
        authority,
        total_bytes: total,
    })
}

pub(crate) fn verify(
    artifact: ArtifactSpec<'_>,
    scratch: &mut [u8],
) -> Result<(), DeploymentError> {
    let mut file = open_regular(artifact.path)?;
    if file.metadata()?.len() != artifact.bytes {
        return Err(DeploymentError::LengthMismatch);
    }
    verify_stream(&mut file, artifact, scratch)
}

fn verify_stream(
    file: &mut impl Read,
    artifact: ArtifactSpec<'_>,
    scratch: &mut [u8],
) -> Result<(), DeploymentError> {
    let mut remaining = artifact.bytes;
    let mut hash = Sha256::new();
    while remaining != 0 {
        let length = remaining.min(scratch.len() as u64) as usize;
        file.read_exact(&mut scratch[..length])?;
        hash.update(&scratch[..length]);
        remaining -= length as u64;
    }
    let mut tail = [0; 1];
    if file.read(&mut tail)? != 0 {
        return Err(DeploymentError::LengthMismatch);
    }
    let actual: [u8; 32] = hash.finalize().into();
    if actual != artifact.sha256 {
        return Err(DeploymentError::DigestMismatch);
    }
    Ok(())
}

struct ClearScratch<'a>(&'a mut [u8]);
impl Drop for ClearScratch<'_> {
    fn drop(&mut self) {
        self.0.fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Cursor, path::Path};

    #[test]
    fn exact_stream_digest_and_length_are_required() {
        let bytes = b"local-runtime";
        let mut spec = ArtifactSpec {
            path: Path::new("not-opened"),
            bytes: bytes.len() as u64,
            sha256: Sha256::digest(bytes).into(),
        };
        let mut scratch = [0; 3];
        assert!(verify_stream(&mut Cursor::new(bytes), spec, &mut scratch).is_ok());
        spec.sha256[0] ^= 1;
        assert!(matches!(
            verify_stream(&mut Cursor::new(bytes), spec, &mut scratch),
            Err(DeploymentError::DigestMismatch)
        ));
        spec.bytes -= 1;
        assert!(matches!(
            verify_stream(&mut Cursor::new(bytes), spec, &mut scratch),
            Err(DeploymentError::LengthMismatch)
        ));
    }

    struct Deny;
    impl DeploymentAuthority for Deny {
        fn admit(&self, _: &DeploymentManifest<'_>) -> bool {
            false
        }
    }

    #[test]
    fn budget_and_authority_refusals_do_not_open_artifacts_and_wipe_scratch() {
        let root = std::env::temp_dir();
        let path = root.join("qualia-deployment-test-not-opened.dll");
        let manifest = DeploymentManifest {
            deployment_id: [1; 32],
            target_id: 1,
            provider_kind: ProviderKind::OpenVinoNpu,
            protected_root: &root,
            runtime: ArtifactSpec {
                path: &path,
                bytes: 100,
                sha256: [2; 32],
            },
            provider: None,
        };
        let mut scratch = [9; 8];
        assert!(matches!(
            validate_deployment(&manifest, 99, &mut scratch, &Deny),
            Err(DeploymentError::InvalidBudget)
        ));
        assert_eq!(scratch, [0; 8]);
        assert!(matches!(
            validate_deployment(&manifest, 100, &mut scratch, &Deny),
            Err(DeploymentError::NotAuthorized)
        ));
        assert_eq!(scratch, [0; 8]);
    }

    #[test]
    fn scratch_is_cleared_when_verification_unwinds() {
        let mut bytes = [0; 16];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let guard = ClearScratch(&mut bytes);
            guard.0.fill(7);
            panic!("simulated authority failure after scratch use");
        }));
        assert!(result.is_err());
        assert_eq!(bytes, [0; 16]);
    }
}
