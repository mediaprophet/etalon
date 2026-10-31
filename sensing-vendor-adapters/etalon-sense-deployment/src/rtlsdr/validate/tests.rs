use super::*;
use std::{cell::Cell, path::Path};

fn declaration<'a>(root: &'a Path, runtime: &'a Path) -> RtlSdrManifest<'a> {
    RtlSdrManifest {
        deployment_id: [1; 32],
        target_id: 7,
        generation: 3,
        device_index: 2,
        max_pairs: 512,
        protected_root: root,
        runtime: ArtifactSpec {
            path: runtime,
            bytes: 100,
            sha256: [2; 32],
        },
        dependencies: &[],
    }
}

struct Deny(Cell<u32>);
impl RtlSdrAuthority for Deny {
    fn admit(&self, _: &RtlSdrManifest<'_>) -> bool {
        self.0.set(self.0.get() + 1);
        false
    }
}

#[test]
fn budget_and_policy_refusals_precede_artifact_io_and_clear_scratch() {
    let root = std::env::temp_dir();
    let path = root.join("qualia-rtlsdr-not-opened-library");
    let manifest = declaration(&root, &path);
    let authority = Deny(Cell::new(0));
    let mut scratch = [0xaa; 8];
    assert!(matches!(
        validate_rtl_sdr_deployment(&manifest, 99, &mut scratch, &authority),
        Err(DeploymentError::InvalidBudget)
    ));
    assert_eq!(authority.0.get(), 0);
    assert_eq!(scratch, [0; 8]);
    scratch.fill(0xaa);
    assert!(matches!(
        validate_rtl_sdr_deployment(&manifest, 100, &mut scratch, &authority),
        Err(DeploymentError::NotAuthorized)
    ));
    assert_eq!(authority.0.get(), 1);
    assert_eq!(scratch, [0; 8]);
}

#[test]
fn invalid_closure_and_capacity_do_not_reach_the_authority() {
    let root = std::env::temp_dir();
    let path = root.join("qualia-rtlsdr-not-opened-library");
    let mut manifest = declaration(&root, &path);
    let duplicate = [manifest.runtime];
    manifest.dependencies = &duplicate;
    assert!(matches!(
        validate_manifest(&manifest),
        Err(DeploymentError::InvalidManifest)
    ));
    manifest.dependencies = &[];
    manifest.generation = 0;
    assert!(matches!(
        validate_manifest(&manifest),
        Err(DeploymentError::InvalidManifest)
    ));
    manifest.generation = 3;
    manifest.max_pairs = MAX_RTLSDR_PAIRS + 1;
    assert!(matches!(
        validate_manifest(&manifest),
        Err(DeploymentError::InvalidManifest)
    ));
    manifest.max_pairs = 1;
    manifest.runtime.path = &root;
    assert!(matches!(
        validate_manifest(&manifest),
        Err(DeploymentError::InvalidManifest)
    ));
}

#[test]
fn token_keeps_exact_selection_and_observes_lease_revocation() {
    struct Lease(Cell<bool>);
    impl RtlSdrAuthority for Lease {
        fn admit(&self, _: &RtlSdrManifest<'_>) -> bool {
            self.0.get()
        }
    }
    let root = std::env::temp_dir();
    let path = root.join("qualia-rtlsdr-not-opened-library");
    let manifest = declaration(&root, &path);
    let authority = Lease(Cell::new(true));
    let admitted = AdmittedRtlSdrDeployment {
        manifest: &manifest,
        authority: &authority,
        total_bytes: 100,
    };
    assert_eq!(
        (
            admitted.target_id(),
            admitted.generation(),
            admitted.device_index(),
            admitted.max_pairs()
        ),
        (7, 3, 2, 512)
    );
    assert!(admitted.revalidate_authority().is_ok());
    authority.0.set(false);
    assert!(matches!(
        admitted.revalidate_authority(),
        Err(DeploymentError::NotAuthorized)
    ));
}
