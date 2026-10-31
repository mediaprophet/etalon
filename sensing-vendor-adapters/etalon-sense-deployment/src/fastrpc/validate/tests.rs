use super::*;
use std::{cell::Cell, path::Path};

struct Deny(Cell<usize>);
impl FastRpcAuthority for Deny {
    fn admit(&self, _: &FastRpcManifest<'_>) -> bool {
        self.0.set(self.0.get() + 1);
        false
    }
}

fn manifest<'a>(root: &'a Path, runtime: &'a Path, module: &'a Path) -> FastRpcManifest<'a> {
    FastRpcManifest {
        deployment_id: [1; 32],
        target_id: 1,
        provider_generation: 1,
        service_max_pairs: 8,
        protected_root: root,
        runtime: ArtifactSpec {
            path: runtime,
            bytes: 11,
            sha256: [2; 32],
        },
        module: ArtifactSpec {
            path: module,
            bytes: 13,
            sha256: [3; 32],
        },
        firmware_and_dependencies: &[],
        module_uri: c"file:///sense_skel.so?invoke&_dom=cdsp",
        domain: 3,
        session: 0,
        service_schema_id: 1,
        service_schema_sha256: [4; 32],
    }
}

#[test]
fn authority_and_budget_rejections_precede_file_reads() {
    let root = std::env::temp_dir();
    let runtime = root.join("qualia-fastrpc-not-opened.so");
    let module = root.join("qualia-skeleton-not-opened.so");
    let manifest = manifest(&root, &runtime, &module);
    let authority = Deny(Cell::new(0));
    let mut scratch = [9; 8];
    assert!(matches!(
        validate_fast_rpc_deployment(&manifest, 23, &mut scratch, &authority),
        Err(DeploymentError::InvalidBudget)
    ));
    assert_eq!(authority.0.get(), 0);
    assert_eq!(scratch, [0; 8]);
    assert!(matches!(
        validate_fast_rpc_deployment(&manifest, 24, &mut scratch, &authority),
        Err(DeploymentError::NotAuthorized)
    ));
    assert_eq!(authority.0.get(), 1);
    assert_eq!(scratch, [0; 8]);
}

#[test]
fn incomplete_identity_and_repeated_or_outside_artifacts_are_rejected() {
    let root = std::env::temp_dir();
    let runtime = root.join("qualia-fastrpc-not-opened.so");
    let module = root.join("qualia-skeleton-not-opened.so");
    let mut manifest = manifest(&root, &runtime, &module);
    manifest.service_schema_sha256 = [0; 32];
    assert!(matches!(
        validate_manifest(&manifest),
        Err(DeploymentError::InvalidManifest)
    ));
    manifest.service_schema_sha256 = [4; 32];
    manifest.module.path = &runtime;
    assert!(matches!(
        validate_manifest(&manifest),
        Err(DeploymentError::InvalidManifest)
    ));
    manifest.module.path = &module;
    let duplicate = [manifest.runtime];
    manifest.firmware_and_dependencies = &duplicate;
    assert!(matches!(
        validate_manifest(&manifest),
        Err(DeploymentError::InvalidManifest)
    ));
    manifest.firmware_and_dependencies = &[];
    manifest.module.path = &root;
    assert!(matches!(
        validate_manifest(&manifest),
        Err(DeploymentError::InvalidManifest)
    ));
}

#[test]
fn lease_revocation_is_observed_by_the_admitted_token() {
    struct Lease(Cell<bool>);
    impl FastRpcAuthority for Lease {
        fn admit(&self, _: &FastRpcManifest<'_>) -> bool {
            self.0.get()
        }
    }
    let root = std::env::temp_dir();
    let runtime = root.join("qualia-fastrpc-not-opened.so");
    let module = root.join("qualia-skeleton-not-opened.so");
    let manifest = manifest(&root, &runtime, &module);
    let lease = Lease(Cell::new(true));
    let token = AdmittedFastRpcDeployment {
        manifest: &manifest,
        authority: &lease,
        total_bytes: 24,
    };
    assert!(token.revalidate_authority().is_ok());
    lease.0.set(false);
    assert!(matches!(
        token.revalidate_authority(),
        Err(DeploymentError::NotAuthorized)
    ));
}

#[test]
fn image_configuration_must_be_complete_before_admission() {
    let root = std::env::temp_dir();
    let runtime = root.join("qualia-fastrpc-not-opened.so");
    let module = root.join("qualia-skeleton-not-opened.so");
    let mut declaration = manifest(&root, &runtime, &module);
    declaration.provider_generation = 0;
    assert!(matches!(
        validate_manifest(&declaration),
        Err(DeploymentError::InvalidManifest)
    ));
    declaration.provider_generation = 1;
    declaration.service_max_pairs = 0;
    assert!(matches!(
        validate_manifest(&declaration),
        Err(DeploymentError::InvalidManifest)
    ));
    declaration.service_max_pairs = MAX_FASTRPC_SERVICE_PAIRS + 1;
    assert!(matches!(
        validate_manifest(&declaration),
        Err(DeploymentError::InvalidManifest)
    ));
}
