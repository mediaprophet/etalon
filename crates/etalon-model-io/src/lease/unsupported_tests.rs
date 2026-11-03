use super::{LeaseError, LeaseLimits, ModelLease};
use std::sync::atomic::AtomicBool;

#[test]
fn unsupported_platform_refuses_without_touching_requested_path() {
    let fixture = tempfile::tempdir().unwrap();
    let path = fixture.path().join("absent-model.bin");
    let result = ModelLease::open(
        &path,
        [1; 32],
        LeaseLimits {
            max_source_bytes: 1,
            max_mapped_bytes: 1,
            max_total_bytes: 2,
        },
        &AtomicBool::new(false),
    );
    assert!(matches!(result, Err(LeaseError::UnsupportedPlatform)));
    assert!(!path.exists());
    assert_eq!(std::fs::read_dir(fixture.path()).unwrap().count(), 0);
}
