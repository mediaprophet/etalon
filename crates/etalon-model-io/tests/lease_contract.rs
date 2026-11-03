#![cfg(windows)]

use etalon_model_io::lease::*;
use sha2::{Digest, Sha256};
use std::{
    fs,
    fs::OpenOptions,
    path::Path,
    sync::{atomic::AtomicBool, Arc},
};
#[path = "support/lease_allocation.rs"]
mod allocation_counter;

fn dir() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("independent-lease-")
        .tempdir_in(env!("CARGO_MANIFEST_DIR"))
        .unwrap()
}
fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn limits(n: u64) -> LeaseLimits {
    LeaseLimits {
        max_source_bytes: n,
        max_mapped_bytes: n,
        max_total_bytes: n * 2,
    }
}
fn open(path: &Path, bytes: &[u8]) -> ModelLease {
    ModelLease::open(
        path,
        digest(bytes),
        limits(bytes.len() as u64),
        &AtomicBool::new(false),
    )
    .unwrap()
}
fn writable(path: &Path) -> bool {
    OpenOptions::new().write(true).open(path).is_ok()
}

#[test]
fn independent_full_multichunk_tail_hash_and_each_admission_limit() {
    let d = dir();
    let p = d.path().join("source");
    let mut bytes = vec![0u8; 131073];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = (i.wrapping_mul(71) % 251) as u8;
    }
    fs::write(&p, &bytes).unwrap();
    let expected = digest(&bytes);
    let lease = open(&p, &bytes);
    assert_eq!(lease.bytes(), bytes);
    assert_eq!(lease.identity().byte_len, 131073);
    drop(lease);
    for kind in 0..3 {
        let mut cap = limits(bytes.len() as u64);
        match kind {
            0 => cap.max_source_bytes -= 1,
            1 => cap.max_mapped_bytes -= 1,
            _ => cap.max_total_bytes -= 1,
        }
        assert!(matches!(
            ModelLease::open(&p, expected, cap, &AtomicBool::new(false)),
            Err(LeaseError::Budget)
        ));
        assert!(writable(&p));
    }
    // Same header, same length, different last byte must fail full-file identity.
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(&p, &bytes).unwrap();
    assert!(matches!(
        ModelLease::open(
            &p,
            expected,
            limits(bytes.len() as u64),
            &AtomicBool::new(false)
        ),
        Err(LeaseError::ContentMismatch)
    ));
    assert!(writable(&p));
    assert!(matches!(
        ModelLease::open(
            &p,
            digest(&bytes),
            limits(bytes.len() as u64),
            &AtomicBool::new(true)
        ),
        Err(LeaseError::Cancelled)
    ));
    assert!(matches!(
        ModelLease::open(
            &p,
            [0; 32],
            limits(bytes.len() as u64),
            &AtomicBool::new(false)
        ),
        Err(LeaseError::InvalidIdentity)
    ));
    assert!(writable(&p));
}

#[test]
fn independent_actual_alias_arc_exclusion_and_normal_unwind_release() {
    let d = dir();
    let p = d.path().join("source");
    let alias = d.path().join("alias");
    fs::write(&p, b"immutable source").unwrap();
    fs::hard_link(&p, &alias).unwrap();
    let owner = Arc::new(open(&p, b"immutable source"));
    let retained = owner.clone();
    assert!(!writable(&p));
    assert!(!writable(&alias));
    drop(owner);
    assert!(!writable(&alias));
    assert_eq!(retained.bytes(), b"immutable source");
    drop(retained);
    assert!(writable(&alias));
    let result = std::panic::catch_unwind(|| {
        let lease = open(&p, b"immutable source");
        assert!(!writable(&alias));
        std::hint::black_box(lease.bytes());
        panic!("independent caller unwind");
    });
    assert!(result.is_err());
    assert!(writable(&p));
    assert!(writable(&alias));
    let owned = d.path().to_path_buf();
    d.close().unwrap();
    assert!(!owned.exists());
}

#[test]
fn independent_open_writer_and_closed_file_writable_mapping_refuse() {
    use std::os::windows::fs::OpenOptionsExt;
    let d = dir();
    let p = d.path().join("source");
    fs::write(&p, [9u8; 4096]).unwrap();
    let writer = OpenOptions::new()
        .read(true)
        .write(true)
        .share_mode(7)
        .open(&p)
        .unwrap();
    assert!(ModelLease::open(
        &p,
        digest(&[9; 4096]),
        limits(4096),
        &AtomicBool::new(false)
    )
    .is_err());
    // Actual writable view remains after its original File is closed.
    let mapping = unsafe { memmap2::MmapOptions::new().map_mut(&writer) }.unwrap();
    drop(writer);
    assert!(ModelLease::open(
        &p,
        digest(&[9; 4096]),
        limits(4096),
        &AtomicBool::new(false)
    )
    .is_err());
    drop(mapping);
    let lease = open(&p, &[9; 4096]);
    assert_eq!(lease.bytes(), &[9; 4096]);
    drop(lease);
    assert!(writable(&p));
}

#[test]
fn independent_bundle_manual_encoding_and_all_resource_fields() {
    let d = dir();
    let p = d.path().join("base");
    let q = d.path().join("helper");
    fs::write(&p, b"base bytes").unwrap();
    fs::write(&q, b"helper bytes").unwrap();
    let base = open(&p, b"base bytes");
    let helper = open(&q, b"helper bytes");
    let parts = [
        ModelPart {
            role: ModelPartRole::BaseModel,
            format_version: 4,
            ordinal: 0,
            lease: &base,
        },
        ModelPart {
            role: ModelPartRole::HelperMetadata,
            format_version: 8,
            ordinal: 2,
            lease: &helper,
        },
    ];
    let cap = BundleLimits {
        max_parts: 2,
        max_source_bytes: 22,
    };
    let actual = bundle_digest(&parts, cap).unwrap();
    let mut raw = Vec::new();
    raw.extend_from_slice(b"qualia:g3:immutable-model-bundle:v1\0");
    raw.extend_from_slice(&1u32.to_le_bytes());
    raw.extend_from_slice(&2u32.to_le_bytes());
    for (role, version, ordinal, bytes) in [
        (1u8, 4u32, 0u32, &b"base bytes"[..]),
        (5, 8, 2, &b"helper bytes"[..]),
    ] {
        raw.push(role);
        raw.extend_from_slice(&version.to_le_bytes());
        raw.extend_from_slice(&ordinal.to_le_bytes());
        raw.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        raw.extend_from_slice(&digest(bytes));
    }
    assert_eq!(actual, digest(&raw));
    for changed in 0..4 {
        let mut rows = [
            ModelPart {
                role: ModelPartRole::BaseModel,
                format_version: 4,
                ordinal: 0,
                lease: &base,
            },
            ModelPart {
                role: ModelPartRole::HelperMetadata,
                format_version: 8,
                ordinal: 2,
                lease: &helper,
            },
        ];
        match changed {
            0 => rows[1].role = ModelPartRole::LoraAdapter,
            1 => rows[1].format_version = 9,
            2 => rows[1].ordinal = 3,
            _ => rows[1].lease = &base,
        }
        assert_ne!(bundle_digest(&rows, cap).unwrap(), actual);
    }
    assert!(bundle_digest(
        &parts,
        BundleLimits {
            max_parts: 1,
            ..cap
        }
    )
    .is_err());
    assert!(bundle_digest(
        &parts,
        BundleLimits {
            max_source_bytes: 21,
            ..cap
        }
    )
    .is_err());
    let reversed = [
        ModelPart {
            role: ModelPartRole::HelperMetadata,
            format_version: 8,
            ordinal: 2,
            lease: &helper,
        },
        ModelPart {
            role: ModelPartRole::BaseModel,
            format_version: 4,
            ordinal: 0,
            lease: &base,
        },
    ];
    assert!(bundle_digest(&reversed, cap).is_err());
    assert!(bundle_digest(&[], cap).is_err());
}

#[test]
fn independent_borrow_and_bundle_success_refusal_zero_allocation_calls() {
    use allocation_counter::AllocGuard;
    let positive = AllocGuard::begin("positive", true);
    let allocated = vec![1u8; 73];
    std::hint::black_box(&allocated);
    assert!(positive.check().is_err());
    drop(allocated);
    let d = dir();
    let p = d.path().join("source");
    fs::write(&p, b"hot bytes").unwrap();
    let lease = Arc::new(open(&p, b"hot bytes"));
    let parts = [ModelPart {
        role: ModelPartRole::BaseModel,
        format_version: 1,
        ordinal: 0,
        lease: &lease,
    }];
    let measured = AllocGuard::begin("borrow and bundle", true);
    for _ in 0..64 {
        let clone = lease.clone();
        assert_eq!(clone.bytes(), b"hot bytes");
        assert_eq!(clone.identity().byte_len, 9);
        assert!(bundle_digest(
            &parts,
            BundleLimits {
                max_parts: 1,
                max_source_bytes: 9
            }
        )
        .is_ok());
        assert!(bundle_digest(
            &parts,
            BundleLimits {
                max_parts: 1,
                max_source_bytes: 8
            }
        )
        .is_err());
    }
    assert!(measured.check().is_ok());
}
