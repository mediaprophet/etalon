#![cfg(windows)]
use super::*;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    os::windows::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    sync::{atomic::AtomicBool, Arc},
};
#[path = "allocation_counter.rs"]
mod allocation_counter;
fn dir() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("lease-fixture-")
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
        max_total_bytes: n.checked_mul(2).unwrap(),
    }
}
fn fixture(parent: &Path) -> PathBuf {
    let p = parent.join("model.bin");
    fs::write(&p, [7u8; 4096]).unwrap();
    p
}
fn open(path: &Path) -> ModelLease {
    ModelLease::open(
        path,
        digest(&[7; 4096]),
        limits(4096),
        &AtomicBool::new(false),
    )
    .unwrap()
}
fn writable(path: &Path) -> bool {
    OpenOptions::new().write(true).open(path).is_ok()
}

#[test]
fn writer_exclusion_persists_through_last_arc_and_all_aliases() {
    let d = dir();
    let path = fixture(d.path());
    let alias = d.path().join("alias");
    fs::hard_link(&path, &alias).unwrap();
    let lease = Arc::new(open(&path));
    let clone = Arc::clone(&lease);
    assert_eq!(
        lease.identity(),
        LeaseIdentity {
            byte_len: 4096,
            content_sha256: digest(&[7; 4096])
        }
    );
    assert_eq!(lease.bytes(), &[7; 4096]);
    for p in [&path, &alias] {
        assert!(!writable(p));
        assert!(OpenOptions::new().append(true).open(p).is_err());
        assert!(OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(p)
            .is_err());
    }
    assert!(fs::remove_file(&path).is_err());
    assert!(fs::rename(&path, d.path().join("moved")).is_err());
    // Secondary namespace entries are not an immutable identity. Actual Windows
    // permits deleting the alias; its replacement must not affect leased bytes.
    fs::remove_file(&alias).unwrap();
    fs::write(&alias, b"independent replacement").unwrap();
    assert!(writable(&alias));
    assert_eq!(lease.bytes(), &[7; 4096]);
    assert_eq!(lease.identity().content_sha256, digest(lease.bytes()));
    let replacement = d.path().join("replacement");
    fs::write(&replacement, b"replacement").unwrap();
    assert!(fs::rename(&replacement, &path).is_err());
    drop(lease);
    assert!(!writable(&path));
    assert_eq!(clone.bytes()[0], 7);
    drop(clone);
    assert!(writable(&path));
    assert!(writable(&alias));
    let retained = d.path().to_path_buf();
    drop(d);
    assert!(!retained.exists());
}
#[test]
fn preexisting_write_delete_and_closed_file_writable_mapping_refuse() {
    let d = dir();
    let path = fixture(d.path());
    let writer = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    let attempt = || {
        ModelLease::open(
            &path,
            digest(&[7; 4096]),
            limits(4096),
            &AtomicBool::new(false),
        )
    };
    assert!(attempt().is_err());
    // No immutable view exists during this probe; lease must refuse before mmap.
    let mut mapping = unsafe { memmap2::MmapOptions::new().map_mut(&writer).unwrap() };
    drop(writer);
    assert!(attempt().is_err());
    mapping[0] = 7;
    mapping.flush().unwrap();
    drop(mapping);
    let delete = OpenOptions::new()
        .access_mode(0x0001_0000)
        .share_mode(7)
        .open(&path)
        .unwrap();
    assert!(attempt().is_err());
    drop(delete);
    drop(attempt().unwrap());
    assert!(writable(&path));
}
#[test]
fn independent_readers_share_bytes_but_cannot_create_mutable_source_views() {
    let d = dir();
    let path = fixture(d.path());
    let reader = fs::File::open(&path).unwrap();
    // Test owns this fixture and holds no mutating handle while read views live.
    let readmap = unsafe { memmap2::MmapOptions::new().map(&reader).unwrap() };
    let lease = open(&path);
    let second = open(&path);
    assert_eq!(&readmap[..], lease.bytes());
    assert_eq!(second.identity(), lease.identity());
    assert!(unsafe { memmap2::MmapOptions::new().map_mut(&reader) }.is_err());
    drop(reader);
    drop(readmap);
    drop(lease);
    assert!(!writable(&path));
    drop(second);
    assert!(writable(&path));
}
#[test]
fn exact_length_total_budget_full_content_and_checked_arithmetic() {
    let d = dir();
    let path = fixture(d.path());
    let cancelled = AtomicBool::new(false);
    for l in [
        LeaseLimits {
            max_source_bytes: 4095,
            ..limits(4096)
        },
        LeaseLimits {
            max_mapped_bytes: 4095,
            ..limits(4096)
        },
        LeaseLimits {
            max_total_bytes: 8191,
            ..limits(4096)
        },
    ] {
        assert!(matches!(
            ModelLease::open(&path, digest(&[7; 4096]), l, &cancelled),
            Err(LeaseError::Budget)
        ));
        assert!(writable(&path));
    }
    assert!(limits(1).admit(0).is_err());
    assert!(LeaseLimits {
        max_source_bytes: u64::MAX,
        max_mapped_bytes: u64::MAX,
        max_total_bytes: u64::MAX
    }
    .admit(u64::MAX)
    .is_err());
    assert!(ModelLease::open(&path, [0; 32], limits(4096), &cancelled).is_err());
    assert!(matches!(
        ModelLease::open(&path, [1; 32], limits(4096), &cancelled),
        Err(LeaseError::ContentMismatch)
    ));
    assert!(writable(&path));
    let lease = open(&path);
    assert_eq!(lease.identity().byte_len, 4096);
    drop(lease);
    let mut changed = [7u8; 4096];
    changed[4095] = 8;
    fs::write(&path, changed).unwrap();
    // Same path/length/header, different actual weight byte: old identity refuses.
    assert!(matches!(
        ModelLease::open(&path, digest(&[7; 4096]), limits(4096), &cancelled),
        Err(LeaseError::ContentMismatch)
    ));
    assert!(ModelLease::open(&path, digest(&changed), limits(4096), &cancelled).is_ok());
    fs::write(&path, []).unwrap();
    assert!(ModelLease::open(&path, digest(&[]), limits(4096), &cancelled).is_err());
}
#[test]
fn error_cancellation_and_unwind_release_every_owned_handle() {
    let d = dir();
    let path = fixture(d.path());
    let before = fs::read_dir(d.path()).unwrap().count();
    for point in 1..=4 {
        let cancelled = AtomicBool::new(false);
        super::backing::FAULT.set(point);
        let result = std::panic::catch_unwind(|| {
            ModelLease::open(&path, digest(&[7; 4096]), limits(4096), &cancelled)
        });
        super::backing::FAULT.set(0);
        if point == 3 {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_err());
        }
        assert!(writable(&path));
        assert_eq!(fs::read_dir(d.path()).unwrap().count(), before);
    }
    assert!(matches!(
        ModelLease::open(
            &path,
            digest(&[7; 4096]),
            limits(4096),
            &AtomicBool::new(true)
        ),
        Err(LeaseError::Cancelled)
    ));
    let retained = d.path().to_path_buf();
    drop(d);
    assert!(!retained.exists());
}
#[test]
fn concurrent_growth_and_truncation_cannot_change_leased_bytes() {
    let d = dir();
    let path = fixture(d.path());
    let lease = Arc::new(open(&path));
    let copy = Arc::clone(&lease);
    let p = path.clone();
    let worker = std::thread::spawn(move || {
        assert!(OpenOptions::new().append(true).open(&p).is_err());
        assert!(OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&p)
            .is_err());
        assert_eq!(copy.identity().content_sha256, digest(copy.bytes()));
    });
    worker.join().unwrap();
    assert_eq!(lease.bytes(), &[7; 4096]);
    drop(lease);
    OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&path)
        .unwrap();
    assert_eq!(fs::metadata(&path).unwrap().len(), 0);
}
#[test]
fn resource_bundle_binds_role_version_order_length_and_complete_content() {
    let d = dir();
    let path = fixture(d.path());
    let lease = open(&path);
    let lim = BundleLimits {
        max_parts: 8,
        max_source_bytes: 32768,
    };
    let parts = |role, format_version, ordinal| {
        [
            ModelPart {
                role: ModelPartRole::BaseModel,
                format_version: 1,
                ordinal: 0,
                lease: &lease,
            },
            ModelPart {
                role,
                format_version,
                ordinal,
                lease: &lease,
            },
        ]
    };
    let baseline = bundle_digest(&parts(ModelPartRole::LoraAdapter, 1, 0), lim).unwrap();
    for role in [
        ModelPartRole::Shard,
        ModelPartRole::Tokenizer,
        ModelPartRole::Configuration,
        ModelPartRole::HelperMetadata,
        ModelPartRole::DraftModel,
    ] {
        assert_ne!(bundle_digest(&parts(role, 1, 0), lim).unwrap(), baseline);
    }
    assert_ne!(
        bundle_digest(&parts(ModelPartRole::LoraAdapter, 2, 0), lim).unwrap(),
        baseline
    );
    assert_ne!(
        bundle_digest(&parts(ModelPartRole::LoraAdapter, 1, 1), lim).unwrap(),
        baseline
    );
    assert!(bundle_digest(&parts(ModelPartRole::LoraAdapter, 0, 0), lim).is_err());
    assert!(bundle_digest(&parts(ModelPartRole::BaseModel, 1, 0), lim).is_err());
    assert!(bundle_digest(
        &parts(ModelPartRole::LoraAdapter, 1, 0),
        BundleLimits {
            max_parts: 1,
            ..lim
        }
    )
    .is_err());
    assert!(bundle_digest(
        &parts(ModelPartRole::LoraAdapter, 1, 0),
        BundleLimits {
            max_source_bytes: 8191,
            ..lim
        }
    )
    .is_err());
    let changed = d.path().join("lora");
    let mut content = [7; 4096];
    content[4] = 8;
    fs::write(&changed, content).unwrap();
    let other = ModelLease::open(
        &changed,
        digest(&content),
        limits(4096),
        &AtomicBool::new(false),
    )
    .unwrap();
    let mut mutated = parts(ModelPartRole::LoraAdapter, 1, 0);
    mutated[1].lease = &other;
    assert_ne!(
        bundle_digest(&mutated, lim).unwrap(),
        baseline,
        "LoRA header must bind"
    );
    let short = d.path().join("short");
    fs::write(&short, [7u8; 64]).unwrap();
    let shorter = ModelLease::open(
        &short,
        digest(&[7; 64]),
        limits(64),
        &AtomicBool::new(false),
    )
    .unwrap();
    mutated[1].lease = &shorter;
    assert_ne!(bundle_digest(&mutated, lim).unwrap(), baseline);
    let reversed = [
        ModelPart {
            role: ModelPartRole::BaseModel,
            format_version: 1,
            ordinal: 0,
            lease: &lease,
        },
        ModelPart {
            role: ModelPartRole::Configuration,
            format_version: 1,
            ordinal: 0,
            lease: &lease,
        },
        ModelPart {
            role: ModelPartRole::Tokenizer,
            format_version: 1,
            ordinal: 0,
            lease: &lease,
        },
    ];
    assert!(bundle_digest(&reversed, lim).is_err());
    assert!(bundle_digest(&[], lim).is_err());
}
#[test]
fn hot_borrow_identity_bundle_and_arc_clone_have_zero_allocation() {
    use allocation_counter::{assert_zero_alloc, AllocGuard};
    let control = AllocGuard::begin("positive control", true);
    let allocation = vec![7u8; 32];
    std::hint::black_box(&allocation);
    assert!(control.check().is_err());
    drop(allocation);
    let d = dir();
    let path = fixture(d.path());
    let lease = Arc::new(open(&path));
    assert_zero_alloc("live backing access", || {
        let clone = Arc::clone(&lease);
        assert_eq!(clone.bytes().len(), 4096);
        std::hint::black_box(clone.identity());
        let parts = [ModelPart {
            role: ModelPartRole::BaseModel,
            format_version: 1,
            ordinal: 0,
            lease: &clone,
        }];
        assert!(bundle_digest(
            &parts,
            BundleLimits {
                max_parts: 1,
                max_source_bytes: 4096
            }
        )
        .is_ok());
        assert!(bundle_digest(
            &parts,
            BundleLimits {
                max_parts: 0,
                max_source_bytes: 4096
            }
        )
        .is_err());
        drop(clone);
    });
}
