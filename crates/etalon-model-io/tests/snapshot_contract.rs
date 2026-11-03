//! Public snapshot ownership tests using only privately owned local fixtures.
#![cfg(windows)]
use etalon_model_io::{
    lease::LeaseIdentity,
    snapshot::{Snapshot, SnapshotErrorKind, SnapshotLimits, COPY_BUFFER_BYTES},
};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read},
    os::windows::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
#[path = "support/lease_allocation.rs"]
mod allocation;

fn parent() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("independent-snapshot-")
        .tempdir_in(env!("CARGO_MANIFEST_DIR"))
        .unwrap()
}
fn identity(bytes: &[u8]) -> LeaseIdentity {
    LeaseIdentity {
        content_sha256: Sha256::digest(bytes).into(),
        byte_len: bytes.len() as u64,
    }
}
fn limits(n: u64) -> SnapshotLimits {
    SnapshotLimits {
        max_source_bytes: n + 1,
        max_disk_bytes: n,
        max_mapped_bytes: n,
        max_total_bytes: 3 * n + 1 + COPY_BUFFER_BYTES as u64,
        max_read_calls: 100,
        max_write_calls: 100,
    }
}
fn empty(parent: &Path) {
    assert_eq!(fs::read_dir(parent).unwrap().count(), 0);
}
fn child(parent: &Path) -> PathBuf {
    let mut entries = fs::read_dir(parent).unwrap();
    let path = entries.next().unwrap().unwrap().path();
    assert!(entries.next().is_none());
    assert_eq!(path.parent(), Some(parent));
    assert!(path
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("etalon-model-snapshot-"));
    path
}
struct NeverRead;
impl Read for NeverRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        panic!("preflight must not read source")
    }
}

#[test]
fn every_preflight_budget_and_invalid_identity_refuses_without_artifacts_or_reads() {
    let parent = parent();
    let expected = identity(b"a");
    for field in 0..6 {
        let mut cap = limits(1);
        match field {
            0 => cap.max_source_bytes = 1,
            1 => cap.max_disk_bytes = 0,
            2 => cap.max_mapped_bytes = 0,
            3 => cap.max_total_bytes -= 1,
            4 => cap.max_read_calls = 1,
            _ => cap.max_write_calls = 0,
        }
        let error = Snapshot::copy_from(
            &mut NeverRead,
            parent.path(),
            expected,
            cap,
            &AtomicBool::new(false),
        )
        .unwrap_err();
        match field {
            4 => assert!(matches!(error.kind(), SnapshotErrorKind::ReadCallBudget)),
            5 => assert!(matches!(error.kind(), SnapshotErrorKind::WriteCallBudget)),
            _ => assert!(matches!(error.kind(), SnapshotErrorKind::Budget)),
        }
        assert!(!error.cleanup_pending());
        empty(parent.path());
    }
    for bad in [
        LeaseIdentity {
            content_sha256: [0; 32],
            byte_len: 1,
        },
        LeaseIdentity {
            content_sha256: [1; 32],
            byte_len: 0,
        },
        LeaseIdentity {
            content_sha256: [1; 32],
            byte_len: u64::MAX,
        },
    ] {
        assert!(Snapshot::copy_from(
            &mut NeverRead,
            parent.path(),
            bad,
            limits(1),
            &AtomicBool::new(false)
        )
        .is_err());
        empty(parent.path());
    }
    let error = Snapshot::copy_from(
        &mut NeverRead,
        parent.path(),
        expected,
        limits(1),
        &AtomicBool::new(true),
    )
    .unwrap_err();
    assert!(matches!(error.kind(), SnapshotErrorKind::Cancelled));
    let missing = parent.path().join("absent-parent");
    assert!(Snapshot::copy_from(
        &mut NeverRead,
        &missing,
        expected,
        limits(1),
        &AtomicBool::new(false)
    )
    .is_err());
    assert!(!missing.exists());
    empty(parent.path());
}

#[test]
fn multichunk_full_digest_and_arc_lifetime_exclude_writers_until_final_drop() {
    let parent = parent();
    let bytes: Vec<_> = (0..COPY_BUFFER_BYTES * 2 + 17)
        .map(|n| ((n * 73) % 251) as u8)
        .collect();
    let mut cap = limits(bytes.len() as u64);
    cap.max_read_calls = 4;
    cap.max_write_calls = 3;
    let snapshot = Snapshot::copy_from(
        &mut &bytes[..],
        parent.path(),
        identity(&bytes),
        cap,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(snapshot.bytes(), bytes);
    assert_eq!(snapshot.identity(), identity(&bytes));
    let directory = child(parent.path());
    assert_eq!(fs::read(directory.join("model.bin")).unwrap(), bytes);
    assert!(OpenOptions::new()
        .write(true)
        .open(directory.join("model.bin"))
        .is_err());
    let snapshot = Arc::new(snapshot);
    let second = snapshot.clone();
    drop(snapshot);
    assert!(directory.exists());
    assert_eq!(second.bytes(), bytes);
    drop(second);
    assert!(!directory.exists());
    empty(parent.path());
}

struct OneByte<'a>(&'a [u8]);
impl Read for OneByte<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if self.0.is_empty() {
            return Ok(0);
        }
        output[0] = self.0[0];
        self.0 = &self.0[1..];
        Ok(1)
    }
}
struct Interrupted {
    calls: u64,
}
impl Read for Interrupted {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        self.calls += 1;
        Err(io::ErrorKind::Interrupted.into())
    }
}
struct BadCount;
impl Read for BadCount {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        Ok(output.len() + 1)
    }
}

#[test]
fn actual_partial_reads_and_interrupted_calls_obey_both_runtime_call_limits() {
    let parent = parent();
    let expected = identity(b"abc");
    let mut cap = limits(3);
    cap.max_write_calls = 1;
    let error = Snapshot::copy_from(
        &mut OneByte(b"abc"),
        parent.path(),
        expected,
        cap,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(matches!(error.kind(), SnapshotErrorKind::WriteCallBudget));
    empty(parent.path());
    cap = limits(3);
    cap.max_read_calls = 3;
    let error = Snapshot::copy_from(
        &mut OneByte(b"abc"),
        parent.path(),
        expected,
        cap,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(matches!(error.kind(), SnapshotErrorKind::ReadCallBudget));
    empty(parent.path());
    let mut source = Interrupted { calls: 0 };
    let error = Snapshot::copy_from(
        &mut source,
        parent.path(),
        expected,
        cap,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(matches!(error.kind(), SnapshotErrorKind::ReadCallBudget));
    assert_eq!(source.calls, 3);
    empty(parent.path());
    let error = Snapshot::copy_from(
        &mut BadCount,
        parent.path(),
        expected,
        limits(3),
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(matches!(error.kind(), SnapshotErrorKind::InvalidReadCount));
    empty(parent.path());
    cap.max_read_calls = 4;
    cap.max_write_calls = 3;
    Snapshot::copy_from(
        &mut OneByte(b"abc"),
        parent.path(),
        expected,
        cap,
        &AtomicBool::new(false),
    )
    .unwrap()
    .close()
    .unwrap();
    empty(parent.path());
}

#[test]
fn short_long_corrupt_and_midread_cancellation_never_return_a_snapshot() {
    let parent = parent();
    for (index, source) in [b"ab".as_slice(), b"abcd".as_slice(), b"abd".as_slice()]
        .into_iter()
        .enumerate()
    {
        let error = Snapshot::copy_from(
            &mut &source[..],
            parent.path(),
            identity(b"abc"),
            limits(3),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        match index {
            0 => assert!(matches!(error.kind(), SnapshotErrorKind::ShortSource)),
            1 => assert!(matches!(error.kind(), SnapshotErrorKind::LongSource)),
            _ => assert!(matches!(error.kind(), SnapshotErrorKind::ContentMismatch)),
        }
        assert!(!error.cleanup_pending());
        empty(parent.path());
    }
    struct Cancel<'a>(&'a AtomicBool);
    impl Read for Cancel<'_> {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            out[0] = b'a';
            self.0.store(true, Ordering::Release);
            Ok(1)
        }
    }
    let cancelled = AtomicBool::new(false);
    let error = Snapshot::copy_from(
        &mut Cancel(&cancelled),
        parent.path(),
        identity(b"a"),
        limits(1),
        &cancelled,
    )
    .unwrap_err();
    assert!(matches!(error.kind(), SnapshotErrorKind::Cancelled));
    empty(parent.path());
}

#[test]
fn real_foreign_read_handle_keeps_close_and_retry_cleanup_obligation_alive() {
    let parent = parent();
    let snapshot = Snapshot::copy_from(
        &mut &b"abc"[..],
        parent.path(),
        identity(b"abc"),
        limits(3),
        &AtomicBool::new(false),
    )
    .unwrap();
    let directory = child(parent.path());
    let held = OpenOptions::new()
        .read(true)
        .share_mode(1)
        .open(directory.join("model.bin"))
        .unwrap();
    let os_refusal = fs::remove_file(directory.join("model.bin")).unwrap_err();
    let pending = snapshot.close().unwrap_err();
    assert!(directory.exists());
    assert_eq!(pending.last_error(), os_refusal.kind());
    let pending = pending.retry().unwrap_err();
    assert!(directory.exists());
    drop(held);
    pending.retry().unwrap();
    assert!(!directory.exists());
    empty(parent.path());
}

#[test]
fn failed_constructor_preserves_real_os_blocked_cleanup_and_notfound_retry() {
    struct HoldDirectory<'a> {
        parent: &'a Path,
        held: Option<File>,
    }
    impl Read for HoldDirectory<'_> {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            let path = child(self.parent);
            self.held = Some(
                OpenOptions::new()
                    .read(true)
                    .share_mode(3)
                    .custom_flags(0x02000000)
                    .open(path)
                    .unwrap(),
            );
            Err(io::ErrorKind::Other.into())
        }
    }
    let parent = parent();
    let mut source = HoldDirectory {
        parent: parent.path(),
        held: None,
    };
    let mut error = Snapshot::copy_from(
        &mut source,
        parent.path(),
        identity(b"a"),
        limits(1),
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(matches!(error.kind(), SnapshotErrorKind::Io(_)));
    assert!(error.cleanup_pending());
    let pending = error.take_cleanup().unwrap();
    assert!(!error.cleanup_pending());
    assert!(error.take_cleanup().is_none());
    drop(error);
    let directory = child(parent.path());
    let pending = pending.retry().unwrap_err();
    drop(source.held.take());
    // Only the known directory created inside this test-owned parent is removed.
    fs::remove_dir_all(&directory).unwrap();
    pending.retry().unwrap();
    empty(parent.path());
    fs::create_dir(&directory).unwrap();
    fs::write(directory.join("replacement"), b"separate later object").unwrap();
    assert!(directory.join("replacement").exists());
    fs::remove_dir_all(directory).unwrap();
    empty(parent.path());
}

#[test]
fn reader_unwind_and_completed_owner_unwind_clean_owned_artifacts() {
    struct PanicReader;
    impl Read for PanicReader {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            panic!("controlled reader unwind")
        }
    }
    let parent = parent();
    let result = std::panic::catch_unwind(|| {
        let _ = Snapshot::copy_from(
            &mut PanicReader,
            parent.path(),
            identity(b"a"),
            limits(1),
            &AtomicBool::new(false),
        );
    });
    assert!(result.is_err());
    empty(parent.path());
    let result = std::panic::catch_unwind(|| {
        let snapshot = Snapshot::copy_from(
            &mut &b"abc"[..],
            parent.path(),
            identity(b"abc"),
            limits(3),
            &AtomicBool::new(false),
        )
        .unwrap();
        std::hint::black_box(snapshot.bytes());
        panic!("controlled owner unwind");
    });
    assert!(result.is_err());
    empty(parent.path());
}

#[test]
fn borrowed_snapshot_access_and_arc_cloning_are_zero_heap_after_cold_copy() {
    let positive = allocation::AllocGuard::begin("positive", true);
    let bytes = vec![3u8; 99];
    std::hint::black_box(&bytes);
    drop(bytes);
    assert!(positive.check().is_err());
    let parent = parent();
    let snapshot = Arc::new(
        Snapshot::copy_from(
            &mut &b"abc"[..],
            parent.path(),
            identity(b"abc"),
            limits(3),
            &AtomicBool::new(false),
        )
        .unwrap(),
    );
    let expected = identity(b"abc");
    let measured = allocation::AllocGuard::begin("hot borrow", true);
    for _ in 0..128 {
        let borrowed = snapshot.clone();
        assert_eq!(borrowed.bytes(), b"abc");
        assert_eq!(borrowed.identity(), expected);
    }
    assert!(measured.check().is_ok());
    Arc::try_unwrap(snapshot).unwrap().close().unwrap();
    empty(parent.path());
}
