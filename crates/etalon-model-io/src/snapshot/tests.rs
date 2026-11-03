//! Cohesive lifecycle fixtures exceed 500 lines to keep owned-directory setup and cleanup challenges together.
use super::*;
use crate::lease::LeaseIdentity;
use std::{io::Read, sync::atomic::AtomicBool};

#[cfg(windows)]
use sha2::{Digest, Sha256};
#[cfg(windows)]
fn expected(bytes: &[u8]) -> LeaseIdentity {
    LeaseIdentity {
        content_sha256: Sha256::digest(bytes).into(),
        byte_len: bytes.len() as u64,
    }
}
#[cfg(windows)]
fn limits(len: u64) -> SnapshotLimits {
    SnapshotLimits {
        max_source_bytes: len + 1,
        max_disk_bytes: len,
        max_mapped_bytes: len,
        max_total_bytes: len * 3 + 1 + COPY_BUFFER_BYTES as u64,
        max_read_calls: 100,
        max_write_calls: 100,
    }
}
#[cfg(windows)]
fn parent() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}
#[cfg(windows)]
fn empty(parent: &tempfile::TempDir) {
    assert_eq!(std::fs::read_dir(parent.path()).unwrap().count(), 0);
}

struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
        panic!("source must not be called")
    }
}

#[cfg(windows)]
#[test]
fn exact_multichunk_snapshot_lives_until_last_arc_then_cleans() {
    let parent = parent();
    let bytes: Vec<_> = (0..COPY_BUFFER_BYTES * 2 + 17).map(|n| n as u8).collect();
    let owner = Snapshot::copy_from(
        &mut &bytes[..],
        parent.path(),
        expected(&bytes),
        limits(bytes.len() as u64),
        &AtomicBool::new(false),
    )
    .unwrap();
    let path = owner.directory.path().to_owned();
    assert_eq!(owner.bytes(), bytes);
    let a = std::sync::Arc::new(owner);
    let b = a.clone();
    drop(a);
    assert!(path.exists());
    assert_eq!(b.identity(), expected(&bytes));
    assert!(std::fs::OpenOptions::new()
        .write(true)
        .open(path.join("model.bin"))
        .is_err());
    drop(b);
    assert!(!path.exists());
    empty(&parent);
}

#[cfg(windows)]
#[test]
fn identity_all_budgets_and_cancellation_refuse_before_any_read_or_directory() {
    let parent = parent();
    let e = expected(b"a");
    for field in 0..6 {
        let mut l = limits(1);
        match field {
            0 => l.max_source_bytes -= 1,
            1 => l.max_disk_bytes = 0,
            2 => l.max_mapped_bytes = 0,
            3 => l.max_total_bytes -= 1,
            4 => l.max_read_calls = 1,
            _ => l.max_write_calls = 0,
        }
        assert!(
            Snapshot::copy_from(&mut NoRead, parent.path(), e, l, &AtomicBool::new(false)).is_err()
        );
        empty(&parent);
    }
    for e in [
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
        LeaseIdentity {
            content_sha256: [1; 32],
            byte_len: isize::MAX as u64,
        },
    ] {
        assert!(Snapshot::copy_from(
            &mut NoRead,
            parent.path(),
            e,
            limits(1),
            &AtomicBool::new(false)
        )
        .is_err());
        empty(&parent);
    }
    assert!(matches!(
        Snapshot::copy_from(
            &mut NoRead,
            parent.path(),
            e,
            limits(1),
            &AtomicBool::new(true)
        )
        .unwrap_err()
        .kind(),
        SnapshotErrorKind::Cancelled
    ));
    empty(&parent);
    let long = std::path::PathBuf::from("a".repeat(32_001));
    assert!(matches!(
        Snapshot::copy_from(&mut NoRead, &long, e, limits(1), &AtomicBool::new(false))
            .unwrap_err()
            .kind(),
        SnapshotErrorKind::PathBudget
    ));
    empty(&parent);
}

#[cfg(windows)]
#[test]
fn short_long_wrong_digest_and_io_failures_clean() {
    let parent = parent();
    for (bytes, case) in [(&b"ab"[..], 0), (&b"abcd"[..], 1), (&b"abd"[..], 2)] {
        let error = Snapshot::copy_from(
            &mut &bytes[..],
            parent.path(),
            expected(b"abc"),
            limits(3),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(matches!(
            (case, error.kind()),
            (0, SnapshotErrorKind::ShortSource)
                | (1, SnapshotErrorKind::LongSource)
                | (2, SnapshotErrorKind::ContentMismatch)
        ));
        assert!(!error.cleanup_pending());
        empty(&parent);
    }
    struct Fail;
    impl Read for Fail {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }
    }
    assert!(matches!(
        Snapshot::copy_from(
            &mut Fail,
            parent.path(),
            expected(b"a"),
            limits(1),
            &AtomicBool::new(false)
        )
        .unwrap_err()
        .kind(),
        SnapshotErrorKind::Io(_)
    ));
    empty(&parent);
}

#[cfg(windows)]
#[test]
fn interrupted_calls_short_reads_and_extra_byte_probe_are_counted() {
    struct Pieces {
        interrupted: bool,
        bytes: &'static [u8],
    }
    impl Read for Pieces {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            if !self.interrupted {
                self.interrupted = true;
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            if self.bytes.is_empty() {
                return Ok(0);
            }
            out[0] = self.bytes[0];
            self.bytes = &self.bytes[1..];
            Ok(1)
        }
    }
    let parent = parent();
    for calls in [3, 4, 5] {
        let mut l = limits(3);
        l.max_read_calls = calls;
        let result = Snapshot::copy_from(
            &mut Pieces {
                interrupted: false,
                bytes: b"abc",
            },
            parent.path(),
            expected(b"abc"),
            l,
            &AtomicBool::new(false),
        );
        if calls == 5 {
            result.unwrap().close().unwrap();
        } else {
            assert!(matches!(
                result.unwrap_err().kind(),
                SnapshotErrorKind::ReadCallBudget
            ));
        }
        empty(&parent);
    }
}

#[cfg(windows)]
#[test]
fn dishonest_reader_and_panic_are_bounded_and_clean() {
    struct Excess;
    impl Read for Excess {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            Ok(b.len() + 1)
        }
    }
    let parent = parent();
    assert!(matches!(
        Snapshot::copy_from(
            &mut Excess,
            parent.path(),
            expected(b"a"),
            limits(1),
            &AtomicBool::new(false)
        )
        .unwrap_err()
        .kind(),
        SnapshotErrorKind::InvalidReadCount
    ));
    empty(&parent);
    assert!(std::panic::catch_unwind(|| Snapshot::copy_from(
        &mut NoRead,
        parent.path(),
        expected(b"a"),
        limits(1),
        &AtomicBool::new(false)
    ))
    .is_err());
    empty(&parent);
}

#[cfg(windows)]
#[test]
fn cancellation_observed_after_reader_returns_prevents_copy_success() {
    struct Cancel<'a>(&'a AtomicBool);
    impl Read for Cancel<'_> {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            b[0] = b'a';
            self.0.store(true, std::sync::atomic::Ordering::Release);
            Ok(1)
        }
    }
    let parent = parent();
    let cancelled = AtomicBool::new(false);
    assert!(matches!(
        Snapshot::copy_from(
            &mut Cancel(&cancelled),
            parent.path(),
            expected(b"a"),
            limits(1),
            &cancelled
        )
        .unwrap_err()
        .kind(),
        SnapshotErrorKind::Cancelled
    ));
    empty(&parent);
}

#[cfg(windows)]
#[test]
fn explicit_close_and_constructor_errors_retain_failed_cleanup_for_retry() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            super::owner::FAIL_REMOVE.set(false);
        }
    }
    let parent = parent();
    let owner = Snapshot::copy_from(
        &mut &b"abc"[..],
        parent.path(),
        expected(b"abc"),
        limits(3),
        &AtomicBool::new(false),
    )
    .unwrap();
    let _reset = Reset;
    super::owner::FAIL_REMOVE.set(true);
    let pending = owner.close().unwrap_err();
    assert_eq!(pending.last_error(), std::io::ErrorKind::PermissionDenied);
    assert_eq!(std::fs::read_dir(parent.path()).unwrap().count(), 1);
    let pending = pending.retry().unwrap_err();
    super::owner::FAIL_REMOVE.set(false);
    pending.retry().unwrap();
    empty(&parent);
    super::owner::FAIL_REMOVE.set(true);
    let mut error = Snapshot::copy_from(
        &mut &b"x"[..],
        parent.path(),
        expected(b"abc"),
        limits(3),
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(matches!(error.kind(), SnapshotErrorKind::ShortSource));
    assert!(error.cleanup_pending());
    let pending = error.take_cleanup().unwrap();
    drop(error);
    assert_eq!(std::fs::read_dir(parent.path()).unwrap().count(), 1);
    super::owner::FAIL_REMOVE.set(false);
    pending.retry().unwrap();
    empty(&parent);
}

#[cfg(windows)]
#[test]
fn unsupported_parent_never_reads_source_or_creates_output() {
    let parent = parent();
    let file = parent.path().join("ordinary-file");
    std::fs::write(&file, b"original").unwrap();
    assert!(Snapshot::copy_from(
        &mut NoRead,
        &file,
        expected(b"a"),
        limits(1),
        &AtomicBool::new(false)
    )
    .is_err());
    assert_eq!(std::fs::read(&file).unwrap(), b"original");
    assert_eq!(std::fs::read_dir(parent.path()).unwrap().count(), 1);
}

#[cfg(not(windows))]
#[test]
fn unsupported_platform_does_not_call_source() {
    let result = Snapshot::copy_from(
        &mut NoRead,
        std::path::Path::new("absent"),
        LeaseIdentity {
            content_sha256: [1; 32],
            byte_len: 1,
        },
        SnapshotLimits {
            max_source_bytes: 2,
            max_disk_bytes: 1,
            max_mapped_bytes: 1,
            max_total_bytes: 65540,
            max_read_calls: 2,
            max_write_calls: 1,
        },
        &AtomicBool::new(false),
    );
    assert!(matches!(
        result.unwrap_err().kind(),
        SnapshotErrorKind::UnsupportedPlatform
    ));
}

#[cfg(windows)]
#[test]
fn final_local_lease_rechecks_actual_bytes_length_cancellation_and_unwind() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            super::copy::FINAL_FAULT.set(0);
        }
    }
    let parent = parent();
    for fault in 1..=4 {
        let _reset = Reset;
        super::copy::FINAL_FAULT.set(fault);
        let result = std::panic::catch_unwind(|| {
            Snapshot::copy_from(
                &mut &b"abc"[..],
                parent.path(),
                expected(b"abc"),
                limits(3),
                &AtomicBool::new(false),
            )
        });
        if fault == 4 {
            assert!(result.is_err());
        } else {
            let error = result.unwrap().unwrap_err();
            if fault == 3 {
                assert!(matches!(error.kind(), SnapshotErrorKind::Cancelled));
            } else if fault == 1 {
                assert!(matches!(
                    error.kind(),
                    SnapshotErrorKind::Lease(crate::lease::LeaseError::ContentMismatch)
                ));
            } else {
                assert!(matches!(
                    error.kind(),
                    SnapshotErrorKind::Lease(crate::lease::LeaseError::Budget)
                ));
            }
        }
        empty(&parent);
    }
}

#[cfg(windows)]
#[test]
fn actual_open_reader_can_delay_cleanup_and_retry_keeps_ownership() {
    use std::os::windows::fs::OpenOptionsExt;
    let parent = parent();
    let owner = Snapshot::copy_from(
        &mut &b"abc"[..],
        parent.path(),
        expected(b"abc"),
        limits(3),
        &AtomicBool::new(false),
    )
    .unwrap();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .share_mode(1)
        .open(owner.directory.path().join("model.bin"))
        .unwrap();
    let pending = owner.close().unwrap_err();
    assert_eq!(std::fs::read_dir(parent.path()).unwrap().count(), 1);
    drop(file);
    pending.retry().unwrap();
    empty(&parent);
}

#[cfg(windows)]
#[test]
fn successful_and_already_removed_cleanup_never_delete_a_later_replacement() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            super::owner::RECREATE.set(false);
        }
    }
    let parent = parent();
    for already_removed in [false, true] {
        let dir = tempfile::tempdir_in(parent.path()).unwrap();
        let path = dir.path().to_owned();
        if already_removed {
            std::fs::remove_dir(&path).unwrap();
        }
        let _reset = Reset;
        super::owner::RECREATE.set(true);
        super::owner::CleanupPending::new(dir).retry().unwrap();
        assert_eq!(
            std::fs::read(path.join("replacement")).unwrap(),
            b"not the snapshot"
        );
        std::fs::remove_file(path.join("replacement")).unwrap();
        std::fs::remove_dir(path).unwrap();
    }
    empty(&parent);
}

#[cfg(windows)]
#[test]
fn actual_write_loop_counts_interrupted_and_partial_attempts() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            super::copy::WRITE_MODE.set(0);
        }
    }
    let parent = parent();
    for (mode, calls, success) in [(1, 2, false), (2, 2, false), (2, 3, true)] {
        let _reset = Reset;
        super::copy::WRITE_MODE.set(mode);
        let mut l = limits(3);
        l.max_write_calls = calls;
        let result = Snapshot::copy_from(
            &mut &b"abc"[..],
            parent.path(),
            expected(b"abc"),
            l,
            &AtomicBool::new(false),
        );
        if success {
            let snapshot = result.unwrap();
            assert_eq!(snapshot.bytes(), b"abc");
            snapshot.close().unwrap();
        } else {
            assert!(matches!(
                result.unwrap_err().kind(),
                SnapshotErrorKind::WriteCallBudget
            ));
        }
        empty(&parent);
    }
}
