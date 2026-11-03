use super::{Snapshot, SnapshotError, SnapshotErrorKind, SnapshotLimits};
use crate::lease::LeaseIdentity;
use std::{io::Read, path::Path, sync::atomic::AtomicBool};

pub(super) fn copy_from<R: Read + ?Sized>(
    source: &mut R,
    parent: &Path,
    expected: LeaseIdentity,
    limits: SnapshotLimits,
    cancelled: &AtomicBool,
) -> Result<Snapshot, SnapshotError> {
    #[cfg(not(windows))]
    {
        let _ = (source, parent, expected, limits, cancelled);
        Err(SnapshotErrorKind::UnsupportedPlatform.into())
    }
    #[cfg(windows)]
    {
        use super::CleanupPending;
        use crate::lease::platform;
        limits.admit(expected)?;
        check_cancel(cancelled)?;
        check_path(parent)?;
        let parent = parent.canonicalize().map_err(SnapshotErrorKind::Io)?;
        check_path(&parent)?;
        let _parent_handle =
            platform::open_snapshot_parent(&parent).map_err(SnapshotErrorKind::Lease)?;
        check_cancel(cancelled)?;
        let directory = tempfile::Builder::new()
            .prefix("etalon-model-snapshot-")
            .tempdir_in(&parent)
            .map_err(SnapshotErrorKind::Io)?;
        let result = build(source, &directory, expected, limits, cancelled);
        match result {
            Ok(lease) => Ok(Snapshot { lease, directory }),
            Err(kind) => Err(SnapshotError {
                kind,
                cleanup: CleanupPending::new(directory).retry().err(),
            }),
        }
    }
}
#[cfg(windows)]
fn check_path(path: &Path) -> Result<(), SnapshotErrorKind> {
    use std::os::windows::ffi::OsStrExt;
    // Reserve space for TempDir's name and the private payload filename.
    if path.as_os_str().encode_wide().take(32_001).count() > 32_000 {
        return Err(SnapshotErrorKind::PathBudget);
    }
    Ok(())
}
#[cfg(windows)]
fn check_cancel(cancelled: &AtomicBool) -> Result<(), SnapshotErrorKind> {
    if cancelled.load(std::sync::atomic::Ordering::Acquire) {
        Err(SnapshotErrorKind::Cancelled)
    } else {
        Ok(())
    }
}
#[cfg(windows)]
fn read_once<R: Read + ?Sized>(
    source: &mut R,
    buffer: &mut [u8],
    calls: &mut u64,
    maximum: u64,
    cancelled: &AtomicBool,
) -> Result<usize, SnapshotErrorKind> {
    loop {
        check_cancel(cancelled)?;
        if *calls >= maximum {
            return Err(SnapshotErrorKind::ReadCallBudget);
        }
        *calls += 1;
        match source.read(buffer) {
            Ok(n) if n > buffer.len() => return Err(SnapshotErrorKind::InvalidReadCount),
            Ok(n) => {
                check_cancel(cancelled)?;
                return Ok(n);
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(SnapshotErrorKind::Io(e)),
        }
    }
}
#[cfg(windows)]
fn build<R: Read + ?Sized>(
    source: &mut R,
    directory: &tempfile::TempDir,
    expected: LeaseIdentity,
    limits: SnapshotLimits,
    cancelled: &AtomicBool,
) -> Result<crate::lease::ModelLease, SnapshotErrorKind> {
    use super::COPY_BUFFER_BYTES;
    use crate::lease::{platform, LeaseLimits, ModelLease};
    use sha2::{Digest, Sha256};
    check_cancel(cancelled)?;
    let path = directory.path().join("model.bin");
    let mut writer = platform::create_snapshot_file(&path).map_err(SnapshotErrorKind::Lease)?;
    let mut buffer = [0u8; COPY_BUFFER_BYTES];
    let mut hash = Sha256::new();
    let mut remaining = expected.byte_len;
    let mut calls = 0;
    let mut write_calls = 0;
    while remaining > 0 {
        let requested = remaining.min(COPY_BUFFER_BYTES as u64) as usize;
        let n = read_once(
            source,
            &mut buffer[..requested],
            &mut calls,
            limits.max_read_calls,
            cancelled,
        )?;
        if n == 0 {
            return Err(SnapshotErrorKind::ShortSource);
        }
        let mut written = 0;
        while written < n {
            check_cancel(cancelled)?;
            if write_calls >= limits.max_write_calls {
                return Err(SnapshotErrorKind::WriteCallBudget);
            }
            write_calls += 1;
            match write_once(&mut writer, &buffer[written..n]) {
                Ok(0) => return Err(SnapshotErrorKind::Io(std::io::ErrorKind::WriteZero.into())),
                Ok(count) => written += count,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(SnapshotErrorKind::Io(e)),
            }
        }
        hash.update(&buffer[..n]);
        remaining -= n as u64;
    }
    if read_once(
        source,
        &mut buffer[..1],
        &mut calls,
        limits.max_read_calls,
        cancelled,
    )? != 0
    {
        return Err(SnapshotErrorKind::LongSource);
    }
    let copied_digest: [u8; 32] = hash.finalize().into();
    if copied_digest != expected.content_sha256 {
        return Err(SnapshotErrorKind::ContentMismatch);
    }
    check_cancel(cancelled)?;
    writer.sync_all().map_err(SnapshotErrorKind::Io)?;
    drop(writer);
    #[cfg(test)]
    final_phase_fault(&path, cancelled);
    check_cancel(cancelled)?;
    let lease = ModelLease::open(
        &path,
        expected.content_sha256,
        LeaseLimits {
            max_source_bytes: expected.byte_len,
            max_mapped_bytes: expected.byte_len,
            max_total_bytes: expected
                .byte_len
                .checked_mul(2)
                .ok_or(SnapshotErrorKind::Budget)?,
        },
        cancelled,
    )
    .map_err(SnapshotErrorKind::Lease)?;
    if lease.identity() != expected {
        return Err(SnapshotErrorKind::ContentMismatch);
    }
    Ok(lease)
}

#[cfg(windows)]
fn write_once(writer: &mut std::fs::File, bytes: &[u8]) -> std::io::Result<usize> {
    use std::io::Write;
    #[cfg(test)]
    match WRITE_MODE.get() {
        1 => return Err(std::io::ErrorKind::Interrupted.into()),
        2 => return writer.write(&bytes[..1]),
        _ => {}
    }
    writer.write(bytes)
}
#[cfg(all(test, windows))]
thread_local! { pub(super) static WRITE_MODE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) }; }

#[cfg(all(test, windows))]
thread_local! { pub(super) static FINAL_FAULT: std::cell::Cell<u8> = const { std::cell::Cell::new(0) }; }
#[cfg(all(test, windows))]
fn final_phase_fault(path: &Path, cancelled: &AtomicBool) {
    match FINAL_FAULT.get() {
        1 => std::fs::write(path, b"abd").unwrap(),
        2 => std::fs::write(path, b"abcd").unwrap(),
        3 => cancelled.store(true, std::sync::atomic::Ordering::Release),
        4 => panic!("snapshot final-phase unwind"),
        _ => {}
    }
}
