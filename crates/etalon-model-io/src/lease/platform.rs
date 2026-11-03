//! Conservative opened-handle support gate: local disk NTFS only. No input-path
//! spelling is evidence of locality. Raw-volume/kernel-driver trust remains an
//! OS boundary, not a claim that remote sharing is exploitable or qualified.
use super::LeaseError;
use std::{
    ffi::c_void,
    fs::{File, OpenOptions},
    os::windows::{fs::OpenOptionsExt, io::AsRawHandle},
    path::Path,
};
#[repr(C)]
struct IoStatus {
    status_or_pointer: StatusOrPointer,
    information: usize,
}
#[repr(C)]
union StatusOrPointer {
    status: i32,
    pointer: *mut c_void,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DeviceInfo {
    device_type: u32,
    characteristics: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct AttributeInfo {
    attributes: u32,
    max_component: i32,
    name_bytes: u32,
    name: [u16; 32],
}
#[link(name = "ntdll")]
unsafe extern "system" {
    fn NtQueryVolumeInformationFile(
        handle: *mut c_void,
        status: *mut IoStatus,
        buffer: *mut c_void,
        length: u32,
        class: i32,
    ) -> i32;
}
const REMOTE_MASK: u32 = 0x10 | 0x1000 | 0x2000 | 0x80000; // remote, TS, WebDAV, VSMB
                                                           // Known local flags from Windows FILE_FS_DEVICE_INFORMATION. Unknown future
                                                           // characteristics require qualification instead of being silently admitted.
const LOCAL_MASK: u32 = 0x1 | 0x2 | 0x4 | 0x8 | 0x20 | 0x40 | 0x100 | 0x4000 | 0x20000;
fn completed(status: i32, io: &IoStatus, maximum: usize) -> Result<usize, LeaseError> {
    // SAFETY: initialized as Status before the call, and this query writes the
    // documented NTSTATUS member (never interpret it as a pointer).
    if status != 0 || unsafe { io.status_or_pointer.status } != 0 || io.information > maximum {
        Err(LeaseError::UnsupportedBacking)
    } else {
        Ok(io.information)
    }
}
fn classify(
    device: DeviceInfo,
    device_len: usize,
    fs: &AttributeInfo,
    fs_len: usize,
) -> Result<(), LeaseError> {
    if device_len != 8
        || device.device_type != 7
        || device.characteristics & REMOTE_MASK != 0
        || device.characteristics & !LOCAL_MASK != 0
        || fs.name_bytes != 8
        || fs_len < 20
        || fs_len > std::mem::size_of::<AttributeInfo>()
        || fs.name[..4] != [b'N' as u16, b'T' as u16, b'F' as u16, b'S' as u16]
    {
        return Err(LeaseError::UnsupportedBacking);
    }
    Ok(())
}
pub(super) fn open_supported(path: &Path) -> Result<File, LeaseError> {
    // Default std File uses synchronous I/O: no FILE_FLAG_OVERLAPPED is set.
    // We own creation; no arbitrary external/asynchronous File is accepted.
    let file = OpenOptions::new()
        .read(true)
        .share_mode(1)
        .open(path)
        .map_err(LeaseError::Io)?;
    let mut device = DeviceInfo::default();
    let mut fs = AttributeInfo::default();
    let mut device_status = IoStatus {
        status_or_pointer: StatusOrPointer { status: -1 },
        information: 0,
    };
    let mut fs_status = IoStatus {
        status_or_pointer: StatusOrPointer { status: -1 },
        information: 0,
    };
    // SAFETY: both fixed repr(C) buffers have Windows ABI sizes/alignment; this
    // live synchronous handle remains owned through both completed queries.
    // Each output pointer is valid for the supplied exact size. No request on
    // this synchronous handle can retain these stack buffers after returning.
    let status = unsafe {
        NtQueryVolumeInformationFile(
            file.as_raw_handle(),
            &mut device_status,
            std::ptr::from_mut(&mut device).cast(),
            std::mem::size_of::<DeviceInfo>() as u32,
            4,
        )
    };
    let device_len = completed(status, &device_status, std::mem::size_of::<DeviceInfo>())?;
    let status = unsafe {
        NtQueryVolumeInformationFile(
            file.as_raw_handle(),
            &mut fs_status,
            std::ptr::from_mut(&mut fs).cast(),
            std::mem::size_of::<AttributeInfo>() as u32,
            5,
        )
    };
    let fs_len = completed(status, &fs_status, std::mem::size_of::<AttributeInfo>())?;
    classify(device, device_len, &fs, fs_len)?;
    Ok(file)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn queried_abi_and_classifier_refuse_unqualified_backing() {
        assert_eq!(std::mem::size_of::<DeviceInfo>(), 8);
        assert_eq!(std::mem::offset_of!(AttributeInfo, name), 12);
        assert_eq!(
            std::mem::size_of::<IoStatus>(),
            2 * std::mem::size_of::<usize>()
        );
        let local = DeviceInfo {
            device_type: 7,
            characteristics: 0x20,
        };
        let mut fs = AttributeInfo {
            name_bytes: 8,
            ..Default::default()
        };
        fs.name[..4].copy_from_slice(&[78, 84, 70, 83]);
        assert!(classify(local, 8, &fs, 20).is_ok());
        for flags in [0x10, 0x1000, 0x2000, 0x80000, 0x8000_0000] {
            assert!(classify(
                DeviceInfo {
                    characteristics: flags,
                    ..local
                },
                8,
                &fs,
                20
            )
            .is_err());
        }
        for kind in [0, 2, 8, 20, u32::MAX] {
            assert!(classify(
                DeviceInfo {
                    device_type: kind,
                    ..local
                },
                8,
                &fs,
                20
            )
            .is_err());
        }
        for n in [0, 7, 9, usize::MAX] {
            assert!(classify(local, n, &fs, 20).is_err());
        }
        for n in [0, 19, 77, usize::MAX] {
            assert!(classify(local, 8, &fs, n).is_err());
        }
        for n in [0, 7, 9, 64, u32::MAX] {
            let mut bad = fs;
            bad.name_bytes = n;
            assert!(classify(local, 8, &bad, 20).is_err());
        }
        for name in [
            [82, 101, 70, 83],
            [70, 65, 84, 0],
            [110, 116, 102, 115],
            [0; 4],
        ] {
            let mut bad = fs;
            bad.name[..4].copy_from_slice(&name);
            assert!(classify(local, 8, &bad, 20).is_err());
        }
        for status in [1, 0x103, -1, i32::MIN] {
            assert!(completed(
                status,
                &IoStatus {
                    status_or_pointer: StatusOrPointer { status: 0 },
                    information: 8
                },
                8
            )
            .is_err());
        }
        assert!(completed(
            0,
            &IoStatus {
                status_or_pointer: StatusOrPointer { status: 1 },
                information: 8
            },
            8
        )
        .is_err());
        assert!(completed(
            0,
            &IoStatus {
                status_or_pointer: StatusOrPointer { status: 0 },
                information: 9
            },
            8
        )
        .is_err());
    }
}
