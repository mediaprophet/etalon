//! C ABI re-expression of pinned Qualcomm `remote.h`, not the kernel ioctl ABI.
//!
//! Reference: ../sdk-reference/qualcomm-fastrpc/remote.h at
//! a56e9d4de3614a8d8f4b0e15ff561350c4b686e3 (BSD-3-Clause).

use core::ffi::{c_char, c_int, c_void};

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RemoteBuf {
    pub pv: *mut c_void,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RemoteDmaHandle {
    pub fd: i32,
    pub offset: u32,
}

// remote_handle64_invoke takes remote_arg, NOT remote_arg64.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) union RemoteArg {
    pub buf: RemoteBuf,
    pub h: u32,
    pub h64: u64,
    pub dma: RemoteDmaHandle,
}

pub(crate) type Open = unsafe extern "C" fn(*const c_char, *mut u64) -> c_int;
pub(crate) type Invoke = unsafe extern "C" fn(u64, u32, *mut RemoteArg) -> c_int;
pub(crate) type Close = unsafe extern "C" fn(u64) -> c_int;

pub(crate) const fn buffer_scalars(method: u8, attributes: u8) -> Option<u32> {
    if method > 31 || attributes > 7 { return None; }
    Some(((attributes as u32) << 29) | ((method as u32) << 24) | (1 << 16) | (1 << 8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_scalars_do_not_truncate_method_or_attributes() {
        assert_eq!(buffer_scalars(0, 0), Some(0x0001_0100));
        assert_eq!(buffer_scalars(31, 7), Some(0xff01_0100));
        assert_eq!(buffer_scalars(32, 0), None);
        assert_eq!(buffer_scalars(0, 8), None);
        assert_eq!(buffer_scalars(etalon_sense_dsp_wire::METHOD_INDEX, 0),
            Some(etalon_sense_dsp_wire::METHOD_SCALARS));
    }

    #[test]
    fn native_argument_layout_contains_pointer_size_and_handle() {
        assert_eq!(core::mem::size_of::<RemoteBuf>(), 2 * core::mem::size_of::<usize>());
        assert_eq!(core::mem::size_of::<RemoteArg>(),
            core::mem::size_of::<RemoteBuf>().max(core::mem::size_of::<u64>()));
        assert!(core::mem::align_of::<RemoteArg>() >= core::mem::align_of::<u64>());
    }
}
