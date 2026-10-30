//! External memory and timeline semaphore synchronization bridge.

use super::lease::AllocationId;

/// External memory handle imported from an OS mapping or foreign driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalMemoryHandle {
    OpaqueFd(i32),
    OpaqueWin32(*mut core::ffi::c_void),
    HostPointer { ptr: *mut u8, size_bytes: usize },
}

unsafe impl Send for ExternalMemoryHandle {}
unsafe impl Sync for ExternalMemoryHandle {}

/// Descriptor for importing external memory into Etalon.
#[derive(Clone, Copy, Debug)]
pub struct ExternalMemoryImportDesc {
    pub allocation: AllocationId,
    pub handle: ExternalMemoryHandle,
    pub size_bytes: u64,
    pub dedicated: bool,
}
