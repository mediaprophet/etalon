use super::*;
// ─── FFI Firewall: Type-Erased Interface ─────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmlStatus {
    Success = 0,
    InvalidPointer = 1,
    DeviceCreationFailed = 2,
    DirectStorageFailed = 3,
    IocpStubbed = 4,
}

impl DmlStatus {
    /// Stable human-readable status string.
    pub fn message(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::InvalidPointer => "invalid pointer passed to DirectML FFI",
            Self::DeviceCreationFailed => "DirectML device creation failed",
            Self::DirectStorageFailed => "DirectStorage operation failed",
            Self::IocpStubbed => {
                "IOCP implementation stubbed due to DirectX API version conflicts. Use MmapGridManager instead."
            }
        }
    }
}

/// FFI-safe wrapper for DirectStorage read operation.
///
/// This function accepts an opaque `DmlDevice` owner pointer plus type-erased
/// Windows handles/resources for the file and GPU buffer.
///
/// # Safety
/// - `device_ptr` must be a valid pointer previously returned by
///   `create_d3d12_device_ffi`
/// - `file_handle` must be a valid Windows HANDLE
/// - `gpu_buffer_ptr` must be a valid pointer to ID3D12Resource
#[no_mangle]
pub unsafe extern "C" fn directstorage_read_ffi(
    device_ptr: *mut DmlDevice,
    file_handle: *mut core::ffi::c_void,
    gpu_buffer_ptr: *mut core::ffi::c_void,
    offset: u64,
    size: u64,
) -> DmlStatus {
    match directstorage_read_impl(device_ptr, file_handle, gpu_buffer_ptr, offset, size) {
        Ok(()) => DmlStatus::Success,
        Err(err) => err.into(),
    }
}

unsafe fn directstorage_read_impl(
    device_ptr: *mut DmlDevice,
    file_handle: *mut core::ffi::c_void,
    gpu_buffer_ptr: *mut core::ffi::c_void,
    offset: u64,
    size: u64,
) -> Result<(), DmlError> {
    if device_ptr.is_null() || file_handle.is_null() || gpu_buffer_ptr.is_null() {
        return Err(DmlError::InvalidPointer);
    }

    // Cast type-erased pointers back to specific Windows types
    let _device = &(*device_ptr).d3d12;
    let handle = HANDLE(file_handle);
    let _gpu_buffer = &*(gpu_buffer_ptr as *const ID3D12Resource);

    // DirectStorage requires the Microsoft.DirectStorage SDK (dstorage.dll) runtime.
    // When dstorage.dll is not linked/loaded, fail closed honestly with DmlStatus::DirectStorageFailed
    // so callers immediately route through MmapGridManager or IOCP file streaming rather than
    // silently reading uninitialized GPU memory.
    log::warn!(
        "DirectStorage DMA bypass requested (handle={:?}, offset={}, size={}), but Microsoft.DirectStorage SDK runtime (dstorage.dll) is not loaded. Failing closed to invoke fallback.",
        handle,
        offset,
        size
    );

    Err(DmlError::DirectStorageFailed(
        "DirectStorage hardware DMA bypass requires Microsoft.DirectStorage SDK (dstorage.dll). \
         Runtime not loaded; route through MmapGridManager or IOCP streaming path."
            .to_string(),
    ))
}

/// FFI-safe wrapper for D3D12 device creation.
///
/// Returns an owning opaque pointer to the heap-allocated `DmlDevice`.
#[no_mangle]
pub unsafe extern "C" fn create_d3d12_device_ffi(out_device: *mut *mut DmlDevice) -> DmlStatus {
    if out_device.is_null() {
        return DmlStatus::InvalidPointer;
    }

    *out_device = std::ptr::null_mut();
    match DmlDevice::new() {
        Ok(device) => {
            *out_device = Box::into_raw(Box::new(device));
            DmlStatus::Success
        }
        Err(err) => DmlError::from(err).into(),
    }
}

/// FFI-safe wrapper for D3D12 device destruction.
#[no_mangle]
pub unsafe extern "C" fn destroy_d3d12_device_ffi(device_ptr: *mut DmlDevice) {
    if device_ptr.is_null() {
        return;
    }
    let _ = Box::from_raw(device_ptr);
}

// ─── IOCP FFI Firewall for Host Module ─────────────────────────────────────────────

/// Opaque handle for IOCP grid manager (type-erased to avoid Windows type imports in host.rs)
pub struct IocpHandle {
    file: std::fs::File,
    chunk_buffer: Box<[u8; 65536]>,
    bytes_read: usize,
    pending: bool,
}

/// FFI-safe wrapper for creating an IOCP grid manager.
#[no_mangle]
pub unsafe extern "C" fn iocp_create_ffi(
    file_path: *const u8,
    file_path_len: usize,
    out_handle: *mut *mut IocpHandle,
) -> DmlStatus {
    if out_handle.is_null() || file_path.is_null() {
        return DmlStatus::InvalidPointer;
    }

    *out_handle = std::ptr::null_mut();
    let slice = std::slice::from_raw_parts(file_path, file_path_len);
    let path_str = match std::str::from_utf8(slice) {
        Ok(s) => s,
        Err(_) => return DmlStatus::InvalidPointer,
    };
    match iocp_create_impl(path_str) {
        Ok(handle) => {
            *out_handle = handle;
            DmlStatus::Success
        }
        Err(err) => err.into(),
    }
}

fn iocp_create_impl(path: &str) -> Result<*mut IocpHandle, DmlError> {
    let file = std::fs::File::open(path).map_err(|e| {
        DmlError::DirectStorageFailed(format!("Failed to open file for IOCP: {}", e))
    })?;
    let handle = Box::new(IocpHandle {
        file,
        chunk_buffer: Box::new([0u8; 65536]),
        bytes_read: 0,
        pending: false,
    });
    Ok(Box::into_raw(handle))
}

/// FFI-safe wrapper for destroying an IOCP grid manager.
#[no_mangle]
pub unsafe extern "C" fn iocp_destroy_ffi(handle: *mut IocpHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

/// FFI-safe wrapper for async read chunk.
#[no_mangle]
pub unsafe extern "C" fn iocp_async_read_ffi(handle: *mut IocpHandle, offset: u64) -> DmlStatus {
    if handle.is_null() {
        return DmlStatus::InvalidPointer;
    }
    let h = &mut *handle;
    match iocp_async_read_impl(h, offset) {
        Ok(()) => DmlStatus::Success,
        Err(err) => err.into(),
    }
}

fn iocp_async_read_impl(handle: &mut IocpHandle, offset: u64) -> Result<(), DmlError> {
    use std::os::windows::fs::FileExt;
    let bytes_read = handle
        .file
        .seek_read(&mut handle.chunk_buffer[..], offset)
        .map_err(|e| DmlError::DirectStorageFailed(format!("IOCP read error: {}", e)))?;
    handle.bytes_read = bytes_read;
    handle.pending = true;
    Ok(())
}

/// FFI-safe wrapper for polling completion.
#[no_mangle]
pub unsafe extern "C" fn iocp_poll_completion_ffi(
    handle: *mut IocpHandle,
    out_buffer: *mut *const u8,
    out_size: *mut usize,
) -> bool {
    if handle.is_null() {
        return false;
    }
    let h = &mut *handle;
    if !h.pending {
        return false;
    }
    if !out_buffer.is_null() {
        *out_buffer = h.chunk_buffer.as_ptr();
    }
    if !out_size.is_null() {
        *out_size = h.bytes_read;
    }
    h.pending = false;
    true
}

#[derive(Debug)]
pub enum DmlError {
    DeviceCreationFailed(String),
    DirectStorageFailed(String),
    IocpStubbed(String),
    InvalidPointer,
}

impl From<windows::core::Error> for DmlError {
    fn from(e: windows::core::Error) -> Self {
        DmlError::DeviceCreationFailed(e.to_string())
    }
}

impl From<DmlError> for DmlStatus {
    fn from(value: DmlError) -> Self {
        match value {
            DmlError::InvalidPointer => DmlStatus::InvalidPointer,
            DmlError::DeviceCreationFailed(_) => DmlStatus::DeviceCreationFailed,
            DmlError::DirectStorageFailed(_) => DmlStatus::DirectStorageFailed,
            DmlError::IocpStubbed(_) => DmlStatus::IocpStubbed,
        }
    }
}
