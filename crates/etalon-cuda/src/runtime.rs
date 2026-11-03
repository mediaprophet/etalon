//! Owned CUDA contexts, buffers, and explicitly unsafe native kernel execution.

use std::{
    cell::Cell,
    ffi::{c_char, c_void, CStr, CString},
    fmt, ptr,
    rc::Rc,
};

use libloading::Library;

type Handle = *mut c_void;
type DevicePointer = u64;

#[derive(Debug)]
pub enum CudaError {
    Library(String),
    Compilation(String),
    Driver { operation: &'static str, code: i32 },
    InvalidArgument(&'static str),
    ForeignContext,
    Poisoned,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Library(message) => write!(f, "CUDA driver loading failed: {message}"),
            Self::Compilation(message) => write!(f, "CUDA kernel compilation failed: {message}"),
            Self::Driver { operation, code } => {
                write!(f, "{operation} failed with CUDA error {code}")
            }
            Self::InvalidArgument(message) => f.write_str(message),
            Self::ForeignContext => f.write_str("CUDA buffer belongs to another context"),
            Self::Poisoned => f.write_str("CUDA context is poisoned after a driver failure"),
        }
    }
}

impl std::error::Error for CudaError {}

fn check(code: i32, operation: &'static str) -> Result<(), CudaError> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaError::Driver { operation, code })
    }
}

macro_rules! driver_api {
    ($($field:ident: $ty:ty = $symbol:literal),* $(,)?) => {
        struct DriverApi {
            $($field: $ty,)*
            _library: Library,
        }

        impl DriverApi {
            fn load() -> Result<Self, CudaError> {
                let library = load_driver_library()?;
                Ok(Self {
                    $($field: unsafe {
                        *library.get::<$ty>($symbol)
                            .map_err(|error| CudaError::Library(error.to_string()))?
                    },)*
                    _library: library,
                })
            }
        }
    };
}

driver_api! {
    init: unsafe extern "system" fn(u32) -> i32 = b"cuInit\0",
    device_get: unsafe extern "system" fn(*mut i32, i32) -> i32 = b"cuDeviceGet\0",
    device_name: unsafe extern "system" fn(*mut c_char, i32, i32) -> i32 = b"cuDeviceGetName\0",
    device_pci_bus_id: unsafe extern "system" fn(*mut c_char, i32, i32) -> i32 = b"cuDeviceGetPCIBusId\0",
    device_attribute: unsafe extern "system" fn(*mut i32, i32, i32) -> i32 = b"cuDeviceGetAttribute\0",
    context_create: unsafe extern "system" fn(*mut Handle, u32, i32) -> i32 = b"cuCtxCreate_v2\0",
    context_destroy: unsafe extern "system" fn(Handle) -> i32 = b"cuCtxDestroy_v2\0",
    context_push: unsafe extern "system" fn(Handle) -> i32 = b"cuCtxPushCurrent_v2\0",
    context_pop: unsafe extern "system" fn(*mut Handle) -> i32 = b"cuCtxPopCurrent_v2\0",
    context_synchronize: unsafe extern "system" fn() -> i32 = b"cuCtxSynchronize\0",
    stream_create: unsafe extern "system" fn(*mut Handle, u32) -> i32 = b"cuStreamCreate\0",
    stream_destroy: unsafe extern "system" fn(Handle) -> i32 = b"cuStreamDestroy_v2\0",
    stream_synchronize: unsafe extern "system" fn(Handle) -> i32 = b"cuStreamSynchronize\0",
    mem_alloc: unsafe extern "system" fn(*mut DevicePointer, usize) -> i32 = b"cuMemAlloc_v2\0",
    mem_free: unsafe extern "system" fn(DevicePointer) -> i32 = b"cuMemFree_v2\0",
    mem_zero: unsafe extern "system" fn(DevicePointer, u8, usize, Handle) -> i32 = b"cuMemsetD8Async\0",
    copy_htod: unsafe extern "system" fn(DevicePointer, *const c_void, usize) -> i32 = b"cuMemcpyHtoD_v2\0",
    copy_dtoh: unsafe extern "system" fn(*mut c_void, DevicePointer, usize) -> i32 = b"cuMemcpyDtoH_v2\0",
    module_load: unsafe extern "system" fn(*mut Handle, *const c_void) -> i32 = b"cuModuleLoadData\0",
    module_unload: unsafe extern "system" fn(Handle) -> i32 = b"cuModuleUnload\0",
    module_function: unsafe extern "system" fn(*mut Handle, Handle, *const c_char) -> i32 = b"cuModuleGetFunction\0",
    function_attribute: unsafe extern "system" fn(*mut i32, i32, Handle) -> i32 = b"cuFuncGetAttribute\0",
    launch_kernel: unsafe extern "system" fn(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut *mut c_void, *mut *mut c_void) -> i32 = b"cuLaunchKernel\0",
}

fn load_driver_library() -> Result<Library, CudaError> {
    #[cfg(windows)]
    {
        unsafe { libloading::os::windows::Library::load_with_flags("nvcuda.dll", 0x0000_0800) }
            .map(Into::into)
            .map_err(|error| CudaError::Library(error.to_string()))
    }
    #[cfg(target_os = "linux")]
    {
        unsafe { Library::new("libcuda.so.1") }
            .map_err(|error| CudaError::Library(error.to_string()))
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        Err(CudaError::Library("CUDA requires Windows or Linux".into()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaDeviceInfo {
    pub name: String,
    pub ordinal: i32,
    pub compute_capability: (u32, u32),
    pub pci_bus_id: String,
}

#[derive(Clone, Copy, Debug)]
struct LaunchLimits {
    block: [u32; 3],
    grid: [u32; 3],
    threads_per_block: u32,
    shared_memory: u32,
}

fn device_attribute(api: &DriverApi, device: i32, attribute: i32) -> Result<u32, CudaError> {
    let mut value = 0;
    check(
        unsafe { (api.device_attribute)(&mut value, attribute, device) },
        "cuDeviceGetAttribute",
    )?;
    u32::try_from(value).map_err(|_| CudaError::InvalidArgument("negative CUDA device attribute"))
}

fn driver_string(bytes: &[u8]) -> Result<String, CudaError> {
    CStr::from_bytes_until_nul(bytes)
        .map(|value| value.to_string_lossy().into_owned())
        .map_err(|_| CudaError::InvalidArgument("unterminated CUDA driver string"))
}

struct ContextInner {
    api: Rc<DriverApi>,
    raw: Handle,
    stream: Cell<Handle>,
    info: CudaDeviceInfo,
    limits: LaunchLimits,
    poisoned: Cell<bool>,
}

struct CurrentContext<'a> {
    inner: &'a ContextInner,
    active: bool,
}

impl CurrentContext<'_> {
    fn restore(mut self) -> Result<(), CudaError> {
        self.pop()
    }

    fn pop(&mut self) -> Result<(), CudaError> {
        if !self.active {
            return Ok(());
        }
        self.active = false;
        let mut popped = ptr::null_mut();
        let result = check(
            unsafe { (self.inner.api.context_pop)(&mut popped) },
            "cuCtxPopCurrent",
        );
        if result.is_err() || popped != self.inner.raw {
            self.inner.poisoned.set(true);
        }
        result?;
        if popped != self.inner.raw {
            return Err(CudaError::InvalidArgument(
                "CUDA context stack changed unexpectedly",
            ));
        }
        Ok(())
    }
}

impl Drop for CurrentContext<'_> {
    fn drop(&mut self) {
        let _ = self.pop();
    }
}

impl ContextInner {
    fn enter(&self, allow_poisoned: bool) -> Result<CurrentContext<'_>, CudaError> {
        if !allow_poisoned && self.poisoned.get() {
            return Err(CudaError::Poisoned);
        }
        check(
            unsafe { (self.api.context_push)(self.raw) },
            "cuCtxPushCurrent",
        )?;
        Ok(CurrentContext {
            inner: self,
            active: true,
        })
    }

    fn with_current<T>(
        &self,
        operation: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        let guard = self.enter(false)?;
        let result = operation();
        let restoration = guard.restore();
        match (result, restoration) {
            (Err(error), _) | (_, Err(error)) => Err(error),
            (Ok(value), Ok(())) => Ok(value),
        }
    }

    fn synchronize(&self) -> Result<(), CudaError> {
        let result = check(
            unsafe { (self.api.stream_synchronize)(self.stream.get()) },
            "cuStreamSynchronize",
        );
        if result.is_err() {
            self.poisoned.set(true);
        }
        result
    }

    fn synchronize_context(&self) -> Result<(), CudaError> {
        let result = check(
            unsafe { (self.api.context_synchronize)() },
            "cuCtxSynchronize",
        );
        if result.is_err() {
            self.poisoned.set(true);
        }
        result
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        let cleaned = (|| {
            let guard = self.enter(true)?;
            self.synchronize_context()?;
            if !self.stream.get().is_null() {
                check(
                    unsafe { (self.api.stream_destroy)(self.stream.get()) },
                    "cuStreamDestroy",
                )?;
            }
            guard.restore()?;
            check(
                unsafe { (self.api.context_destroy)(self.raw) },
                "cuCtxDestroy",
            )
        })();
        if cleaned.is_err() {
            // Keep driver code loaded for resources whose completion or destruction is uncertain.
            std::mem::forget(Rc::clone(&self.api));
        }
    }
}

/// An owned device context and dedicated stream. Contexts and their resources stay on one thread.
#[derive(Clone)]
pub struct CudaContext {
    inner: Rc<ContextInner>,
}

impl fmt::Debug for CudaContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaContext")
            .field("device", &self.inner.info)
            .field("poisoned", &self.inner.poisoned.get())
            .finish()
    }
}

impl CudaContext {
    pub fn new(ordinal: i32) -> Result<Self, CudaError> {
        if ordinal < 0 {
            return Err(CudaError::InvalidArgument(
                "CUDA device ordinal must be nonnegative",
            ));
        }
        let api = Rc::new(DriverApi::load()?);
        check(unsafe { (api.init)(0) }, "cuInit")?;
        let mut device = 0;
        check(
            unsafe { (api.device_get)(&mut device, ordinal) },
            "cuDeviceGet",
        )?;
        let mut name = [0u8; 256];
        check(
            unsafe { (api.device_name)(name.as_mut_ptr().cast(), name.len() as i32, device) },
            "cuDeviceGetName",
        )?;
        let mut pci = [0u8; 32];
        check(
            unsafe { (api.device_pci_bus_id)(pci.as_mut_ptr().cast(), pci.len() as i32, device) },
            "cuDeviceGetPCIBusId",
        )?;
        let info = CudaDeviceInfo {
            name: driver_string(&name)?,
            ordinal,
            compute_capability: (
                device_attribute(&api, device, 75)?,
                device_attribute(&api, device, 76)?,
            ),
            pci_bus_id: driver_string(&pci)?,
        };
        let limits = LaunchLimits {
            threads_per_block: device_attribute(&api, device, 1)?,
            block: [
                device_attribute(&api, device, 2)?,
                device_attribute(&api, device, 3)?,
                device_attribute(&api, device, 4)?,
            ],
            grid: [
                device_attribute(&api, device, 5)?,
                device_attribute(&api, device, 6)?,
                device_attribute(&api, device, 7)?,
            ],
            shared_memory: device_attribute(&api, device, 8)?,
        };
        let mut raw = ptr::null_mut();
        check(
            unsafe { (api.context_create)(&mut raw, 0, device) },
            "cuCtxCreate",
        )?;
        let mut popped = ptr::null_mut();
        let pop_result = check(unsafe { (api.context_pop)(&mut popped) }, "cuCtxPopCurrent");
        if pop_result.is_err() || popped != raw {
            if unsafe { (api.context_destroy)(raw) } != 0 {
                std::mem::forget(Rc::clone(&api));
            }
            pop_result?;
            return Err(CudaError::InvalidArgument(
                "CUDA context creation did not restore the context stack",
            ));
        }
        let inner = Rc::new(ContextInner {
            api,
            raw,
            stream: Cell::new(ptr::null_mut()),
            info,
            limits,
            poisoned: Cell::new(false),
        });
        inner.with_current(|| {
            let mut stream = ptr::null_mut();
            check(
                unsafe { (inner.api.stream_create)(&mut stream, 1) },
                "cuStreamCreate",
            )?;
            inner.stream.set(stream);
            Ok(())
        })?;
        Ok(Self { inner })
    }

    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.inner.info
    }

    /// Allocate a nonempty, zero-initialized device buffer.
    pub fn allocate(&self, size: usize) -> Result<CudaBuffer, CudaError> {
        if size == 0 {
            return Err(CudaError::InvalidArgument(
                "CUDA allocation size must be nonzero",
            ));
        }
        self.inner.with_current(|| {
            let mut pointer = 0;
            check(
                unsafe { (self.inner.api.mem_alloc)(&mut pointer, size) },
                "cuMemAlloc",
            )?;
            let buffer = CudaBuffer {
                context: self.clone(),
                pointer,
                size,
            };
            let initialization = check(
                unsafe { (self.inner.api.mem_zero)(pointer, 0, size, self.inner.stream.get()) },
                "cuMemsetD8Async",
            );
            let completion = self.inner.synchronize();
            initialization?;
            completion?;
            Ok(buffer)
        })
    }

    /// Load a PTX module into this context.
    ///
    /// # Safety
    /// PTX is native, unvalidated code. The caller must trust the module and enforce its
    /// memory-access and launch contracts; Etalon's shader validation does not apply.
    pub unsafe fn load_ptx(&self, ptx: &str) -> Result<CudaModule, CudaError> {
        let ptx = CString::new(ptx)
            .map_err(|_| CudaError::InvalidArgument("PTX contains an interior NUL"))?;
        self.inner.with_current(|| {
            let mut module = ptr::null_mut();
            check(
                unsafe { (self.inner.api.module_load)(&mut module, ptx.as_ptr().cast()) },
                "cuModuleLoadData",
            )?;
            Ok(CudaModule {
                context: self.clone(),
                raw: module,
            })
        })
    }
}

/// An owned device allocation. Safe transfers operate on a checked prefix.
pub struct CudaBuffer {
    context: CudaContext,
    pointer: DevicePointer,
    size: usize,
}

impl fmt::Debug for CudaBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBuffer")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

fn validate_transfer(capacity: usize, length: usize) -> Result<(), CudaError> {
    if length > capacity {
        Err(CudaError::InvalidArgument(
            "CUDA transfer exceeds buffer length",
        ))
    } else {
        Ok(())
    }
}

impl CudaBuffer {
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn write(&self, bytes: &[u8]) -> Result<(), CudaError> {
        validate_transfer(self.size, bytes.len())?;
        self.context.inner.with_current(|| {
            if bytes.is_empty() {
                return Ok(());
            }
            let copy = check(
                unsafe {
                    (self.context.inner.api.copy_htod)(
                        self.pointer,
                        bytes.as_ptr().cast(),
                        bytes.len(),
                    )
                },
                "cuMemcpyHtoD",
            );
            let completion = self.context.inner.synchronize_context();
            copy.and(completion)
        })
    }

    pub fn read(&self, bytes: &mut [u8]) -> Result<(), CudaError> {
        validate_transfer(self.size, bytes.len())?;
        self.context.inner.with_current(|| {
            if bytes.is_empty() {
                return Ok(());
            }
            let copy = check(
                unsafe {
                    (self.context.inner.api.copy_dtoh)(
                        bytes.as_mut_ptr().cast(),
                        self.pointer,
                        bytes.len(),
                    )
                },
                "cuMemcpyDtoH",
            );
            let completion = self.context.inner.synchronize_context();
            copy.and(completion)
        })
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if let Ok(guard) = self.context.inner.enter(true) {
            if self.context.inner.synchronize_context().is_ok() {
                let _ = unsafe { (self.context.inner.api.mem_free)(self.pointer) };
            }
            let _ = guard.restore();
        }
    }
}

/// A loaded module retaining its context until all borrowed kernels are gone.
pub struct CudaModule {
    context: CudaContext,
    raw: Handle,
}

impl fmt::Debug for CudaModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaModule")
            .field("context", &self.context)
            .finish_non_exhaustive()
    }
}

impl CudaModule {
    pub fn kernel(&self, name: &str) -> Result<CudaKernel<'_>, CudaError> {
        let name = CString::new(name)
            .map_err(|_| CudaError::InvalidArgument("CUDA kernel name contains an interior NUL"))?;
        self.kernel_cstr(&name)
    }

    /// Look up a kernel without allocating a temporary C string.
    pub fn kernel_cstr(&self, name: &CStr) -> Result<CudaKernel<'_>, CudaError> {
        self.context.inner.with_current(|| {
            let mut raw = ptr::null_mut();
            check(
                unsafe {
                    (self.context.inner.api.module_function)(&mut raw, self.raw, name.as_ptr())
                },
                "cuModuleGetFunction",
            )?;
            let mut max_threads = 0;
            let mut static_shared = 0;
            check(
                unsafe { (self.context.inner.api.function_attribute)(&mut max_threads, 0, raw) },
                "cuFuncGetAttribute",
            )?;
            check(
                unsafe { (self.context.inner.api.function_attribute)(&mut static_shared, 1, raw) },
                "cuFuncGetAttribute",
            )?;
            Ok(CudaKernel {
                module: self,
                raw,
                max_threads: u32::try_from(max_threads)
                    .map_err(|_| CudaError::InvalidArgument("negative CUDA kernel thread limit"))?,
                static_shared: u32::try_from(static_shared).map_err(|_| {
                    CudaError::InvalidArgument("negative CUDA kernel shared memory size")
                })?,
            })
        })
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        if let Ok(guard) = self.context.inner.enter(true) {
            if self.context.inner.synchronize_context().is_ok() {
                let _ = unsafe { (self.context.inner.api.module_unload)(self.raw) };
            }
            let _ = guard.restore();
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum CudaArgument<'a> {
    Buffer(&'a CudaBuffer),
    U32(u32),
    F32(f32),
    /// By-value parameter bytes; the unsafe launch contract covers layout and size.
    Bytes(&'a [u8]),
}

#[derive(Debug)]
pub struct CudaKernel<'a> {
    module: &'a CudaModule,
    raw: Handle,
    max_threads: u32,
    static_shared: u32,
}

#[derive(Clone, Copy, Debug)]
enum ArgumentStorage {
    Pointer(DevicePointer),
    U32(u32),
    F32(f32),
}

impl ArgumentStorage {
    fn as_mut_ptr(&mut self) -> *mut c_void {
        match self {
            Self::Pointer(value) => ptr::from_mut(value).cast(),
            Self::U32(value) => ptr::from_mut(value).cast(),
            Self::F32(value) => ptr::from_mut(value).cast(),
        }
    }
}

/// Reusable host storage for at most the specified number of launch arguments.
#[derive(Debug)]
pub struct CudaLaunchWorkspace {
    storage: Vec<ArgumentStorage>,
    pointers: Vec<*mut c_void>,
}

impl CudaLaunchWorkspace {
    pub fn new(max_arguments: usize) -> Self {
        Self {
            storage: vec![ArgumentStorage::U32(0); max_arguments],
            pointers: vec![ptr::null_mut(); max_arguments],
        }
    }

    pub fn capacity(&self) -> usize {
        self.storage.len()
    }
}

fn validate_launch(
    limits: LaunchLimits,
    kernel_threads: u32,
    static_shared: u32,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
) -> Result<(), CudaError> {
    if grid
        .iter()
        .zip(limits.grid)
        .any(|(&value, max)| value == 0 || value > max)
        || block
            .iter()
            .zip(limits.block)
            .any(|(&value, max)| value == 0 || value > max)
    {
        return Err(CudaError::InvalidArgument(
            "CUDA launch dimensions exceed device limits",
        ));
    }
    let threads = block
        .into_iter()
        .try_fold(1u32, u32::checked_mul)
        .ok_or(CudaError::InvalidArgument("CUDA block dimensions overflow"))?;
    if threads > limits.threads_per_block.min(kernel_threads) {
        return Err(CudaError::InvalidArgument(
            "CUDA block exceeds kernel thread limit",
        ));
    }
    grid.into_iter()
        .try_fold(u64::from(threads), |total, value| {
            total.checked_mul(u64::from(value))
        })
        .ok_or(CudaError::InvalidArgument(
            "CUDA total thread count overflows",
        ))?;
    let shared = static_shared
        .checked_add(shared)
        .ok_or(CudaError::InvalidArgument(
            "CUDA shared memory size overflows",
        ))?;
    if shared > limits.shared_memory {
        return Err(CudaError::InvalidArgument(
            "CUDA shared memory exceeds device limit",
        ));
    }
    Ok(())
}

impl CudaKernel<'_> {
    /// Launch on the context's dedicated stream and wait for completion.
    ///
    /// # Safety
    /// The caller must match the kernel's argument count, order, and ABI types, and ensure
    /// every kernel access stays within its live buffer or module storage. Scalar values
    /// must not encode invalid pointers. All kernel accesses and synchronization must be
    /// valid for the supplied dimensions and shared memory, including accesses made by
    /// device-launched work. Untrusted kernels require independent validation or isolation.
    pub unsafe fn launch(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem_bytes: u32,
        args: &[CudaArgument<'_>],
    ) -> Result<(), CudaError> {
        let mut workspace = CudaLaunchWorkspace::new(args.len());
        unsafe { self.launch_with_workspace(grid, block, shared_mem_bytes, args, &mut workspace) }
    }

    /// Launch synchronously using reusable host argument storage.
    ///
    /// # Safety
    /// The argument, memory-access, and synchronization contracts of [`Self::launch`] apply.
    pub unsafe fn launch_with_workspace(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem_bytes: u32,
        args: &[CudaArgument<'_>],
        workspace: &mut CudaLaunchWorkspace,
    ) -> Result<(), CudaError> {
        let inner = &self.module.context.inner;
        validate_launch(
            inner.limits,
            self.max_threads,
            self.static_shared,
            grid,
            block,
            shared_mem_bytes,
        )?;
        if args.len() > workspace.capacity() {
            return Err(CudaError::InvalidArgument(
                "CUDA launch argument capacity exceeded",
            ));
        }
        for (index, argument) in args.iter().enumerate() {
            workspace.storage[index] = match argument {
                CudaArgument::Buffer(buffer) => {
                    if !Rc::ptr_eq(inner, &buffer.context.inner) {
                        return Err(CudaError::ForeignContext);
                    }
                    ArgumentStorage::Pointer(buffer.pointer)
                }
                CudaArgument::U32(value) => ArgumentStorage::U32(*value),
                CudaArgument::F32(value) => ArgumentStorage::F32(*value),
                CudaArgument::Bytes(value) => {
                    if value.is_empty() {
                        return Err(CudaError::InvalidArgument(
                            "CUDA by-value argument must be nonempty",
                        ));
                    }
                    workspace.pointers[index] = value.as_ptr().cast_mut().cast();
                    continue;
                }
            };
            workspace.pointers[index] = workspace.storage[index].as_mut_ptr();
        }
        inner.with_current(|| {
            let launch = check(
                unsafe {
                    (inner.api.launch_kernel)(
                        self.raw,
                        grid[0],
                        grid[1],
                        grid[2],
                        block[0],
                        block[1],
                        block[2],
                        shared_mem_bytes,
                        inner.stream.get(),
                        if args.is_empty() {
                            ptr::null_mut()
                        } else {
                            workspace.pointers.as_mut_ptr()
                        },
                        ptr::null_mut(),
                    )
                },
                "cuLaunchKernel",
            );
            let completion = inner.synchronize();
            launch.and(completion)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: LaunchLimits = LaunchLimits {
        block: [1024, 1024, 64],
        grid: [i32::MAX as u32, 65_535, 65_535],
        threads_per_block: 1024,
        shared_memory: 49_152,
    };

    #[test]
    fn checked_launch_rejects_invalid_dimensions_and_shared_memory() {
        assert!(validate_launch(LIMITS, 1024, 0, [1; 3], [32, 4, 1], 0).is_ok());
        for (grid, block) in [
            ([0, 1, 1], [1; 3]),
            ([1; 3], [1, 0, 1]),
            ([1; 3], [1025, 1, 1]),
            ([1, 65_536, 1], [1; 3]),
            ([1; 3], [64, 32, 1]),
        ] {
            assert!(validate_launch(LIMITS, 1024, 0, grid, block, 0).is_err());
        }
        assert!(validate_launch(LIMITS, 64, 0, [1; 3], [128, 1, 1], 0).is_err());
        assert!(validate_launch(LIMITS, 1024, 1024, [1; 3], [1; 3], 49_152).is_err());
        assert!(validate_launch(LIMITS, 1024, u32::MAX, [1; 3], [1; 3], 1).is_err());
    }

    #[test]
    fn checked_launch_rejects_overflow_without_wrapping() {
        let limits = LaunchLimits {
            block: [u32::MAX; 3],
            grid: [u32::MAX; 3],
            threads_per_block: u32::MAX,
            shared_memory: u32::MAX,
        };
        assert!(validate_launch(limits, u32::MAX, 0, [1; 3], [65_536, 65_536, 1], 0).is_err());
        assert!(validate_launch(limits, u32::MAX, 0, [u32::MAX; 3], [1; 3], 0).is_err());
    }

    #[test]
    fn transfer_prefix_is_bounded() {
        assert!(validate_transfer(16, 0).is_ok());
        assert!(validate_transfer(16, 16).is_ok());
        assert!(validate_transfer(16, 17).is_err());
        assert!(validate_transfer(16, usize::MAX).is_err());
    }

    #[test]
    fn negative_device_ordinal_is_rejected_before_driver_loading() {
        assert!(matches!(
            CudaContext::new(-1),
            Err(CudaError::InvalidArgument(_))
        ));
    }

    #[test]
    #[ignore = "requires an NVIDIA CUDA device and driver"]
    fn cuda_runtime_hardware_resources_and_context_restoration() {
        let ordinal = std::env::var("ETALON_CUDA_DEVICE")
            .map(|value| {
                value
                    .parse::<i32>()
                    .expect("ETALON_CUDA_DEVICE must be an integer")
            })
            .unwrap_or(0);
        let outer = CudaContext::new(ordinal).expect("CUDA device is required for qualification");
        eprintln!("CUDA runtime qualification: {:?}", outer.device_info());
        assert_eq!(outer.device_info().ordinal, ordinal);
        assert!(!outer.device_info().name.is_empty());
        assert!(!outer.device_info().pci_bus_id.is_empty());
        assert!(outer.device_info().compute_capability.0 > 0);

        type GetCurrent = unsafe extern "system" fn(*mut Handle) -> i32;
        let api = Rc::clone(&outer.inner.api);
        let get_current = unsafe {
            *api._library
                .get::<GetCurrent>(b"cuCtxGetCurrent\0")
                .unwrap()
        };
        let current = || {
            let mut context = ptr::null_mut();
            check(unsafe { get_current(&mut context) }, "cuCtxGetCurrent").unwrap();
            context
        };
        let previous = current();
        let outer_guard = outer.inner.enter(false).unwrap();
        assert_eq!(current(), outer.inner.raw);
        {
            let context = CudaContext::new(ordinal).unwrap();
            assert_eq!(current(), outer.inner.raw);
            assert!(matches!(
                CudaContext::new(-1),
                Err(CudaError::InvalidArgument(_))
            ));
            assert!(matches!(
                context.allocate(0),
                Err(CudaError::InvalidArgument(_))
            ));
            let buffer = context.allocate(16).unwrap();
            assert_eq!(buffer.len(), 16);
            assert!(!buffer.is_empty());
            let mut bytes = [0xff; 16];
            buffer.read(&mut bytes).unwrap();
            assert_eq!(bytes, [0; 16]);
            assert!(matches!(
                buffer.write(&[7; 17]),
                Err(CudaError::InvalidArgument(_))
            ));
            let mut oversized = [0xa5; 17];
            assert!(matches!(
                buffer.read(&mut oversized),
                Err(CudaError::InvalidArgument(_))
            ));
            assert_eq!(oversized, [0xa5; 17]);
            buffer.write(&[1, 2, 3, 4]).unwrap();
            buffer.read(&mut bytes).unwrap();
            assert_eq!(&bytes[..4], &[1, 2, 3, 4]);
            assert_eq!(&bytes[4..], &[0; 12]);
            buffer.write(&[]).unwrap();
            buffer.read(&mut []).unwrap();
            assert_eq!(current(), outer.inner.raw);

            let ptx = r#"
.version 6.0
.target sm_50
.address_size 64
.visible .entry write_value(.param .u64 output, .param .u32 value) {
    .reg .b64 %address;
    .reg .b32 %value;
    ld.param.u64 %address, [output];
    ld.param.u32 %value, [value];
    st.global.u32 [%address], %value;
    ret;
}
"#;
            // This fixture writes one u32 through its sole buffer argument from one thread.
            let module = unsafe { context.load_ptx(ptx) }.unwrap();
            assert!(module.kernel("missing_kernel").is_err());
            assert_eq!(current(), outer.inner.raw);
            let foreign = outer.allocate(4).unwrap();
            {
                let kernel = module.kernel("write_value").unwrap();
                let mut too_small = CudaLaunchWorkspace::new(1);
                assert!(matches!(
                    unsafe {
                        kernel.launch_with_workspace(
                            [1; 3],
                            [1; 3],
                            0,
                            &[CudaArgument::Buffer(&buffer), CudaArgument::U32(1)],
                            &mut too_small,
                        )
                    },
                    Err(CudaError::InvalidArgument(_))
                ));
                let rejected = unsafe {
                    kernel.launch(
                        [1; 3],
                        [1; 3],
                        0,
                        &[CudaArgument::Buffer(&foreign), CudaArgument::U32(1)],
                    )
                };
                assert!(matches!(rejected, Err(CudaError::ForeignContext)));
                assert_eq!(current(), outer.inner.raw);
                drop(context);
                unsafe {
                    kernel.launch(
                        [1; 3],
                        [1; 3],
                        0,
                        &[
                            CudaArgument::Buffer(&buffer),
                            CudaArgument::U32(0x1122_3344),
                        ],
                    )
                }
                .unwrap();
                buffer.read(&mut bytes).unwrap();
                assert_eq!(
                    u32::from_ne_bytes(bytes[..4].try_into().unwrap()),
                    0x1122_3344
                );
                assert_eq!(&bytes[4..], &[0; 12]);
                assert_eq!(current(), outer.inner.raw);
            }
            drop(module);
            drop(buffer);
            drop(foreign);
            assert_eq!(current(), outer.inner.raw);
        }
        outer_guard.restore().unwrap();
        assert_eq!(current(), previous);
        drop(outer);
        assert_eq!(current(), previous);
    }
}
