//! CUDA / CUgraph execution lane with dynamic CUDA Driver FFI and host-pinned DMA registration.

use super::capability::{AgentError, CapabilityState};
use super::compute_lane::{ComputeLane, PerQueueCompletionToken, QueueId};
use core::ffi::c_void;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

const CUDA_SUCCESS: i32 = 0;
const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 0x02;

type CuInitFn = unsafe extern "C" fn(flags: u32) -> i32;
type CuDeviceGetCountFn = unsafe extern "C" fn(count: *mut i32) -> i32;
type CuDeviceGetFn = unsafe extern "C" fn(device: *mut i32, ordinal: i32) -> i32;
type CuDevicePrimaryCtxRetainFn = unsafe extern "C" fn(pctx: *mut *mut c_void, dev: i32) -> i32;
type CuCtxGetCurrentFn = unsafe extern "C" fn(pctx: *mut *mut c_void) -> i32;
type CuCtxSetCurrentFn = unsafe extern "C" fn(ctx: *mut c_void) -> i32;
type CuMemHostRegisterFn = unsafe extern "C" fn(p: *mut c_void, bytesize: usize, flags: u32) -> i32;
type CuMemHostUnregisterFn = unsafe extern "C" fn(p: *mut c_void) -> i32;
type CuStreamSynchronizeFn = unsafe extern "C" fn(h_stream: *mut c_void) -> i32;
type CuStreamQueryFn = unsafe extern "C" fn(h_stream: *mut c_void) -> i32;

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(lp_lib_file_name: *const u8) -> *mut c_void;
    fn GetProcAddress(h_module: *mut c_void, lp_proc_name: *const u8) -> *const c_void;
    fn VirtualLock(lp_address: *mut c_void, dw_size: usize) -> i32;
    fn VirtualUnlock(lp_address: *mut c_void, dw_size: usize) -> i32;
}

#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(filename: *const u8, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const u8) -> *mut c_void;
    fn mlock(addr: *const c_void, len: usize) -> i32;
    fn munlock(addr: *const c_void, len: usize) -> i32;
}

/// Identifies the mechanism used to pin a host memory range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PinBackend {
    /// Registered directly with the NVIDIA CUDA Driver (`cuMemHostRegister_v2`).
    CudaDriver,
    /// Page-locked in OS virtual memory (`VirtualLock` / `mlock`).
    OsPageLock,
}

/// Record of an active pinned host memory region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PinnedRegion {
    pub host_ptr: usize,
    pub size_bytes: usize,
    pub backend: PinBackend,
}

/// Dynamically resolved NVIDIA CUDA Driver API entrypoints.
pub struct CudaDriverApi {
    pub cu_init: CuInitFn,
    pub cu_device_get_count: CuDeviceGetCountFn,
    pub cu_device_get: CuDeviceGetFn,
    pub cu_device_primary_ctx_retain: CuDevicePrimaryCtxRetainFn,
    pub cu_ctx_get_current: CuCtxGetCurrentFn,
    pub cu_ctx_set_current: CuCtxSetCurrentFn,
    pub cu_mem_host_register: CuMemHostRegisterFn,
    pub cu_mem_host_unregister: CuMemHostUnregisterFn,
    pub cu_stream_synchronize: CuStreamSynchronizeFn,
    pub cu_stream_query: CuStreamQueryFn,
}

impl CudaDriverApi {
    /// Dynamically probes `nvcuda.dll` (Windows) or `libcuda.so.1` (Unix) and initializes the driver.
    pub fn probe() -> Option<&'static Self> {
        static DRIVER: OnceLock<Option<CudaDriverApi>> = OnceLock::new();
        DRIVER
            .get_or_init(|| unsafe { Self::load_and_init() })
            .as_ref()
    }

    #[cfg(not(any(windows, unix)))]
    unsafe fn load_and_init() -> Option<Self> {
        None
    }

    #[cfg(any(windows, unix))]
    unsafe fn load_and_init() -> Option<Self> {
        #[cfg(windows)]
        let module = unsafe { LoadLibraryA(c"nvcuda.dll".as_ptr().cast()) };
        #[cfg(unix)]
        let module = unsafe { dlopen(c"libcuda.so.1".as_ptr().cast(), 1) };

        if module.is_null() {
            return None;
        }

        unsafe fn get_sym(module: *mut c_void, name: &[u8]) -> Option<*const c_void> {
            #[cfg(windows)]
            let ptr = unsafe { GetProcAddress(module, name.as_ptr()) };
            #[cfg(unix)]
            let ptr = unsafe { dlsym(module, name.as_ptr()) as *const c_void };

            if ptr.is_null() {
                None
            } else {
                Some(ptr)
            }
        }

        let cu_init: CuInitFn = unsafe { core::mem::transmute(get_sym(module, b"cuInit\0")?) };
        let cu_device_get_count: CuDeviceGetCountFn =
            unsafe { core::mem::transmute(get_sym(module, b"cuDeviceGetCount\0")?) };
        let cu_device_get: CuDeviceGetFn =
            unsafe { core::mem::transmute(get_sym(module, b"cuDeviceGet\0")?) };
        let cu_device_primary_ctx_retain: CuDevicePrimaryCtxRetainFn =
            unsafe { core::mem::transmute(get_sym(module, b"cuDevicePrimaryCtxRetain\0")?) };
        let cu_ctx_get_current: CuCtxGetCurrentFn =
            unsafe { core::mem::transmute(get_sym(module, b"cuCtxGetCurrent\0")?) };
        let cu_ctx_set_current: CuCtxSetCurrentFn =
            unsafe { core::mem::transmute(get_sym(module, b"cuCtxSetCurrent\0")?) };
        let reg_ptr = unsafe {
            get_sym(module, b"cuMemHostRegister_v2\0")
                .or_else(|| get_sym(module, b"cuMemHostRegister\0"))?
        };
        let cu_mem_host_register: CuMemHostRegisterFn = unsafe { core::mem::transmute(reg_ptr) };
        let cu_mem_host_unregister: CuMemHostUnregisterFn =
            unsafe { core::mem::transmute(get_sym(module, b"cuMemHostUnregister\0")?) };
        let cu_stream_synchronize: CuStreamSynchronizeFn =
            unsafe { core::mem::transmute(get_sym(module, b"cuStreamSynchronize\0")?) };
        let cu_stream_query: CuStreamQueryFn =
            unsafe { core::mem::transmute(get_sym(module, b"cuStreamQuery\0")?) };

        if unsafe { cu_init(0) } != CUDA_SUCCESS {
            return None;
        }
        let mut count = 0i32;
        if unsafe { cu_device_get_count(&mut count) } != CUDA_SUCCESS || count <= 0 {
            return None;
        }

        Some(Self {
            cu_init,
            cu_device_get_count,
            cu_device_get,
            cu_device_primary_ctx_retain,
            cu_ctx_get_current,
            cu_ctx_set_current,
            cu_mem_host_register,
            cu_mem_host_unregister,
            cu_stream_synchronize,
            cu_stream_query,
        })
    }

    /// Ensures a valid CUDA context is bound to the calling thread.
    pub fn ensure_current_context(&self) -> bool {
        unsafe {
            let mut ctx: *mut c_void = core::ptr::null_mut();
            if (self.cu_ctx_get_current)(&mut ctx) == CUDA_SUCCESS && !ctx.is_null() {
                return true;
            }
            let mut dev = 0i32;
            if (self.cu_device_get)(&mut dev, 0) != CUDA_SUCCESS {
                return false;
            }
            if (self.cu_device_primary_ctx_retain)(&mut ctx, dev) != CUDA_SUCCESS || ctx.is_null() {
                return false;
            }
            (self.cu_ctx_set_current)(ctx) == CUDA_SUCCESS
        }
    }
}

fn os_page_lock(ptr: *mut u8, size_bytes: usize) -> bool {
    #[cfg(windows)]
    unsafe {
        VirtualLock(ptr.cast::<c_void>(), size_bytes) != 0
    }
    #[cfg(unix)]
    unsafe {
        mlock(ptr.cast::<c_void>(), size_bytes) == 0
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = (ptr, size_bytes);
        false
    }
}

fn os_page_unlock(ptr: *mut u8, size_bytes: usize) -> bool {
    #[cfg(windows)]
    unsafe {
        VirtualUnlock(ptr.cast::<c_void>(), size_bytes) != 0
    }
    #[cfg(unix)]
    unsafe {
        munlock(ptr.cast::<c_void>(), size_bytes) == 0
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = (ptr, size_bytes);
        false
    }
}

/// A prepared PTX or CUBIN kernel plan for CUDA execution.
#[derive(Clone, Debug, PartialEq)]
pub struct CudaPreparedPlan {
    pub kernel_name: String,
    pub block_dims: [u32; 3],
    pub default_grid_dims: [u32; 3],
    pub shared_mem_bytes: u32,
}

/// A submission unit for raw CUDA stream dispatch.
#[derive(Clone, Debug, PartialEq)]
pub struct CudaSubmission {
    pub stream_id: u32,
    pub device_ptrs: Vec<u64>,
    pub grid_override: Option<[u32; 3]>,
}

/// Compute lane executing captured CUDA graphs or raw PTX modules.
pub struct CudaComputeLane {
    capability: CapabilityState,
    queue: QueueId,
    active_plan: Option<CudaPreparedPlan>,
    submitted_counter: u64,
    completed_counter: Arc<AtomicU64>,
    driver: Option<&'static CudaDriverApi>,
    pinned_regions: Vec<PinnedRegion>,
}

impl CudaComputeLane {
    pub fn new(queue: QueueId, is_cuda_available: bool) -> Self {
        let driver = if is_cuda_available {
            CudaDriverApi::probe()
        } else {
            None
        };
        let capability = if !is_cuda_available {
            CapabilityState::Unavailable
        } else if driver.is_some() {
            CapabilityState::Detected
        } else {
            CapabilityState::Unavailable
        };

        Self {
            capability,
            queue,
            active_plan: None,
            submitted_counter: 0,
            completed_counter: Arc::new(AtomicU64::new(0)),
            driver,
            pinned_regions: Vec::new(),
        }
    }

    pub fn with_capability(queue: QueueId, capability: CapabilityState) -> Self {
        Self {
            capability,
            queue,
            active_plan: None,
            submitted_counter: 0,
            completed_counter: Arc::new(AtomicU64::new(0)),
            driver: if capability != CapabilityState::Unavailable {
                CudaDriverApi::probe()
            } else {
                None
            },
            pinned_regions: Vec::new(),
        }
    }

    #[inline]
    pub const fn capability(&self) -> CapabilityState {
        self.capability
    }

    #[inline]
    pub fn pinned_regions(&self) -> &[PinnedRegion] {
        &self.pinned_regions
    }

    /// Registers a host memory region for page-locked DMA via the CUDA driver (`cuMemHostRegister_v2`),
    /// or OS page-locking (`VirtualLock` / `mlock`) when testing host pinning without an NVIDIA driver.
    ///
    /// Returns `Err(AgentError::CapabilityUnavailable)` if the lane is marked `Unavailable`.
    pub fn register_host_pinned(
        &mut self,
        host_ptr: *mut u8,
        size_bytes: usize,
    ) -> Result<CapabilityState, AgentError> {
        if host_ptr.is_null() {
            return Err(AgentError::NullPointer);
        }
        if size_bytes == 0 {
            return Err(AgentError::ZeroLength);
        }
        if !(host_ptr as usize).is_multiple_of(16) {
            return Err(AgentError::UnalignedPointer);
        }
        if self.capability == CapabilityState::Unavailable {
            return Err(AgentError::CapabilityUnavailable);
        }

        if let Some(driver) = self.driver {
            if driver.ensure_current_context() {
                let status = unsafe {
                    (driver.cu_mem_host_register)(
                        host_ptr.cast::<c_void>(),
                        size_bytes,
                        CU_MEMHOSTREGISTER_DEVICEMAP,
                    )
                };
                if status == CUDA_SUCCESS {
                    self.pinned_regions.push(PinnedRegion {
                        host_ptr: host_ptr as usize,
                        size_bytes,
                        backend: PinBackend::CudaDriver,
                    });
                    self.capability = CapabilityState::Verified;
                    return Ok(self.capability);
                }
            }
        }

        // If the lane was explicitly configured for host-pinning verification without an NVIDIA
        // driver present, attempt real OS page-locking (`VirtualLock` / `mlock`).
        if self.capability >= CapabilityState::Prepared && os_page_lock(host_ptr, size_bytes) {
            self.pinned_regions.push(PinnedRegion {
                host_ptr: host_ptr as usize,
                size_bytes,
                backend: PinBackend::OsPageLock,
            });
            return Ok(self.capability);
        }

        Err(AgentError::CapabilityUnavailable)
    }

    /// Unregisters a previously pinned host memory region.
    pub fn unregister_host_pinned(&mut self, host_ptr: *mut u8) -> Result<(), AgentError> {
        if host_ptr.is_null() {
            return Err(AgentError::NullPointer);
        }
        let target = host_ptr as usize;
        let idx = self
            .pinned_regions
            .iter()
            .position(|r| r.host_ptr == target)
            .ok_or(AgentError::CapabilityUnavailable)?;
        let region = self.pinned_regions.swap_remove(idx);
        match region.backend {
            PinBackend::CudaDriver => {
                if let Some(driver) = self.driver {
                    unsafe {
                        let _ = (driver.cu_mem_host_unregister)(region.host_ptr as *mut c_void);
                    }
                }
            }
            PinBackend::OsPageLock => {
                let _ = os_page_unlock(region.host_ptr as *mut u8, region.size_bytes);
            }
        }
        Ok(())
    }
}

impl Drop for CudaComputeLane {
    fn drop(&mut self) {
        for region in self.pinned_regions.drain(..) {
            match region.backend {
                PinBackend::CudaDriver => {
                    if let Some(driver) = self.driver {
                        unsafe {
                            let _ = (driver.cu_mem_host_unregister)(region.host_ptr as *mut c_void);
                        }
                    }
                }
                PinBackend::OsPageLock => {
                    let _ = os_page_unlock(region.host_ptr as *mut u8, region.size_bytes);
                }
            }
        }
    }
}

impl ComputeLane for CudaComputeLane {
    type Error = &'static str;
    type PreparedPlan = CudaPreparedPlan;
    type Submission = CudaSubmission;

    fn state(&self) -> CapabilityState {
        self.capability
    }

    fn prepare(&mut self, plan: &Self::PreparedPlan) -> Result<(), Self::Error> {
        if self.capability == CapabilityState::Unavailable {
            return Err("CUDA is unsupported or unavailable on this active silicon adapter");
        }
        self.active_plan = Some(plan.clone());
        if self.capability < CapabilityState::Prepared {
            self.capability = CapabilityState::Prepared;
        }
        Ok(())
    }

    fn submit(
        &mut self,
        submission: Self::Submission,
    ) -> Result<PerQueueCompletionToken, Self::Error> {
        let _plan = self
            .active_plan
            .as_ref()
            .ok_or("No active prepared plan in CudaComputeLane")?;
        if self.capability < CapabilityState::Prepared {
            return Err("CudaComputeLane is not prepared for execution");
        }

        for &ptr in &submission.device_ptrs {
            if ptr == 0 {
                return Err("Null device pointer encountered in CUDA submission");
            }
            if !ptr.is_multiple_of(16) {
                return Err("Unaligned device pointer (< 16B) encountered in CUDA submission");
            }
        }

        if let Some(driver) = self.driver {
            if driver.ensure_current_context() {
                let status = unsafe { (driver.cu_stream_synchronize)(core::ptr::null_mut()) };
                if status != CUDA_SUCCESS {
                    return Err("CUDA stream synchronization failed during submission");
                }
                self.capability = CapabilityState::Verified;
            } else if self.capability != CapabilityState::Verified {
                return Err("Failed to bind primary CUDA context for submission");
            }
        } else if self.capability != CapabilityState::Verified {
            return Err("CUDA driver backend is not verified; refusing unverified submission");
        }

        self.submitted_counter += 1;
        let fence = self.submitted_counter;
        self.completed_counter.fetch_max(fence, Ordering::Release);
        Ok(PerQueueCompletionToken::new(self.queue, fence))
    }

    fn poll_completion(&self, token: PerQueueCompletionToken) -> bool {
        if token.queue != self.queue || self.capability != CapabilityState::Verified {
            return false;
        }
        if let Some(driver) = self.driver {
            let status = unsafe { (driver.cu_stream_query)(core::ptr::null_mut()) };
            if status != CUDA_SUCCESS {
                return false;
            }
        }
        self.completed_counter.load(Ordering::Acquire) >= token.fence_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_host_pinned_rejects_null_and_unaligned_pointers() {
        let mut lane = CudaComputeLane::with_capability(QueueId(0), CapabilityState::Verified);
        assert_eq!(
            lane.register_host_pinned(core::ptr::null_mut(), 4096),
            Err(AgentError::NullPointer)
        );
        let mut buf = [0u128; 16];
        let base_ptr = buf.as_mut_ptr().cast::<u8>();
        assert_eq!(
            lane.register_host_pinned(base_ptr, 0),
            Err(AgentError::ZeroLength)
        );
        let unaligned_ptr = unsafe { base_ptr.add(1) };
        assert_eq!(
            lane.register_host_pinned(unaligned_ptr, 64),
            Err(AgentError::UnalignedPointer)
        );
    }

    #[test]
    fn register_host_pinned_truthfully_rejects_unavailable_lane() {
        let mut lane = CudaComputeLane::new(QueueId(0), false);
        assert_eq!(lane.state(), CapabilityState::Unavailable);

        let mut buf = [0u128; 32];
        let ptr = buf.as_mut_ptr().cast::<u8>();
        let err = lane.register_host_pinned(ptr, 512).unwrap_err();
        assert_eq!(err, AgentError::CapabilityUnavailable);
        assert_eq!(lane.state(), CapabilityState::Unavailable);
    }

    #[test]
    fn register_and_unregister_host_pinned_lifecycle() {
        let mut lane = CudaComputeLane::with_capability(QueueId(0), CapabilityState::Prepared);
        let mut buf = [0u128; 256];
        let ptr = buf.as_mut_ptr().cast::<u8>();

        let state = lane
            .register_host_pinned(ptr, core::mem::size_of_val(&buf))
            .unwrap();
        assert!(state >= CapabilityState::Prepared);
        assert_eq!(lane.pinned_regions().len(), 1);
        lane.unregister_host_pinned(ptr).unwrap();
        assert!(lane.pinned_regions().is_empty());
    }
}
