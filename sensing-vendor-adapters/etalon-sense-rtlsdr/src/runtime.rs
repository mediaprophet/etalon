use crate::{abi::Api, Error, RtlSdrDevice, RtlSdrGate};
use etalon_sense_deployment::rtlsdr::AdmittedRtlSdrDeployment;
use std::{
    cell::Cell,
    marker::PhantomData,
    panic::{catch_unwind, AssertUnwindSafe},
    rc::Rc,
    sync::Mutex,
};

pub(crate) static LIFECYCLE: Mutex<()> = Mutex::new(());

pub(crate) trait DeploymentLease {
    fn current(&self) -> bool;
}
impl DeploymentLease for AdmittedRtlSdrDeployment<'_, '_, '_> {
    fn current(&self) -> bool {
        self.revalidate_authority().is_ok()
    }
}

pub struct RtlSdrRuntime<'lease> {
    lease: &'lease dyn DeploymentLease,
    #[cfg(any(windows, unix))]
    library: Option<libloading::Library>,
    pub(crate) api: Api,
    pub(crate) target_id: u64,
    pub(crate) generation: u32,
    pub(crate) device_index: u32,
    pub(crate) max_pairs: usize,
    retain: Cell<bool>,
    claimed: Cell<bool>,
    outstanding: Cell<bool>,
    _thread: PhantomData<Rc<()>>,
}

impl<'lease> RtlSdrRuntime<'lease> {
    /// # Safety
    /// Trust the admitted library, constructors/destructors and complete native
    /// dependency closure to implement the pinned C ABI without retaining
    /// borrowed pointers, overrunning buffers or unwinding across C. Hold the
    /// immutable artifact/device/policy lease and no-egress isolation through
    /// every use. Prevent alternate code resolution and global runtime reuse.
    /// Other bindings must not access this device, use async operations, or
    /// overlap lifecycle calls outside host coordination. Native calls may block
    /// indefinitely and allocate beyond all caller budgets. Never reuse this
    /// generation for another instance, even after failed open or cleanup.
    /// The gain-list count must remain stable for an exclusively owned handle
    /// across its count/query calls, as in the pinned upstream implementation.
    pub unsafe fn load(
        deployment: &'lease AdmittedRtlSdrDeployment<'_, '_, '_>,
    ) -> Result<Self, Error> {
        #[cfg(not(any(windows, unix)))]
        {
            let _ = deployment;
            Err(Error::UnsupportedPlatform)
        }
        #[cfg(any(windows, unix))]
        {
            if !lease_current(deployment) {
                return Err(Error::Admission);
            }
            let _lifecycle = LIFECYCLE.lock().map_err(|_| Error::Synchronization)?;
            let library = unsafe { load_library(deployment.runtime_path()) }?;
            let api = match unsafe { Api::load(&library) } {
                Ok(api) => api,
                Err(error) => {
                    core::mem::forget(library);
                    return Err(error);
                }
            };
            drop(_lifecycle);
            if !lease_current(deployment) {
                core::mem::forget(library);
                return Err(Error::Admission);
            }
            Ok(Self {
                lease: deployment,
                library: Some(library),
                api,
                target_id: deployment.target_id(),
                generation: deployment.generation(),
                device_index: deployment.device_index(),
                max_pairs: deployment.max_pairs() as usize,
                retain: Cell::new(false),
                claimed: Cell::new(false),
                outstanding: Cell::new(false),
                _thread: PhantomData,
            })
        }
    }

    pub fn open<'runtime>(
        &'runtime self,
        gate: &impl RtlSdrGate,
    ) -> Result<RtlSdrDevice<'runtime, 'lease>, Error> {
        RtlSdrDevice::open(self, gate)
    }

    pub(crate) fn current(&self) -> bool {
        lease_current(self.lease)
    }
    pub(crate) fn retain(&self) {
        self.retain.set(true);
    }
    pub(crate) fn native_opened(&self) {
        self.outstanding.set(true);
    }
    pub(crate) fn native_closed(&self) {
        self.outstanding.set(false);
    }
    pub(crate) fn must_retain(&self) -> bool {
        self.retain.get() || self.outstanding.get()
    }
    pub(crate) fn claim(&self) -> Result<(), Error> {
        if self.claimed.replace(true) {
            Err(Error::InvalidState)
        } else {
            Ok(())
        }
    }
}

impl Drop for RtlSdrRuntime<'_> {
    fn drop(&mut self) {
        #[cfg(any(windows, unix))]
        if let Some(library) = self.library.take() {
            if self.must_retain() {
                core::mem::forget(library);
            } else {
                let _lifecycle = LIFECYCLE.lock().unwrap_or_else(|error| error.into_inner());
                drop(library);
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn mock_runtime(lease: &dyn DeploymentLease, api: Api) -> RtlSdrRuntime<'_> {
    RtlSdrRuntime {
        lease,
        #[cfg(any(windows, unix))]
        library: None,
        api,
        target_id: 7,
        generation: 1,
        device_index: 3,
        max_pairs: 8,
        retain: Cell::new(false),
        claimed: Cell::new(false),
        outstanding: Cell::new(false),
        _thread: PhantomData,
    }
}

fn lease_current(lease: &dyn DeploymentLease) -> bool {
    catch_unwind(AssertUnwindSafe(|| lease.current())).unwrap_or(false)
}

#[cfg(unix)]
unsafe fn load_library(path: &std::path::Path) -> Result<libloading::Library, Error> {
    unsafe {
        libloading::os::unix::Library::open(
            Some(path),
            libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_LOCAL,
        )
    }
    .map(Into::into)
    .map_err(|_| Error::LibraryLoad)
}

#[cfg(windows)]
unsafe fn load_library(path: &std::path::Path) -> Result<libloading::Library, Error> {
    // Search the DLL's own directory and the protected system directory.
    unsafe { libloading::os::windows::Library::load_with_flags(path, 0x0000_0100 | 0x0000_0800) }
        .map(Into::into)
        .map_err(|_| Error::LibraryLoad)
}
