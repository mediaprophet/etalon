use crate::{
    abi::{self, RemoteArg, RemoteBuf},
    FastRpcError,
};
use etalon_sense_deployment::fastrpc::AdmittedFastRpcDeployment;
use std::{
    marker::PhantomData,
    panic::{catch_unwind, AssertUnwindSafe},
    rc::Rc,
    sync::Mutex,
};

// remote.h requires coordination across independently owned open/close calls.
static RUNTIME_LIFECYCLE: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionState {
    Open,
    Faulted,
    Closed,
    CloseFailed,
}

trait DeploymentLease {
    fn current(&self) -> bool;
}

impl DeploymentLease for AdmittedFastRpcDeployment<'_, '_, '_> {
    fn current(&self) -> bool {
        self.revalidate_authority().is_ok()
    }
}

/// One thread-affine, synchronous handle; dropping it attempts one close.
/// A failed close has an unknown remote outcome and is never retried. The
/// runtime mapping is retained in that case because it may own live resources.
pub struct FastRpcSession<'lease> {
    lease: &'lease dyn DeploymentLease,
    #[cfg(any(target_os = "linux", target_os = "android"))]
    library: Option<libloading::Library>,
    invoke: abi::Invoke,
    close: abi::Close,
    handle: Option<u64>,
    state: SessionState,
    target_id: u64,
    generation: u32,
    domain: u32,
    session: u32,
    max_pairs: u32,
    _thread_affinity: PhantomData<Rc<()>>,
}

impl<'lease> FastRpcSession<'lease> {
    /// Loads the exact admitted runtime and opens the admitted module URI.
    ///
    /// # Safety
    /// The caller must trust this runtime, its entire dependency closure,
    /// constructors/destructors and selected service to obey the pinned C ABI.
    /// It must implement the exact Etalon service schema, neither mutate input
    /// buffers nor access outside slices, retain pointers after return or unwind
    /// across C. The admission authority must hold an immutable artifact and
    /// policy lease for this session, prevent library-name interposition and
    /// process-global runtime reuse from selecting other code, and authenticate
    /// URI-to-module/export/domain/session resolution. Hash equality alone does
    /// not establish any of these properties. Native execution can block and
    /// allocate beyond the caller buffers; establish isolation before loading.
    /// Other FastRPC bindings in the process must join the host's lifecycle
    /// coordination: this crate cannot serialize unrelated native callers.
    /// Exclude other bindings, isolate this provider in a process, or hold an
    /// outer coordinator around all wrapper/native lifecycle calls and drops.
    /// Successful close/unload must leave no outstanding native references.
    pub unsafe fn open(
        deployment: &'lease AdmittedFastRpcDeployment<'_, '_, '_>,
        max_pairs: u32,
    ) -> Result<Self, FastRpcError> {
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            let _ = (deployment, max_pairs);
            Err(FastRpcError::UnsupportedPlatform)
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            use etalon_sense_dsp_wire::{MAX_PAIRS, SCHEMA_SHA256, SERVICE_ID};
            if max_pairs == 0 || max_pairs > MAX_PAIRS || max_pairs > deployment.service_max_pairs()
            {
                return Err(FastRpcError::InvalidLimits);
            }
            if deployment.service_schema_id() != SERVICE_ID
                || deployment.service_schema_sha256() != SCHEMA_SHA256
            {
                return Err(FastRpcError::ServiceSchema);
            }
            let uri_generation =
                etalon_sense_dsp_wire::generation_from_uri(deployment.module_uri().to_bytes())
                    .map_err(|_| FastRpcError::ServiceSchema)?;
            if uri_generation != deployment.provider_generation() {
                return Err(FastRpcError::DeploymentAdmission);
            }
            if !lease_current(deployment) {
                return Err(FastRpcError::DeploymentAdmission);
            }
            let lifecycle = RUNTIME_LIFECYCLE
                .lock()
                .map_err(|_| FastRpcError::RuntimeSynchronization)?;
            // Absolute protected path and closure were admitted; NOW avoids
            // deferred symbol resolution. It is not an isolation mechanism.
            let library: libloading::Library = unsafe {
                libloading::os::unix::Library::open(
                    Some(deployment.runtime_path()),
                    libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_LOCAL,
                )
            }
            .map_err(|_| FastRpcError::LibraryLoad)?
            .into();
            let (open, invoke, close) = match unsafe { symbols(&library) } {
                Ok(symbols) => symbols,
                Err(error) => {
                    core::mem::forget(library);
                    return Err(error);
                }
            };
            drop(lifecycle);
            if !lease_current(deployment) {
                core::mem::forget(library);
                return Err(FastRpcError::DeploymentAdmission);
            }
            let lifecycle = match RUNTIME_LIFECYCLE.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    core::mem::forget(library);
                    return Err(FastRpcError::RuntimeSynchronization);
                }
            };
            let mut handle = 0u64;
            let status = unsafe { open(deployment.module_uri().as_ptr(), &mut handle) };
            if status != 0 {
                // An unsuccessful native open may still have runtime-global
                // state. Keep code mapped rather than guessing its teardown.
                core::mem::forget(library);
                return Err(FastRpcError::Open(status));
            }
            drop(lifecycle);
            let mut owner = Self {
                lease: deployment,
                library: Some(library),
                invoke,
                close,
                handle: Some(handle),
                state: SessionState::Open,
                target_id: deployment.target_id(),
                generation: deployment.provider_generation(),
                domain: deployment.domain(),
                session: deployment.session(),
                max_pairs,
                _thread_affinity: PhantomData,
            };
            if !owner.authority_current() {
                if let Err(FastRpcError::Close(code)) = owner.close() {
                    return Err(FastRpcError::AdmissionCleanupFailed(code));
                }
                return Err(FastRpcError::DeploymentAdmission);
            }
            Ok(owner)
        }
    }

    pub fn state(&self) -> SessionState {
        self.state
    }
    pub fn target_id(&self) -> u64 {
        self.target_id
    }
    pub fn domain(&self) -> u32 {
        self.domain
    }
    pub fn session(&self) -> u32 {
        self.session
    }
    pub fn max_pairs(&self) -> u32 {
        self.max_pairs
    }

    pub(crate) fn authority_current(&self) -> bool {
        lease_current(self.lease)
    }

    pub(crate) fn invoke_buffers(
        &mut self,
        request: &[u8],
        response: &mut [u8],
    ) -> Result<(), FastRpcError> {
        response.fill(0);
        let result = self.invoke_inner(request, response);
        if result.is_err() {
            response.fill(0);
            if self.state == SessionState::Open {
                self.state = SessionState::Faulted;
            }
        }
        result
    }

    fn invoke_inner(&mut self, request: &[u8], response: &mut [u8]) -> Result<(), FastRpcError> {
        use etalon_sense_dsp_wire::{request_bytes, response_bytes};
        if self.state != SessionState::Open {
            return Err(FastRpcError::InvalidState);
        }
        if !self.authority_current() {
            return Err(FastRpcError::DeploymentAdmission);
        }
        let max_in = request_bytes(self.max_pairs).map_err(|_| FastRpcError::InvalidLimits)?;
        let max_out = response_bytes(self.max_pairs).map_err(|_| FastRpcError::InvalidLimits)?;
        if request.is_empty()
            || response.is_empty()
            || request.len() > max_in
            || response.len() > max_out
        {
            return Err(FastRpcError::BufferSize);
        }
        let (identity, pairs, _) =
            etalon_sense_dsp_wire::decode_request(request).map_err(|_| FastRpcError::Wire)?;
        if identity.provider_id() != self.target_id
            || identity.generation() != self.generation
            || pairs > self.max_pairs
            || response.len() != response_bytes(pairs).map_err(|_| FastRpcError::Wire)?
        {
            return Err(FastRpcError::Wire);
        }
        let scalars = abi::buffer_scalars(etalon_sense_dsp_wire::METHOD_INDEX, 0)
            .ok_or(FastRpcError::ServiceSchema)?;
        if scalars != etalon_sense_dsp_wire::METHOD_SCALARS {
            return Err(FastRpcError::ServiceSchema);
        }
        let input_ptr = request.as_ptr().cast_mut().cast();
        let output_ptr = response.as_mut_ptr().cast();
        let mut args = [
            RemoteArg {
                buf: RemoteBuf {
                    pv: input_ptr,
                    len: request.len(),
                },
            },
            RemoteArg {
                buf: RemoteBuf {
                    pv: output_ptr,
                    len: response.len(),
                },
            },
        ];
        let handle = self.handle.ok_or(FastRpcError::InvalidState)?;
        // &mut self serializes close/reset; both buffers remain borrowed until
        // synchronous return. The unsafe constructor established native trust.
        let status = unsafe { (self.invoke)(handle, scalars, args.as_mut_ptr()) };
        if status != 0 {
            return Err(FastRpcError::Invoke(status));
        }
        let (input_arg, output_arg) = unsafe { (args[0].buf, args[1].buf) };
        if input_arg.pv != input_ptr
            || input_arg.len != request.len()
            || output_arg.pv != output_ptr
            || output_arg.len != response.len()
        {
            return Err(FastRpcError::ArgumentMutation);
        }
        if !self.authority_current() {
            return Err(FastRpcError::DeploymentAdmission);
        }
        Ok(())
    }

    /// Cleanup remains permitted after authority revocation. A nonzero status
    /// is an unknown outcome; do not retry or reuse that handle.
    pub fn close(&mut self) -> Result<(), FastRpcError> {
        if self.state == SessionState::CloseFailed {
            return Err(FastRpcError::CloseOutcomeUnknown);
        }
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        // Cleanup is still necessary after poisoning, but future opens fail.
        let _lifecycle = RUNTIME_LIFECYCLE
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let status = unsafe { (self.close)(handle) };
        if status != 0 {
            self.state = SessionState::CloseFailed;
            self.retain_runtime();
            return Err(FastRpcError::Close(status));
        }
        self.state = SessionState::Closed;
        Ok(())
    }

    fn retain_runtime(&mut self) {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        if let Some(library) = self.library.take() {
            core::mem::forget(library);
        }
    }
}

impl Drop for FastRpcSession<'_> {
    fn drop(&mut self) {
        let _ = self.close();
        #[cfg(any(target_os = "linux", target_os = "android"))]
        if let Some(library) = self.library.take() {
            let _lifecycle = RUNTIME_LIFECYCLE
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            drop(library);
        }
    }
}

fn lease_current(lease: &dyn DeploymentLease) -> bool {
    catch_unwind(AssertUnwindSafe(|| lease.current())).unwrap_or(false)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
unsafe fn symbols(
    library: &libloading::Library,
) -> Result<(abi::Open, abi::Invoke, abi::Close), FastRpcError> {
    let open = *unsafe { library.get::<abi::Open>(b"remote_handle64_open\0") }
        .map_err(|_| FastRpcError::MissingSymbol)?;
    let invoke = *unsafe { library.get::<abi::Invoke>(b"remote_handle64_invoke\0") }
        .map_err(|_| FastRpcError::MissingSymbol)?;
    let close = *unsafe { library.get::<abi::Close>(b"remote_handle64_close\0") }
        .map_err(|_| FastRpcError::MissingSymbol)?;
    Ok((open, invoke, close))
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;
