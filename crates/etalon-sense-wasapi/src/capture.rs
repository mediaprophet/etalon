use core::{marker::PhantomData, ptr};
#[path = "capture_output.rs"]
mod capture_output;
use capture_output::CaptureOutput;
#[path = "capture_read.rs"]
mod capture_read;
use std::rc::Rc;

use crate::remainder::{PacketMetadata, Remainder};
use crate::{config::WasapiConfig, format::WasapiFormat, helpers::*};
use etalon_sense_core::{
    adapter::{
        AdapterError, AdapterId, AuthorizedExtendedCapture, AuthorizedExtendedCaptureBuffer,
        ExtendedAdapterDescriptor, ExtendedCaptureSession, ExtendedCapturedSamples,
        ExtendedDiscoveredSource, ExtendedSensorAdapter,
    },
    authority::TrustedCaptureClock,
    extended_capability::ExtendedCapability,
    time::ClockStamp,
    ContractError,
};
use windows::{
    core::{IUnknown, PWSTR},
    Win32::{
        Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0, WAIT_TIMEOUT},
        Media::Audio::{
            eCapture, eConsole, IAudioCaptureClient, IAudioClient, IMMDevice, IMMDeviceEnumerator,
            MMDeviceEnumerator, AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY, AUDCLNT_BUFFERFLAGS_SILENT,
            AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR, AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_EVENTCALLBACK, WAVEFORMATEX,
        },
        System::{
            Com::{
                CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL,
                COINIT_MULTITHREADED,
            },
            Threading::{CreateEventW, WaitForSingleObject},
        },
    },
};

/// Timing and discontinuity metadata for the last successful frame. Native
/// QPC values are preserved raw; no cross-clock or UTC mapping is inferred.
/// A batch starting in a retained packet tail has no inferred QPC position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureMetadata {
    pub format: WasapiFormat,
    pub host_clock_at_dispatch: ClockStamp,
    pub first_device_position: Option<u64>,
    pub first_qpc_position: Option<u64>,
    pub frames: u32,
    pub silent_frames: u32,
    pub data_discontinuity: bool,
    pub timestamp_error: bool,
}

struct ComApartment(PhantomData<Rc<()>>);

impl ComApartment {
    fn initialize() -> Result<Self, AdapterError> {
        let result = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if result.is_err() {
            return Err(AdapterError::Unavailable);
        }
        Ok(Self(PhantomData))
    }
}

impl Drop for ComApartment {
    fn drop(&mut self) {
        unsafe { CoUninitialize() }
    }
}

struct TaskMemWide(PWSTR);

impl Drop for TaskMemWide {
    fn drop(&mut self) {
        unsafe { CoTaskMemFree(Some(self.0.as_ptr().cast())) }
    }
}

struct EventHandle(HANDLE);

impl EventHandle {
    fn new() -> Result<Self, AdapterError> {
        let handle = unsafe { CreateEventW(None, false, false, None) }
            .map_err(|_| AdapterError::Unavailable)?;
        Ok(Self(handle))
    }
}

impl Drop for EventHandle {
    fn drop(&mut self) {
        let _ = unsafe { CloseHandle(self.0) };
    }
}

struct CaptureRuntime {
    client: IAudioClient,
    capture: IAudioCaptureClient,
    event: EventHandle,
    format: WasapiFormat,
    stopped: bool,
}

impl CaptureRuntime {
    fn stop(&mut self) -> Result<(), AdapterError> {
        if self.stopped {
            return Ok(());
        }
        unsafe { self.client.Stop() }.map_err(|_| AdapterError::BackendFailure)?;
        self.stopped = true;
        Ok(())
    }
}

impl Drop for CaptureRuntime {
    fn drop(&mut self) {
        if !self.stopped {
            let _ = unsafe { self.client.Stop() };
        }
    }
}

/// Default-input WASAPI adapter. COM and audio objects are thread-affine; the
/// `Rc` marker prevents moving this value to another thread after discovery.
pub struct WasapiCaptureAdapter<'scratch, C> {
    descriptor: ExtendedAdapterDescriptor,
    config: WasapiConfig,
    clock: C,
    source: Option<ExtendedDiscoveredSource>,
    endpoint: Option<IMMDevice>,
    runtime: Option<CaptureRuntime>,
    remainder: Remainder<'scratch>,
    // Keep this last so COM interfaces above are released before CoUninitialize.
    apartment: Option<ComApartment>,
    active: Option<ExtendedCaptureSession>,
    next_session_id: u64,
    generation: u32,
    faulted: bool,
    last_metadata: Option<CaptureMetadata>,
    _thread_affinity: PhantomData<Rc<()>>,
}

impl<C: TrustedCaptureClock> WasapiCaptureAdapter<'static, C> {
    /// Compatibility constructor without packet-tail storage. Oversized OS
    /// packets still fail closed; use `with_remainder` for arbitrary batches.
    pub fn new(config: WasapiConfig, clock: C) -> Result<Self, AdapterError> {
        Self::with_remainder(config, clock, &mut [], 0)
    }
}

impl<'scratch, C: TrustedCaptureClock> WasapiCaptureAdapter<'scratch, C> {
    /// Construct without initializing COM or activating a device. The host
    /// decides when to call discovery and supplies all policy/clock inputs.
    /// Scratch is caller-owned and wiped on fault, stop and drop. Startup
    /// requires it to hold the negotiated endpoint's full packet capacity.
    /// This budget covers scratch bytes only, not COM or driver allocations.
    pub fn with_remainder(
        config: WasapiConfig,
        clock: C,
        scratch: &'scratch mut [u8],
        scratch_budget_bytes: usize,
    ) -> Result<Self, AdapterError> {
        if scratch.len() > scratch_budget_bytes {
            return Err(AdapterError::Unsupported);
        }
        let descriptor = ExtendedAdapterDescriptor::new(config.adapter_id, config.modality_id)?;
        Ok(Self {
            descriptor,
            config,
            clock,
            source: None,
            endpoint: None,
            runtime: None,
            remainder: Remainder::new(scratch),
            apartment: None,
            active: None,
            next_session_id: 1,
            generation: 1,
            faulted: false,
            last_metadata: None,
            _thread_affinity: PhantomData,
        })
    }

    pub const fn last_metadata(&self) -> Option<CaptureMetadata> {
        self.last_metadata
    }

    /// Negotiated device format after a successful start. Values describe the
    /// byte representation, not microphone calibration or physical units.
    pub fn negotiated_format(&self) -> Option<WasapiFormat> {
        self.runtime.as_ref().map(|runtime| runtime.format)
    }

    /// Retry cleanup when startup failed after OS capture began and no session
    /// was returned. This never starts or resumes collection.
    pub fn retry_failed_stop(&mut self) -> Result<(), AdapterError> {
        if !self.faulted {
            return Err(AdapterError::InvalidSession);
        }
        let session = self.active.ok_or(AdapterError::InvalidSession)?;
        self.stop_backend(session)
    }
}

impl<C: TrustedCaptureClock> ExtendedSensorAdapter for WasapiCaptureAdapter<'_, C> {
    fn descriptor(&self) -> ExtendedAdapterDescriptor {
        self.descriptor
    }

    fn trusted_now(&mut self) -> Result<ClockStamp, AdapterError> {
        self.clock.now()
    }

    fn discover_backend(
        &mut self,
        out: &mut [ExtendedDiscoveredSource],
    ) -> Result<usize, AdapterError> {
        if out.is_empty() {
            return Err(AdapterError::DiscoveryBufferTooSmall);
        }
        if self.active.is_some() {
            return Err(AdapterError::InvalidSession);
        }
        if let Some(source) = self.source {
            out[0] = source;
            return Ok(1);
        }
        let apartment = ComApartment::initialize()?;
        let enumerator: IMMDeviceEnumerator = unsafe {
            CoCreateInstance::<Option<&IUnknown>, IMMDeviceEnumerator>(
                &MMDeviceEnumerator,
                None,
                CLSCTX_ALL,
            )
        }
        .map_err(|_| AdapterError::Unavailable)?;
        let endpoint = unsafe { enumerator.GetDefaultAudioEndpoint(eCapture, eConsole) }
            .map_err(|_| AdapterError::Unavailable)?;
        let endpoint_id =
            TaskMemWide(unsafe { endpoint.GetId() }.map_err(|_| AdapterError::BackendFailure)?);
        let wide_id =
            unsafe { endpoint_id.0.to_string() }.map_err(|_| AdapterError::InvalidIdentifier)?;
        let source_id = stable_source_id(wide_id.as_bytes());
        let source = ExtendedDiscoveredSource::new(source_id, self.config.capability)?;
        self.endpoint = Some(endpoint);
        self.apartment = Some(apartment);
        self.source = Some(source);
        out[0] = source;
        Ok(1)
    }

    fn start_backend(
        &mut self,
        request: AuthorizedExtendedCapture,
    ) -> Result<ExtendedCaptureSession, AdapterError> {
        let source = self.source.ok_or(AdapterError::Unavailable)?;
        if request.source() != source {
            return Err(AdapterError::AuthorizationDenied);
        }
        if self.active.is_some() || self.runtime.is_some() {
            return Err(AdapterError::InvalidSession);
        }
        if self.faulted {
            return Err(AdapterError::Unavailable);
        }
        self.last_metadata = None;
        request.validate_at(self.trusted_now()?)?;
        let id = self.next_session_id;
        let next_session_id = id.checked_add(1).ok_or(AdapterError::BackendFailure)?;
        let session =
            ExtendedCaptureSession::new(self.descriptor.id(), source, id, self.generation)?;
        let endpoint = self.endpoint.as_ref().ok_or(AdapterError::Unavailable)?;
        let client: IAudioClient = unsafe { endpoint.Activate(CLSCTX_ALL, None) }
            .map_err(|_| AdapterError::Unavailable)?;
        let format_ptr =
            unsafe { client.GetMixFormat() }.map_err(|_| AdapterError::BackendFailure)?;
        let owned_format = TaskMemFormat(format_ptr);
        let format = unsafe { WasapiFormat::parse(owned_format.0) }?;
        validate_capability(self.config.capability, format)?;
        let event = EventHandle::new()?;
        unsafe {
            client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                10_000_000,
                0,
                owned_format.0,
                None,
            )
        }
        .map_err(|_| AdapterError::Unavailable)?;
        if self.remainder.capacity() != 0 {
            let capacity_frames =
                unsafe { client.GetBufferSize() }.map_err(|_| AdapterError::BackendFailure)?;
            let required = (capacity_frames as usize)
                .checked_mul(usize::from(format.block_align()))
                .ok_or(AdapterError::BackendFailure)?;
            if required == 0 || required > self.remainder.capacity() {
                return Err(AdapterError::Unsupported);
            }
        }
        unsafe { client.SetEventHandle(event.0) }.map_err(|_| AdapterError::BackendFailure)?;
        let capture: IAudioCaptureClient =
            unsafe { client.GetService() }.map_err(|_| AdapterError::BackendFailure)?;
        request.validate_at(self.trusted_now()?)?;
        unsafe { client.Start() }.map_err(|_| AdapterError::Unavailable)?;
        self.next_session_id = next_session_id;
        self.runtime = Some(CaptureRuntime {
            client,
            capture,
            event,
            format,
            stopped: false,
        });
        self.active = Some(session);
        let final_check = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.trusted_now().and_then(|now| request.validate_at(now))
        }));
        match final_check {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                self.fail_closed_capture();
                return Err(error);
            }
            Err(panic) => {
                self.fail_closed_capture();
                std::panic::resume_unwind(panic);
            }
        }
        Ok(session)
    }

    fn capture_backend<'a>(
        &mut self,
        output: AuthorizedExtendedCaptureBuffer<'a>,
    ) -> Result<ExtendedCapturedSamples<'a>, AdapterError> {
        self.capture_with_metadata(output)
            .map(|(samples, _)| samples)
    }

    fn capture<'a>(
        &mut self,
        output: AuthorizedExtendedCaptureBuffer<'a>,
    ) -> Result<ExtendedCapturedSamples<'a>, AdapterError> {
        // All preflight checks run inside the output and stream guards.
        self.capture_backend(output)
    }

    fn stop_backend(&mut self, session: ExtendedCaptureSession) -> Result<(), AdapterError> {
        if self.active != Some(session) {
            return Err(AdapterError::InvalidSession);
        }
        self.remainder.clear();
        self.last_metadata = None;
        let stop_result = match self.runtime.as_mut() {
            Some(runtime) => runtime.stop(),
            None => Ok(()),
        };
        let confirmed = stop_result.is_ok();
        let result = finish_stop_state(
            stop_result,
            &mut self.active,
            &mut self.generation,
            &mut self.faulted,
        );
        if confirmed {
            self.runtime = None;
        }
        result
    }
}

impl<C> WasapiCaptureAdapter<'_, C> {
    /// Stop collecting after an authorization/deadline clock failure while
    /// retaining the active session/runtime. A failed OS Stop stays retryable
    /// through `stop`; no capture is reported as successful after this point.
    pub(crate) fn fail_closed_capture(&mut self) {
        self.remainder.clear();
        self.last_metadata = None;
        self.faulted = true;
        if let Some(runtime) = self.runtime.as_mut() {
            let _ = runtime.stop();
        }
    }

    fn check_capture_request(
        &mut self,
        output: &AuthorizedExtendedCaptureBuffer<'_>,
        capture_started: ClockStamp,
    ) -> Result<ClockStamp, AdapterError>
    where
        C: TrustedCaptureClock,
    {
        let now = self.clock.now()?;
        output.validate_at(now)?;
        capture_wait_timeout_ms(
            capture_started,
            now,
            self.config.max_capture_duration_ms,
            self.config.wait_timeout_ms,
        )?;
        Ok(now)
    }
}

impl<C: TrustedCaptureClock> WasapiCaptureAdapter<'_, C> {
    pub(crate) fn capture_with_metadata<'a>(
        &mut self,
        output: AuthorizedExtendedCaptureBuffer<'a>,
    ) -> Result<(ExtendedCapturedSamples<'a>, CaptureMetadata), AdapterError> {
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.capture_impl(output)));
        match result {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(error)) => {
                self.fail_closed_capture();
                Err(error)
            }
            Err(panic) => {
                self.fail_closed_capture();
                std::panic::resume_unwind(panic)
            }
        }
    }
}

struct TaskMemFormat(*mut WAVEFORMATEX);

impl Drop for TaskMemFormat {
    fn drop(&mut self) {
        unsafe { CoTaskMemFree(Some(self.0.cast())) }
    }
}

#[cfg(test)]
#[path = "capture_tests.rs"]
mod tests;
