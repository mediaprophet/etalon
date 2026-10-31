use crate::{
    abi::Dev,
    config, identity,
    runtime::{RtlSdrRuntime, LIFECYCLE},
    AppliedConfig, Config, Error, GateContext, RtlSdrGate, Stage, UsbIdentity,
};
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    ptr::NonNull,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceState {
    Unconfigured,
    Ready,
    Faulted,
    Closed,
    CloseFailed,
}

/// Driver declarations only. Sequence orders successful host reads, not samples.
#[derive(Clone, Debug)]
pub struct ReadReport {
    target_id: u64,
    generation: u32,
    device_index: u32,
    pair_count: usize,
    sequence: u64,
    requested: Config,
    applied: AppliedConfig,
    identity: UsbIdentity,
}

impl ReadReport {
    pub fn target_id(&self) -> u64 {
        self.target_id
    }
    pub fn generation(&self) -> u32 {
        self.generation
    }
    pub fn device_index(&self) -> u32 {
        self.device_index
    }
    pub fn pair_count(&self) -> usize {
        self.pair_count
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn requested_config(&self) -> Config {
        self.requested
    }
    pub fn applied_config(&self) -> AppliedConfig {
        self.applied
    }
    pub fn usb_identity(&self) -> &UsbIdentity {
        &self.identity
    }
}

pub struct RtlSdrDevice<'runtime, 'lease> {
    runtime: &'runtime RtlSdrRuntime<'lease>,
    handle: Option<NonNull<Dev>>,
    state: DeviceState,
    identity: Option<UsbIdentity>,
    configuration: Option<(Config, AppliedConfig)>,
    next_sequence: u64,
    last_error: Option<Error>,
    cleanup_error: Option<i32>,
}

impl<'runtime, 'lease> RtlSdrDevice<'runtime, 'lease> {
    pub(crate) fn open(
        runtime: &'runtime RtlSdrRuntime<'lease>,
        gate: &impl RtlSdrGate,
    ) -> Result<Self, Error> {
        if !runtime.current() {
            return Err(Error::Admission);
        }
        let context = GateContext {
            stage: Stage::Open,
            target_id: runtime.target_id,
            generation: runtime.generation,
            device_index: runtime.device_index,
            requested: None,
            applied: None,
            identity: None,
            pair_count: None,
        };
        if !allowed(gate, &context) {
            return Err(Error::NotAuthorized);
        }
        if !runtime.current() {
            return Err(Error::Admission);
        }
        runtime.claim()?;
        let mut pointer = core::ptr::null_mut();
        let lifecycle = LIFECYCLE.lock().map_err(|_| Error::Synchronization)?;
        runtime.native_opened();
        let code = unsafe { (runtime.api.open)(&mut pointer, runtime.device_index) };
        drop(lifecycle);
        if code != 0 {
            runtime.retain();
            return Err(Error::Native {
                operation: "open",
                code,
            });
        }
        let Some(handle) = NonNull::new(pointer) else {
            runtime.retain();
            return Err(Error::InvalidNativeHandle);
        };
        let mut device = Self {
            runtime,
            handle: Some(handle),
            state: DeviceState::Unconfigured,
            identity: None,
            configuration: None,
            next_sequence: 0,
            last_error: None,
            cleanup_error: None,
        };
        let result = (|| {
            device.identity = Some(unsafe { identity::query(runtime.api, handle.as_ptr()) }?);
            device.authorize(gate, Stage::PostOpen, None, None, None)
        })();
        if let Err(error) = result {
            return Err(device.fail(error));
        }
        Ok(device)
    }

    pub fn target_id(&self) -> u64 {
        self.runtime.target_id
    }
    pub fn generation(&self) -> u32 {
        self.runtime.generation
    }
    pub fn device_index(&self) -> u32 {
        self.runtime.device_index
    }
    pub fn max_pairs(&self) -> usize {
        self.runtime.max_pairs
    }
    pub const fn allocation_class(&self) -> etalon_sense_core::processing_runtime::AllocationClass {
        etalon_sense_core::processing_runtime::AllocationClass::ColdUnbounded
    }
    pub fn state(&self) -> DeviceState {
        self.state
    }
    pub fn last_error(&self) -> Option<Error> {
        self.last_error
    }
    pub fn cleanup_error(&self) -> Option<i32> {
        self.cleanup_error
    }

    pub fn configure(
        &mut self,
        config: Config,
        gate: &impl RtlSdrGate,
    ) -> Result<AppliedConfig, Error> {
        let result = (|| {
            if !matches!(self.state, DeviceState::Unconfigured | DeviceState::Ready) {
                return Err(Error::InvalidState);
            }
            self.authorize(gate, Stage::Configure, Some(config), None, None)?;
            let handle = self.pointer()?;
            let actual = unsafe { config::apply(self.runtime.api, handle, config) }?;
            self.recheck_identity(handle)?;
            self.authorize(gate, Stage::PostConfigure, Some(config), Some(actual), None)?;
            self.configuration = Some((config, actual));
            self.state = DeviceState::Ready;
            Ok(actual)
        })();
        match result {
            Ok(value) => Ok(value),
            Err(error) => Err(self.fail(error)),
        }
    }

    /// Exact raw unsigned-u8 I/Q pairs. No native timeout exists. This is not
    /// the bounded-duration `SdrReader` contract; use host isolation for deadlines.
    pub fn read_into(
        &mut self,
        gate: &impl RtlSdrGate,
        output: &mut [u8],
    ) -> Result<ReadReport, Error> {
        output.fill(0);
        let result = self.read_inner(gate, output);
        match result {
            Ok(report) => Ok(report),
            Err(error) => {
                output.fill(0);
                Err(self.fail(error))
            }
        }
    }

    fn read_inner(
        &mut self,
        gate: &impl RtlSdrGate,
        output: &mut [u8],
    ) -> Result<ReadReport, Error> {
        if self.state != DeviceState::Ready {
            return Err(Error::InvalidState);
        }
        if output.is_empty()
            || !output.len().is_multiple_of(2)
            || output.len() / 2 > self.runtime.max_pairs
            || output.len() > i32::MAX as usize
        {
            return Err(Error::InvalidBuffer);
        }
        let (requested, applied) = self.configuration.ok_or(Error::InvalidState)?;
        let sequence = self.next_sequence;
        self.next_sequence = sequence.checked_add(1).ok_or(Error::SequenceExhausted)?;
        self.authorize(
            gate,
            Stage::Read,
            Some(requested),
            Some(applied),
            Some(output.len() / 2),
        )?;
        let handle = self.pointer()?;
        self.recheck_identity(handle)?;
        if unsafe { config::reported(self.runtime.api, handle, requested) }? != applied {
            return Err(Error::ConfigurationChanged);
        }
        self.authorize(
            gate,
            Stage::ReadReady,
            Some(requested),
            Some(applied),
            Some(output.len() / 2),
        )?;
        let mut count = 0i32;
        let code = unsafe {
            (self.runtime.api.read)(
                handle,
                output.as_mut_ptr().cast(),
                output.len() as i32,
                &mut count,
            )
        };
        config::checked("read_sync", code)?;
        if count < 0 || count as usize != output.len() {
            return Err(Error::PartialRead {
                actual_bytes: count,
                expected_bytes: output.len(),
            });
        }
        self.recheck_identity(handle)?;
        if unsafe { config::reported(self.runtime.api, handle, requested) }? != applied {
            return Err(Error::ConfigurationChanged);
        }
        self.authorize(
            gate,
            Stage::PostRead,
            Some(requested),
            Some(applied),
            Some(output.len() / 2),
        )?;
        Ok(ReadReport {
            target_id: self.target_id(),
            generation: self.generation(),
            device_index: self.device_index(),
            pair_count: output.len() / 2,
            sequence,
            requested,
            applied,
            identity: self.identity.clone().ok_or(Error::InvalidIdentity)?,
        })
    }

    fn authorize(
        &self,
        gate: &impl RtlSdrGate,
        stage: Stage,
        requested: Option<Config>,
        applied: Option<AppliedConfig>,
        pair_count: Option<usize>,
    ) -> Result<(), Error> {
        if !self.runtime.current() {
            return Err(Error::Admission);
        }
        if !allowed(
            gate,
            &GateContext {
                stage,
                target_id: self.target_id(),
                generation: self.generation(),
                device_index: self.device_index(),
                requested,
                applied,
                identity: self.identity.as_ref(),
                pair_count,
            },
        ) {
            return Err(Error::NotAuthorized);
        }
        if !self.runtime.current() {
            return Err(Error::Admission);
        }
        Ok(())
    }

    fn pointer(&self) -> Result<*mut Dev, Error> {
        self.handle.map(NonNull::as_ptr).ok_or(Error::InvalidState)
    }

    fn recheck_identity(&self, handle: *mut Dev) -> Result<(), Error> {
        if Some(unsafe { identity::query(self.runtime.api, handle) }?) != self.identity {
            Err(Error::InvalidIdentity)
        } else {
            Ok(())
        }
    }

    fn fail(&mut self, error: Error) -> Error {
        self.last_error = Some(error);
        if self.handle.is_some() {
            self.state = DeviceState::Faulted;
            if let Err(Error::Native { code, .. }) = self.close() {
                return combine_cleanup(error, code);
            }
        }
        error
    }

    pub fn close(&mut self) -> Result<(), Error> {
        if self.state == DeviceState::CloseFailed {
            return Err(Error::CloseOutcomeUnknown);
        }
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        let _lifecycle = LIFECYCLE.lock().unwrap_or_else(|error| error.into_inner());
        let code = unsafe { (self.runtime.api.close)(handle.as_ptr()) };
        if code != 0 {
            self.cleanup_error = Some(code);
            self.runtime.retain();
            self.state = DeviceState::CloseFailed;
            return Err(Error::Native {
                operation: "close",
                code,
            });
        }
        self.state = DeviceState::Closed;
        self.runtime.native_closed();
        Ok(())
    }
}

impl Drop for RtlSdrDevice<'_, '_> {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn allowed(gate: &impl RtlSdrGate, context: &GateContext<'_>) -> bool {
    catch_unwind(AssertUnwindSafe(|| gate.allow(context))).unwrap_or(false)
}

fn combine_cleanup(error: Error, cleanup_code: i32) -> Error {
    let (primary_operation, primary_code) = match error {
        Error::Native { operation, code } => (operation, Some(code)),
        Error::NotAuthorized => ("authorization", None),
        Error::Admission => ("deployment_admission", None),
        Error::InvalidIdentity => ("device_identity", None),
        _ => ("validation", None),
    };
    Error::CleanupFailed {
        primary_operation,
        primary_code,
        cleanup_code,
    }
}
