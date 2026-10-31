//! Nonblocking ownership of the unqualified WinRT camera lifecycle.
use ::windows::Media::Capture::Frames::{MediaFrameReader, MediaFrameSource};
use ::windows::Media::Capture::{MediaCapture, MediaCaptureInitializationSettings};
use etalon_sense_camera::FrameDeclaration;

use crate::discovery::{self, CameraSourceAuthority, CameraSourceCandidate};
use crate::winrt::{
    begin_create_reader, begin_initialize, begin_reader_start, AsyncPoll, CameraOperation,
    CameraPolicy, CleanupProgress, NativeError, PendingAction, PendingReader, PendingReaderStart,
    RunningFrameReader, StartFailure, StartFailureReason,
};
#[cfg(feature = "unqualified-frame-copy")]
use crate::winrt::{CopyAllocationClass, WindowsCapturedFrame};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Initializing,
    AwaitSource,
    Creating,
    Starting,
    Running,
    Cleanup,
    FailedStart,
    Stopped,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineProgress {
    Pending,
    AwaitSource,
    Running,
    Stopped,
}

/// Owns pending operations and capture handles until stop/close is confirmed.
/// Host must keep policy live, poll regularly, and retain this value until
/// `Stopped`; abandoning it does not prove that Windows stopped capturing.
#[must_use = "request stop and poll until Stopped before dropping"]
pub struct CameraPipeline {
    declaration: FrameDeclaration,
    stage: Stage,
    stop_requested: bool,
    capture: Option<MediaCapture>,
    reader: Option<MediaFrameReader>,
    init: Option<PendingAction>,
    create: Option<PendingReader>,
    start: Option<PendingReaderStart>,
    running: Option<RunningFrameReader>,
    failure: Option<StartFailure>,
}

impl CameraPipeline {
    /// Caller retains its original capture if initialization cannot begin and
    /// must close it in that case. Once begun, poll this owner through stop.
    pub fn begin(
        capture: &MediaCapture,
        settings: &MediaCaptureInitializationSettings,
        declaration: FrameDeclaration,
        policy: &mut impl CameraPolicy,
    ) -> Result<Self, NativeError> {
        let init = begin_initialize(
            capture,
            settings,
            declaration.source_id(),
            declaration.modality(),
            policy,
        )?;
        Ok(Self {
            declaration,
            stage: Stage::Initializing,
            stop_requested: false,
            capture: Some(capture.clone()),
            reader: None,
            init: Some(init),
            create: None,
            start: None,
            running: None,
            failure: None,
        })
    }

    /// Source selection follows completed initialization. The host must bind
    /// this source to the declaration and validate its modality independently.
    fn provide_source(
        &mut self,
        source: &MediaFrameSource,
        policy: &mut impl CameraPolicy,
    ) -> Result<(), NativeError> {
        if self.stage != Stage::AwaitSource || self.stop_requested {
            return Err(NativeError::WrongState);
        }
        let capture = self.capture.as_ref().ok_or(NativeError::WrongState)?;
        match begin_create_reader(capture, source, self.declaration, policy) {
            Ok(create) => {
                self.create = Some(create);
                self.stage = Stage::Creating;
                Ok(())
            }
            Err(error) => {
                self.stop_requested = true;
                self.stage = Stage::Cleanup;
                Err(error)
            }
        }
    }

    /// Enumerate the initialized capture's native sources into caller slots.
    /// Native identifiers and categories are declarations, not identity or
    /// proof of visible, infrared or radiometric capability.
    pub fn list_sources(
        &mut self,
        policy: &mut impl CameraPolicy,
        authority: &mut impl CameraSourceAuthority,
        output: &mut [Option<CameraSourceCandidate>],
    ) -> Result<usize, NativeError> {
        output.fill_with(|| None);
        if self.stage != Stage::AwaitSource || self.stop_requested {
            return Err(NativeError::WrongState);
        }
        let capture = self.capture.as_ref().ok_or(NativeError::WrongState)?;
        discovery::list_sources(capture, self.declaration, policy, authority, output)
    }

    /// Re-resolve the selected identifier against this active capture and
    /// require the host to approve the exact source again before reader setup.
    /// The host must bind the declaration to this candidate and reflect
    /// subsequent source-specific revocations in every `CameraPolicy` check.
    pub fn provide_candidate(
        &mut self,
        candidate: &CameraSourceCandidate,
        policy: &mut impl CameraPolicy,
        authority: &mut impl CameraSourceAuthority,
    ) -> Result<(), NativeError> {
        if self.stage != Stage::AwaitSource || self.stop_requested {
            return Err(NativeError::WrongState);
        }
        let capture = self.capture.as_ref().ok_or(NativeError::WrongState)?;
        let source =
            discovery::resolve_candidate(capture, self.declaration, candidate, policy, authority)?;
        self.provide_source(&source, policy)
    }

    /// Requests cancellation/stop; `poll` must be driven to `Stopped`.
    pub fn request_stop(&mut self) {
        self.stop_requested = true;
    }

    pub fn poll(
        &mut self,
        policy: &mut impl CameraPolicy,
    ) -> Result<PipelineProgress, NativeError> {
        let operation = match self.stage {
            Stage::Initializing => Some(CameraOperation::Initialize),
            Stage::AwaitSource | Stage::Creating => Some(CameraOperation::CreateReader),
            Stage::Starting => Some(CameraOperation::Start),
            Stage::Running => Some(CameraOperation::AcquireFrame),
            _ => None,
        };
        if !self.stop_requested {
            if let Some(operation) = operation {
                let permitted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    policy.permit(
                        self.declaration.source_id(),
                        self.declaration.modality(),
                        operation,
                    )
                }))
                .unwrap_or(false);
                if !permitted {
                    self.stop_requested = true;
                }
            }
        }
        match self.stage {
            Stage::Initializing => {
                let action = self.init.as_mut().ok_or(NativeError::WrongState)?;
                if self.stop_requested {
                    let _ = action.cancel();
                }
                match action.poll() {
                    Ok(AsyncPoll::Pending) => Ok(PipelineProgress::Pending),
                    Ok(AsyncPoll::Ready(())) => {
                        self.init = None;
                        self.stage = if self.stop_requested {
                            Stage::Cleanup
                        } else {
                            Stage::AwaitSource
                        };
                        Ok(PipelineProgress::Pending)
                    }
                    Err(error) => {
                        if action.is_terminal() {
                            self.init = None;
                            self.stage = Stage::Cleanup;
                        }
                        Err(error)
                    }
                }
            }
            Stage::AwaitSource => {
                if self.stop_requested {
                    self.stage = Stage::Cleanup;
                    Ok(PipelineProgress::Pending)
                } else {
                    Ok(PipelineProgress::AwaitSource)
                }
            }
            Stage::Creating => {
                let action = self.create.as_mut().ok_or(NativeError::WrongState)?;
                if self.stop_requested {
                    let _ = action.cancel();
                }
                match action.poll() {
                    Ok(AsyncPoll::Pending) => Ok(PipelineProgress::Pending),
                    Ok(AsyncPoll::Ready(reader)) => {
                        self.create = None;
                        self.reader = Some(reader);
                        if self.stop_requested {
                            self.stage = Stage::Cleanup;
                            return Ok(PipelineProgress::Pending);
                        }
                        let capture = self.capture.as_ref().ok_or(NativeError::WrongState)?;
                        let reader = self.reader.as_ref().ok_or(NativeError::WrongState)?;
                        match begin_reader_start(capture, reader, self.declaration, policy) {
                            Ok(start) => {
                                self.start = Some(start);
                                self.stage = Stage::Starting;
                                Ok(PipelineProgress::Pending)
                            }
                            Err(NativeError::PolicyDenied) => {
                                self.stage = Stage::Cleanup;
                                Err(NativeError::PolicyDenied)
                            }
                            Err(error) => {
                                self.start_failure()?;
                                Err(error)
                            }
                        }
                    }
                    Err(error) => {
                        if action.is_terminal() {
                            self.create = None;
                            self.stage = Stage::Cleanup;
                        }
                        Err(error)
                    }
                }
            }
            Stage::Starting => {
                let action = self.start.as_mut().ok_or(NativeError::WrongState)?;
                if self.stop_requested {
                    let _ = action.cancel();
                }
                match action.poll() {
                    Ok(AsyncPoll::Pending) => Ok(PipelineProgress::Pending),
                    Ok(AsyncPoll::Ready(status)) => {
                        self.start = None;
                        let capture = self.capture.take().ok_or(NativeError::WrongState)?;
                        let reader = self.reader.take().ok_or(NativeError::WrongState)?;
                        match RunningFrameReader::from_started(
                            capture,
                            reader,
                            self.declaration,
                            status,
                            policy,
                        ) {
                            Ok(mut running) => {
                                if self.stop_requested {
                                    let _ = running.request_stop();
                                }
                                self.running = Some(running);
                                self.stage = Stage::Running;
                                Ok(PipelineProgress::Pending)
                            }
                            Err(failure) => {
                                self.failure = Some(failure);
                                self.stage = Stage::FailedStart;
                                Ok(PipelineProgress::Pending)
                            }
                        }
                    }
                    Err(error) => {
                        if action.is_terminal() {
                            self.start = None;
                            self.start_failure()?;
                        }
                        Err(error)
                    }
                }
            }
            Stage::Running => {
                let running = self.running.as_mut().ok_or(NativeError::WrongState)?;
                if self.stop_requested || running.is_stopping() {
                    self.stop_requested = true;
                    running.request_stop()?;
                    match running.poll_stop()? {
                        AsyncPoll::Pending => Ok(PipelineProgress::Pending),
                        AsyncPoll::Ready(()) => {
                            self.running = None;
                            self.stage = Stage::Stopped;
                            Ok(PipelineProgress::Stopped)
                        }
                    }
                } else {
                    Ok(PipelineProgress::Running)
                }
            }
            Stage::Cleanup => {
                if let Some(reader) = self.reader.as_ref() {
                    reader.Close()?;
                    self.reader = None;
                }
                if let Some(capture) = self.capture.as_ref() {
                    capture.Close()?;
                    self.capture = None;
                }
                self.stage = Stage::Stopped;
                Ok(PipelineProgress::Stopped)
            }
            Stage::FailedStart => match self
                .failure
                .as_mut()
                .ok_or(NativeError::WrongState)?
                .retry_cleanup()?
            {
                CleanupProgress::Pending => Ok(PipelineProgress::Pending),
                CleanupProgress::Complete => {
                    self.failure = None;
                    self.stage = Stage::Stopped;
                    Ok(PipelineProgress::Stopped)
                }
            },
            Stage::Stopped => Ok(PipelineProgress::Stopped),
        }
    }

    fn start_failure(&mut self) -> Result<(), NativeError> {
        let capture = self.capture.take().ok_or(NativeError::WrongState)?;
        let reader = self.reader.take().ok_or(NativeError::WrongState)?;
        self.failure = Some(StartFailure::new(
            StartFailureReason::StartOperationFailed,
            capture,
            reader,
            true,
        ));
        self.stage = Stage::FailedStart;
        Ok(())
    }

    #[cfg(feature = "unqualified-frame-copy")]
    pub fn capture_into(
        &mut self,
        policy: &mut impl CameraPolicy,
        allocation: CopyAllocationClass,
        output: &mut [u8],
    ) -> Result<WindowsCapturedFrame, NativeError> {
        output.fill(0);
        if self.stop_requested || self.stage != Stage::Running {
            output.fill(0);
            return Err(NativeError::WrongState);
        }
        self.running
            .as_mut()
            .ok_or(NativeError::WrongState)?
            .capture_into(policy, allocation, output)
    }
}
