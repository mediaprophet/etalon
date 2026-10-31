use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartFailureReason { StartStatus(i32), InvalidDeclaration, StartOperationFailed }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CleanupProgress { Pending, Complete }

/// Failed start that retains COM resources until explicit cleanup succeeds.
/// Call `retry_cleanup` until it returns `Complete` before dropping this value.
#[must_use = "retry cleanup until the retained WinRT handles are closed"]
pub struct StartFailure {
    reason: StartFailureReason,
    capture: Option<MediaCapture>,
    reader: Option<MediaFrameReader>,
    stop_required: bool,
    stop_operation: Option<PendingAction>,
    reader_closed: bool,
    capture_closed: bool,
}

impl StartFailure {
    pub(crate) fn new(
        reason: StartFailureReason,
        capture: MediaCapture,
        reader: MediaFrameReader,
        stop_required: bool,
    ) -> Self {
        Self {
            reason,
            capture: Some(capture),
            reader: Some(reader),
            stop_required,
            stop_operation: None,
            reader_closed: false,
            capture_closed: false,
        }
    }

    pub const fn reason(&self) -> StartFailureReason { self.reason }

    pub fn retry_cleanup(&mut self) -> Result<CleanupProgress, NativeError> {
        if self.stop_required {
            if self.stop_operation.is_none() {
                let capture = self.capture.as_ref().ok_or(NativeError::WrongState)?;
                let reader = self.reader.as_ref().ok_or(NativeError::WrongState)?;
                self.stop_operation = Some(begin_reader_stop(capture, reader)?);
            }
            match self.stop_operation.as_mut().ok_or(NativeError::WrongState)?.poll() {
                Ok(AsyncPoll::Pending) => return Ok(CleanupProgress::Pending),
                Ok(AsyncPoll::Ready(())) => {}
                Err(error) => {
                    if self.stop_operation.as_ref().is_some_and(PendingAction::is_terminal) {
                        self.stop_operation = None;
                    }
                    return Err(error);
                }
            }
            self.stop_operation = None;
            self.stop_required = false;
        }
        if !self.reader_closed {
            self.reader.as_ref().ok_or(NativeError::WrongState)?.Close()?;
            self.reader_closed = true;
            self.reader = None;
        }
        if !self.capture_closed {
            self.capture.as_ref().ok_or(NativeError::WrongState)?.Close()?;
            self.capture_closed = true;
            self.capture = None;
        }
        Ok(CleanupProgress::Complete)
    }
}

/// Metadata paired with the caller-buffered bytes produced by one acquisition.
#[cfg(feature = "unqualified-frame-copy")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowsCapturedFrame {
    pub format: FrameFormat,
    pub metadata: FrameMetadata,
    /// Raw Windows SystemRelativeTime in 100 ns ticks when reported. It is
    /// not UTC and is not mapped into the caller's ClockStamp domain here.
    pub system_relative_time_100ns: Option<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReaderState { Active, Stopping, StopFailed, Stopped }

/// An explicitly running reader that performs a policy check and bounded
/// `SoftwareBitmap` copy for each caller poll. WinRT may continuously acquire
/// and buffer frames between polls; the policy check gates calls through this
/// API, not the camera's internal collection. The host must keep its live
/// authorization valid for the entire running interval and start asynchronous
/// stop immediately when it is withdrawn. It holds the `MediaCapture` and
/// reader COM references until async stop and close complete.
///
/// Call `request_stop` and poll `poll_stop` to completion before dropping this
/// value. Drop cannot synchronously await `StopAsync`; abandoning this value
/// while active cannot confirm the reader stopped.
#[must_use = "stop the active reader and poll poll_stop() to completion before dropping"]
pub struct RunningFrameReader {
    capture: MediaCapture,
    reader: MediaFrameReader,
    declaration: FrameDeclaration,
    state: ReaderState,
    stop_operation: Option<PendingAction>,
    reader_closed: bool,
    capture_closed: bool,
    next_sequence: u64,
}

impl RunningFrameReader {
    pub fn from_started(
        capture: MediaCapture,
        reader: MediaFrameReader,
        declaration: FrameDeclaration,
        status: MediaFrameReaderStartStatus,
        policy: &mut impl CameraPolicy,
    ) -> Result<Self, StartFailure> {
        if declaration.source_id() == 0 {
            return Err(StartFailure::new(
                StartFailureReason::InvalidDeclaration,
                capture,
                reader,
                status == MediaFrameReaderStartStatus::Success,
            ));
        }
        if status != MediaFrameReaderStartStatus::Success {
            return Err(StartFailure::new(
                StartFailureReason::StartStatus(status.0),
                capture,
                reader,
                false,
            ));
        }
        let mut running = Self {
            capture,
            reader,
            declaration,
            state: ReaderState::Active,
            stop_operation: None,
            reader_closed: false,
            capture_closed: false,
            next_sequence: 0,
        };
        // Recheck after the asynchronous Start operation completes. If the
        // permit expired while pending, immediately request asynchronous stop.
        if !policy_permits(
            policy,
            running.declaration.source_id(),
            running.declaration.modality(),
            CameraOperation::Start,
        ) {
            let _ = running.request_stop();
        }
        Ok(running)
    }

    pub const fn is_active(&self) -> bool { matches!(self.state, ReaderState::Active) }
    pub const fn is_stopping(&self) -> bool { matches!(self.state, ReaderState::Stopping | ReaderState::StopFailed) }

    /// Acquire/copy one latest frame. Caller output must hold the declaration's
    /// full byte ceiling before the OS frame is acquired; actual dimensions and
    /// pixel packing are checked against the declaration before bytes escape.
    ///
    /// Latest-frame polling can skip device frames; `gap_before` remains None
    /// unless a future qualified API provides a reliable dropped-frame count.
    #[cfg(feature = "unqualified-frame-copy")]
    pub fn capture_into(
        &mut self,
        policy: &mut impl CameraPolicy,
        allocation: CopyAllocationClass,
        output: &mut [u8],
    ) -> Result<WindowsCapturedFrame, NativeError> {
        output.fill(0);
        if self.state != ReaderState::Active { return Err(NativeError::WrongState); }
        let byte_budget = validate_copy_allocation(
            allocation, u64::from(self.declaration.max_frame_bytes()),
        )?;
        if !policy_permits(policy, self.declaration.source_id(), self.declaration.modality(), CameraOperation::AcquireFrame) {
            self.request_stop()?;
            return Err(NativeError::PolicyDenied);
        }
        let required_capacity = self.declaration.max_frame_bytes() as usize;
        if output.len() < required_capacity {
            return Err(NativeError::BufferTooSmall { required: required_capacity });
        }
        let next_sequence = match self.next_sequence.checked_add(1) {
            Some(sequence) => sequence,
            None => { let _ = self.request_stop(); return Err(NativeError::SequenceOverflow); }
        };
        let mut guarded = FrameOutputGuard { bytes: output, keep: false };
        let copied = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let reference = self.reader.TryAcquireLatestFrame()?;
            let native_time = reference.SystemRelativeTime().ok()
                .and_then(|value| value.Value().ok())
                .map(|value| value.Duration);
            let video = reference.VideoMediaFrame()?;
            let bitmap = video.SoftwareBitmap()?;
            copy_bitmap(&bitmap, self.declaration, byte_budget, guarded.bytes)
                .map(|format| (format, native_time))
        }));
        let copied = match copied {
            Ok(result) => result,
            Err(panic) => {
                let _ = self.request_stop();
                std::panic::resume_unwind(panic);
            }
        };
        match copied {
            Ok((format, native_time)) => {
                if !policy_permits(policy, self.declaration.source_id(), self.declaration.modality(), CameraOperation::AcquireFrame) {
                    let _ = self.request_stop();
                    return Err(NativeError::PolicyDenied);
                }
                self.next_sequence = next_sequence;
                let frame = WindowsCapturedFrame {
                    format,
                    metadata: FrameMetadata {
                        format_id: format.format_id(),
                        sequence: next_sequence - 1,
                        gap_before: None,
                        timestamp: FrameTimestamp::Unknown,
                    },
                    system_relative_time_100ns: native_time,
                };
                guarded.keep = true;
                Ok(frame)
            }
            Err(error) => {
                let _ = self.request_stop();
                Err(error)
            }
        }
    }

    /// Request nonblocking stop. Completion must be observed with `poll_stop`.
    pub fn request_stop(&mut self) -> Result<(), NativeError> {
        match self.state {
            ReaderState::Active | ReaderState::StopFailed => {
                match begin_reader_stop(&self.capture, &self.reader) {
                    Ok(operation) => {
                        self.stop_operation = Some(operation);
                        self.state = ReaderState::Stopping;
                        Ok(())
                    }
                    Err(error) => { self.state = ReaderState::StopFailed; Err(error) }
                }
            }
            ReaderState::Stopping => Ok(()),
            ReaderState::Stopped => Ok(()),
        }
    }

    /// Poll stop without blocking. Once the stop action completes, explicitly
    /// close the reader and capture. Close failures remain retryable.
    pub fn poll_stop(&mut self) -> Result<AsyncPoll<()>, NativeError> {
        if self.state == ReaderState::Active { return Err(NativeError::WrongState); }
        if self.state == ReaderState::StopFailed { return Err(NativeError::StopUnconfirmed); }
        if self.state == ReaderState::Stopping {
            let poll = self.stop_operation.as_mut().ok_or(NativeError::WrongState)?.poll();
            match poll {
                Ok(AsyncPoll::Pending) => return Ok(AsyncPoll::Pending),
                Ok(AsyncPoll::Ready(())) => {
                    self.stop_operation = None;
                    self.state = ReaderState::Stopped;
                }
                Err(error) => {
                    if self.stop_operation.as_ref().is_some_and(PendingAction::is_terminal) {
                        self.stop_operation = None;
                        self.state = ReaderState::StopFailed;
                    }
                    return Err(error);
                }
            }
        }
        self.close_after_stop()?;
        Ok(AsyncPoll::Ready(()))
    }

    fn close_after_stop(&mut self) -> Result<(), NativeError> {
        if self.state != ReaderState::Stopped { return Err(NativeError::WrongState); }
        if !self.reader_closed {
            self.reader.Close()?;
            self.reader_closed = true;
        }
        if !self.capture_closed {
            self.capture.Close()?;
            self.capture_closed = true;
        }
        Ok(())
    }
}

#[cfg(feature = "unqualified-frame-copy")]
struct FrameOutputGuard<'a> { bytes: &'a mut [u8], keep: bool }

#[cfg(feature = "unqualified-frame-copy")]
impl Drop for FrameOutputGuard<'_> {
    fn drop(&mut self) { if !self.keep { self.bytes.fill(0); } }
}

