mod reader;
#[cfg(feature = "unqualified-frame-copy")]
pub use reader::WindowsCapturedFrame;
pub use reader::{CleanupProgress, RunningFrameReader, StartFailure, StartFailureReason};
mod copy;
#[cfg(feature = "unqualified-frame-copy")]
use copy::copy_bitmap;
use copy::validate_copy_allocation;

use etalon_sense_camera::{CameraModality, FrameDeclaration};
#[cfg(feature = "unqualified-frame-copy")]
use etalon_sense_camera::{
    FrameFormat, FrameMetadata, FramePixelFormat, FramePlane, FrameTimestamp,
};
#[cfg(feature = "unqualified-frame-copy")]
use windows::{
    Graphics::Imaging::{BitmapBufferAccessMode, BitmapPixelFormat, SoftwareBitmap},
    Storage::Streams::{Buffer, DataReader},
};
use windows::{
    Media::Capture::Frames::{
        MediaFrameReader, MediaFrameReaderStartStatus, MediaFrameReference, MediaFrameSource,
    },
    Media::Capture::{MediaCapture, MediaCaptureInitializationSettings},
};
use windows_future::{AsyncStatus, IAsyncAction, IAsyncOperation};

/// Nonblocking result of polling a WinRT operation.
pub enum AsyncPoll<T> {
    Pending,
    Ready(T),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeError {
    Windows(i32),
    Canceled,
    OperationFailed(i32),
    AlreadyCompleted,
    WrongState,
    InvalidSource,
    PolicyDenied,
    BufferTooSmall {
        required: usize,
    },
    UnsupportedPixelFormat,
    UnsupportedLayout,
    HotZeroHeapDenied,
    ColdAllocationBudgetExceeded {
        required_bytes: u64,
        budget_bytes: u64,
    },
    InvalidFrame,
    ArithmeticOverflow,
    SequenceOverflow,
    StopUnconfirmed,
    SourceListChanged,
    SourceLimitExceeded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraOperation {
    Initialize,
    EnumerateSources,
    SelectSource,
    CreateReader,
    Start,
    AcquireFrame,
}

/// Admission class for the WinRT intermediate sample buffer. Strict hot
/// zero-heap work cannot use this API because `Buffer::Create` and the WinRT
/// stream wrappers allocate. `byte_budget` bounds the explicitly requested
/// frame payload buffer only; Windows object/runtime overhead is not measured
/// or claimed to fit within that value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyAllocationClass {
    HotZeroHeap,
    ColdBounded { byte_budget: u64 },
}

/// Host-owned policy hook. Callers may implement public, scientific, or other
/// ontology contexts; this crate supplies no universal identity/consent model.
pub trait CameraPolicy {
    fn permit(
        &mut self,
        source_id: u64,
        modality: CameraModality,
        operation: CameraOperation,
    ) -> bool;
}

impl From<windows::core::Error> for NativeError {
    fn from(error: windows::core::Error) -> Self {
        Self::Windows(error.code().0)
    }
}

/// A pending action, such as MediaCapture initialization or reader stop.
/// `poll` is nonblocking. `cancel` requests operation cancellation; it does not
/// prove that a camera or OS buffer has stopped until completion is observed.
#[must_use = "retain and poll or cancel the asynchronous action to terminal state"]
pub struct PendingAction {
    operation: IAsyncAction,
    // Keep dependent WinRT objects alive until the async operation reaches a
    // terminal state. The caller still owns cancellation/teardown sequencing.
    _capture: Option<MediaCapture>,
    _settings: Option<MediaCaptureInitializationSettings>,
    _reader: Option<MediaFrameReader>,
    terminal: bool,
}

impl PendingAction {
    pub(crate) const fn is_terminal(&self) -> bool {
        self.terminal
    }
    fn new(
        operation: IAsyncAction,
        capture: Option<MediaCapture>,
        settings: Option<MediaCaptureInitializationSettings>,
        reader: Option<MediaFrameReader>,
    ) -> Self {
        Self {
            operation,
            _capture: capture,
            _settings: settings,
            _reader: reader,
            terminal: false,
        }
    }

    pub fn poll(&mut self) -> Result<AsyncPoll<()>, NativeError> {
        if self.terminal {
            return Err(NativeError::AlreadyCompleted);
        }
        match self.operation.Status()? {
            AsyncStatus::Started => Ok(AsyncPoll::Pending),
            AsyncStatus::Completed => {
                self.terminal = true;
                self.operation.GetResults()?;
                Ok(AsyncPoll::Ready(()))
            }
            AsyncStatus::Canceled => {
                self.terminal = true;
                Err(NativeError::Canceled)
            }
            AsyncStatus::Error => {
                self.terminal = true;
                Err(NativeError::OperationFailed(self.operation.ErrorCode()?.0))
            }
            _ => {
                self.terminal = true;
                Err(NativeError::WrongState)
            }
        }
    }

    pub fn cancel(&self) -> Result<(), NativeError> {
        if self.terminal {
            return Err(NativeError::AlreadyCompleted);
        }
        self.operation.Cancel().map_err(Into::into)
    }
}

/// Pending `MediaCapture::CreateFrameReaderAsync` operation.
#[must_use = "retain and poll or cancel reader creation to terminal state"]
pub struct PendingReader {
    operation: IAsyncOperation<MediaFrameReader>,
    _capture: MediaCapture,
    _source: MediaFrameSource,
    _declaration: FrameDeclaration,
    terminal: bool,
}

impl PendingReader {
    pub(crate) const fn is_terminal(&self) -> bool {
        self.terminal
    }
    fn new(
        operation: IAsyncOperation<MediaFrameReader>,
        capture: MediaCapture,
        source: MediaFrameSource,
        declaration: FrameDeclaration,
    ) -> Self {
        Self {
            operation,
            _capture: capture,
            _source: source,
            _declaration: declaration,
            terminal: false,
        }
    }

    pub fn poll(&mut self) -> Result<AsyncPoll<MediaFrameReader>, NativeError> {
        if self.terminal {
            return Err(NativeError::AlreadyCompleted);
        }
        match self.operation.Status()? {
            AsyncStatus::Started => Ok(AsyncPoll::Pending),
            AsyncStatus::Completed => {
                self.terminal = true;
                self.operation
                    .GetResults()
                    .map(AsyncPoll::Ready)
                    .map_err(Into::into)
            }
            AsyncStatus::Canceled => {
                self.terminal = true;
                Err(NativeError::Canceled)
            }
            AsyncStatus::Error => {
                self.terminal = true;
                Err(NativeError::OperationFailed(self.operation.ErrorCode()?.0))
            }
            _ => {
                self.terminal = true;
                Err(NativeError::WrongState)
            }
        }
    }

    pub fn cancel(&self) -> Result<(), NativeError> {
        if self.terminal {
            return Err(NativeError::AlreadyCompleted);
        }
        self.operation.Cancel().map_err(Into::into)
    }
}

/// Pending reader start operation. A `RunningFrameReader` may be built only
/// after this operation returns `MediaFrameReaderStartStatus::Success`. If
/// authority changes while it is pending, the host must call `cancel`, keep
/// polling to terminal state, then request/poll reader stop before releasing
/// its capture objects. Cancel is asynchronous and not an immediate stop proof.
#[must_use = "retain and poll or cancel reader start to terminal state"]
pub struct PendingReaderStart {
    operation: IAsyncOperation<MediaFrameReaderStartStatus>,
    _capture: MediaCapture,
    _reader: MediaFrameReader,
    _declaration: FrameDeclaration,
    terminal: bool,
}

impl PendingReaderStart {
    pub(crate) const fn is_terminal(&self) -> bool {
        self.terminal
    }
    fn new(
        operation: IAsyncOperation<MediaFrameReaderStartStatus>,
        capture: MediaCapture,
        reader: MediaFrameReader,
        declaration: FrameDeclaration,
    ) -> Self {
        Self {
            operation,
            _capture: capture,
            _reader: reader,
            _declaration: declaration,
            terminal: false,
        }
    }

    pub fn poll(&mut self) -> Result<AsyncPoll<MediaFrameReaderStartStatus>, NativeError> {
        if self.terminal {
            return Err(NativeError::AlreadyCompleted);
        }
        match self.operation.Status()? {
            AsyncStatus::Started => Ok(AsyncPoll::Pending),
            AsyncStatus::Completed => {
                self.terminal = true;
                self.operation
                    .GetResults()
                    .map(AsyncPoll::Ready)
                    .map_err(Into::into)
            }
            AsyncStatus::Canceled => {
                self.terminal = true;
                Err(NativeError::Canceled)
            }
            AsyncStatus::Error => {
                self.terminal = true;
                Err(NativeError::OperationFailed(self.operation.ErrorCode()?.0))
            }
            _ => {
                self.terminal = true;
                Err(NativeError::WrongState)
            }
        }
    }

    pub fn cancel(&self) -> Result<(), NativeError> {
        if self.terminal {
            return Err(NativeError::AlreadyCompleted);
        }
        self.operation.Cancel().map_err(Into::into)
    }
}

pub fn begin_initialize(
    capture: &MediaCapture,
    settings: &MediaCaptureInitializationSettings,
    source_id: u64,
    modality: CameraModality,
    policy: &mut impl CameraPolicy,
) -> Result<PendingAction, NativeError> {
    // `source_id` is the host's binding for the selected source/group; this
    // projection cannot authenticate that external association.
    if source_id == 0 {
        return Err(NativeError::InvalidSource);
    }
    if !policy_permits(policy, source_id, modality, CameraOperation::Initialize) {
        return Err(NativeError::PolicyDenied);
    }
    Ok(PendingAction::new(
        capture.InitializeWithSettingsAsync(settings)?,
        Some(capture.clone()),
        Some(settings.clone()),
        None,
    ))
}

pub fn begin_create_reader(
    capture: &MediaCapture,
    source: &MediaFrameSource,
    declaration: FrameDeclaration,
    policy: &mut impl CameraPolicy,
) -> Result<PendingReader, NativeError> {
    // The caller selects this source and asserts its ID/modality binding in the
    // declaration. WinRT's projected object does not authenticate that mapping.
    if declaration.source_id() == 0 {
        return Err(NativeError::InvalidSource);
    }
    if !policy_permits(
        policy,
        declaration.source_id(),
        declaration.modality(),
        CameraOperation::CreateReader,
    ) {
        return Err(NativeError::PolicyDenied);
    }
    Ok(PendingReader::new(
        capture.CreateFrameReaderAsync(source)?,
        capture.clone(),
        source.clone(),
        declaration,
    ))
}

pub fn begin_reader_start(
    capture: &MediaCapture,
    reader: &MediaFrameReader,
    declaration: FrameDeclaration,
    policy: &mut impl CameraPolicy,
) -> Result<PendingReaderStart, NativeError> {
    if declaration.source_id() == 0 {
        return Err(NativeError::InvalidSource);
    }
    if !policy_permits(
        policy,
        declaration.source_id(),
        declaration.modality(),
        CameraOperation::Start,
    ) {
        return Err(NativeError::PolicyDenied);
    }
    Ok(PendingReaderStart::new(
        reader.StartAsync()?,
        capture.clone(),
        reader.clone(),
        declaration,
    ))
}

pub fn begin_reader_stop(
    capture: &MediaCapture,
    reader: &MediaFrameReader,
) -> Result<PendingAction, NativeError> {
    Ok(PendingAction::new(
        reader.StopAsync()?,
        Some(capture.clone()),
        None,
        Some(reader.clone()),
    ))
}

fn policy_permits(
    policy: &mut impl CameraPolicy,
    source_id: u64,
    modality: CameraModality,
    operation: CameraOperation,
) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        policy.permit(source_id, modality, operation)
    }))
    .unwrap_or(false)
}
