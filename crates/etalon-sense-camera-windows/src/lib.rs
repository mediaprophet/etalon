#![forbid(unsafe_code)]
//! Low-level Windows MediaCapture / MediaFrameReader components.
//!
//! This crate does not choose a device, infer visible/IR/thermal modality, or
//! claim radiometric calibration. Capture is started explicitly and a reader
//! remains a continuous OS/device stream until its asynchronous stop completes.
//! Hosts must recheck their live policy before every frame acquisition and
//! initiate stop on expiry/revocation; a callback/poll gate cannot prove that
//! the OS stopped buffering immediately. No blocking `.get()` helpers are used.
//!
//! The module deliberately exposes WinRT state transitions rather than
//! implementing the synchronous `FrameProvider` trait: initialization, reader
//! creation, start, and stop are asynchronous. Callers must poll completion on
//! their event loop, then explicitly stop and poll teardown before dropping an
//! active reader.
//!
//! Windows capture APIs are disabled by default. The optional
//! `unqualified-winrt-capture` feature exposes the low-level operations but is
//! not device-qualified and cannot guarantee immediate stop on revocation or
//! drop. The nested `unqualified-frame-copy` feature additionally accepts only
//! tightly packed single-plane SoftwareBitmap layouts; CopyToBuffer behavior
//! still needs Windows/device qualification before production use. The copy
//! call requires a `CopyAllocationClass`: strict `HotZeroHeap` admission is
//! rejected, while `ColdBounded` admits only an explicitly budgeted payload
//! size. WinRT object/runtime overhead is not measured by this crate.

#[cfg(all(windows, feature = "unqualified-winrt-capture"))]
mod winrt;

#[cfg(all(windows, feature = "unqualified-winrt-capture"))]
mod lifecycle;

#[cfg(all(windows, feature = "unqualified-winrt-capture"))]
mod discovery;

#[cfg(all(windows, feature = "unqualified-frame-copy"))]
mod observation;

#[cfg(all(windows, feature = "unqualified-winrt-capture"))]
pub use discovery::{
    CameraSourceAuthority, CameraSourceCandidate, MAX_CAMERA_SOURCES, MAX_CAMERA_SOURCE_ID_UNITS,
};

#[cfg(all(windows, feature = "unqualified-winrt-capture"))]
pub use lifecycle::{CameraPipeline, PipelineProgress};

#[cfg(all(windows, feature = "unqualified-winrt-capture"))]
pub use winrt::{
    begin_create_reader, begin_initialize, begin_reader_start, begin_reader_stop, AsyncPoll,
    CameraOperation, CameraPolicy, CleanupProgress, CopyAllocationClass, NativeError,
    PendingAction, PendingReader, PendingReaderStart, RunningFrameReader, StartFailure,
    StartFailureReason,
};

#[cfg(all(windows, feature = "unqualified-frame-copy"))]
pub use winrt::WindowsCapturedFrame;

#[cfg(all(windows, feature = "unqualified-frame-copy"))]
pub use observation::associate_windows_frame;
