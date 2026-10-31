//! Windows WASAPI capture for Etalon's caller-buffered sensing contract.
//!
//! This crate binds capture to the default active input endpoint when the
//! caller invokes `discover_backend`; actual audio starts only after the host
//! supplies an authorized request to `start`. It does not bypass Windows
//! microphone privacy settings or interpret host policy decisions.
//! Unwinding capture faults the session, clears caller samples and attempts OS
//! stop before resuming the panic. Abort-mode and failed OS stop cannot provide
//! teardown confirmation; hosts must retry explicit stop where possible.
//! `WasapiCaptureAdapter::with_remainder` borrows explicitly budgeted caller
//! scratch for packet tails, allowing batches smaller than native packets.
//! Startup checks scratch against the negotiated endpoint buffer capacity.
//! Scratch is wiped on fault, stop and drop; no QPC offset is inferred for tails.

#![deny(unsafe_code)]

#[cfg(windows)]
mod config;

#[cfg(windows)]
#[allow(unsafe_code)]
mod capture;

#[cfg(windows)]
#[allow(unsafe_code)]
mod format;

#[cfg(windows)]
mod helpers;
#[cfg(windows)]
mod remainder;

#[cfg(windows)]
pub mod observation;

#[cfg(windows)]
pub use capture::{CaptureMetadata, WasapiCaptureAdapter};

#[cfg(windows)]
pub use config::WasapiConfig;

#[cfg(windows)]
pub use format::WasapiFormat;

#[cfg(windows)]
pub use observation::{
    capture_observation, ObservationError as WasapiObservationError,
    WasapiBatchMetadata, WasapiObservation, WasapiObservationMapping,
};

#[cfg(not(windows))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WasapiUnavailable;
