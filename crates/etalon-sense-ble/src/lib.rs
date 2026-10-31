//! Windows BLE advertisement RSSI capture using a bounded event queue.
//!
//! The adapter reports the operating system's received dBm value, Bluetooth
//! address, and raw Windows `DateTime::UniversalTime` ticks. An address is a
//! radio-level label that can be randomized or spoofed; it does not identify a
//! person or device owner. RSSI is not distance, CSI, or calibrated channel
//! data. The host supplies the quantity and unit registry identifiers.
//!
//! Starting the watcher requires a fresh crate-issued capture request. The
//! caller must also provide a live host gate that rechecks trusted time,
//! policy, and revocation before event fields are read and during capture.
//! The host application must declare Windows Bluetooth capability and satisfy
//! OS privacy settings. Passive scanning uses a fixed queue and bounded sample
//! and session duration limits; overflow drops newest advertisements. RSSI
//! payload samples are signed I16 values encoded little-endian. The host gate
//! and OS Stop call must return; teardown cannot be guaranteed if either
//! external operation stalls or the OS fails to stop the watcher.

#![deny(unsafe_code)]

#[cfg(windows)]
mod queue;

#[cfg(windows)]
mod sample;

#[cfg(windows)]
mod config;

#[cfg(windows)]
#[allow(unsafe_code)]
mod runtime;

#[cfg(windows)]
#[allow(unsafe_code)]
mod capture;

#[cfg(windows)]
pub mod observation;

#[cfg(windows)]
pub use capture::{
    BleRssiAdapter,
};

#[cfg(windows)]
pub use observation::{
    capture_observation, BleBatchMetadata, BleDropCountInterval, BleObservation, BleObservationMapping,
    ObservationError as BleObservationError,
};

#[cfg(windows)]
pub use config::{
    BleCollectionGate, BleCollectionScope, BleConfig,
    MAX_CAPTURE_SAMPLES, MAX_SESSION_DURATION_MS, MAX_SESSION_OBSERVATIONS,
};

#[cfg(windows)]
pub use sample::BleAdvertisementSample;

#[cfg(not(windows))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BleUnavailable;
