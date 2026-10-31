#![deny(unsafe_op_in_unsafe_fn)]
//! Read-only Windows WLAN cached-BSS observations.
//!
//! This adapter asks Windows for its current BSS cache. It does not trigger a
//! scan, enter monitor mode, capture packets, or produce CSI. Cache age and
//! timestamp clock mapping are unknown. RSSI and link quality remain distinct
//! fields. A BSSID is only the raw link-layer address reported by the OS; it
//! makes no claim about a person or physical identity.

#[cfg(windows)]
mod native;

#[cfg(windows)]
pub mod observation;

#[cfg(windows)]
pub use native::{
    read_cached_bss, read_current_interface_mac, BssObservation, BssReadSummary,
    InterfaceMacPolicyGate, PolicyGate, WifiError,
    MAX_BSS_OBSERVATIONS,
};

#[cfg(windows)]
pub use observation::{
    adapt_cached_bss, AdaptError as ObservationAdaptError, WifiMetric,
    WifiObservation, WifiObservationMapping, WIFI_METRIC_CAPACITY,
};

/// Returned observations have no verified source-clock timestamp or cache age.
/// This is deliberately explicit so callers cannot treat scan-cache contents
/// as live, synchronized samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimestampStatus {
    Unknown,
}

