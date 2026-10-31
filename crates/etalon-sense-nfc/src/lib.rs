#![deny(unsafe_code)]
//! Bounded host-fed NFC observation and capture boundary.
//!
//! This crate does not discover readers, poll tags, or provide an NFC driver.
//! A host `NfcProvider` supplies one record at a time into a fixed UID buffer.
//! UID bytes are opaque radio/tag labels, not verified person, object, or
//! hardware identities. Protocol, tag kind, proximity, source, and timestamps
//! remain host/provider reports. A mapped `ClockStamp` is optional; no clock
//! conversion or calibration is performed here.

mod observation;

pub use observation::{
    capture_next, HostNfcMetadata, NfcError, NfcOperation, NfcPolicyGate,
    NfcProximity, NfcProvider, NfcProviderError, NfcProtocol, NfcTagKind,
    NfcObservation, MAX_NFC_UID_BYTES,
};

#[cfg(test)]
mod tests;
