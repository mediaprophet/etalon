//! Bounded radio labels, separate from collecting-source and person identity.
//!
//! These are observation metadata: labels can rotate, be reused, or be spoofed.
//! Host policy and provenance remain in the associated observation envelope.

use crate::{time::ClockStamp, ContractError};

/// Six octets in displayed/network order, including zero and broadcast values.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RadioAddress48([u8; 6]);

impl RadioAddress48 {
    pub const fn from_octets(octets: [u8; 6]) -> Self {
        Self(octets)
    }

    /// The first octet occupies bits 40..47. Rejects rather than truncates.
    pub const fn try_from_u64(value: u64) -> Result<Self, ContractError> {
        if value > 0x0000_ffff_ffff_ffff {
            return Err(ContractError::InvalidIdentifier);
        }
        let bytes = value.to_be_bytes();
        Ok(Self([
            bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    pub const fn octets(self) -> [u8; 6] {
        self.0
    }

    pub const fn as_u64(self) -> u64 {
        let [a, b, c, d, e, f] = self.0;
        u64::from_be_bytes([0, 0, a, b, c, d, e, f])
    }
}

/// Backend-reported Bluetooth address type, never inferred from address bits.
/// `Public` is an address type, not a visibility, permission, or trust result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BluetoothAddressKind {
    /// Includes absent, unavailable, and unrecognized backend type reports.
    Unknown,
    Public,
    Random,
}

/// Namespace and reported role are retained even for identical address bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RadioIdentifier {
    /// A MAC label whose more specific radio role is not available.
    Mac(RadioAddress48),
    /// A reported Wi-Fi basic-service-set label, not a collecting-source ID.
    Bssid(RadioAddress48),
    Bluetooth {
        address: RadioAddress48,
        kind: BluetoothAddressKind,
    },
}

impl RadioIdentifier {
    pub const fn address(self) -> RadioAddress48 {
        match self {
            Self::Mac(address) | Self::Bssid(address) | Self::Bluetooth { address, .. } => address,
        }
    }
}

/// Timestamp reported for the label observation, without inferred cache age.
/// Windows wall-clock ticks are not a synchronized monotonic `ClockStamp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadioObservationTime {
    Unknown,
    Clock(ClockStamp),
    /// Raw signed Windows `DateTime::UniversalTime` ticks, 100 ns since 1601 UTC.
    WindowsUtc100ns(i64),
}

/// Side metadata for an associated `ObservationEnvelope`; equality compares
/// reports only. Keep the envelope's source, stream, policy and origin intact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RadioIdentifierObservation {
    pub identifier: RadioIdentifier,
    pub observed_at: RadioObservationTime,
}

#[cfg(test)]
#[path = "radio_tests.rs"]
mod tests;
