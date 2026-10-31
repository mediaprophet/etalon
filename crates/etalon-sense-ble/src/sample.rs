use etalon_sense_core::{
    radio::{BluetoothAddressKind, RadioAddress48, RadioIdentifier, RadioIdentifierObservation, RadioObservationTime},
    ContractError,
};
use windows::Devices::Bluetooth::BluetoothAddressType;

/// One OS-reported BLE advertisement. The address is not authenticated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BleAdvertisementSample {
    pub bluetooth_address: u64,
    pub rssi_dbm: i16,
    /// Windows DateTime::UniversalTime, 100 ns ticks from 1601-01-01 UTC.
    pub os_timestamp_100ns: i64,
    /// Raw Windows BluetoothAddressType value; None when unavailable or failed.
    pub os_address_type: Option<i32>,
}

impl BleAdvertisementSample {
    /// Classifies only the OS type report, never bits of the address itself.
    pub fn bluetooth_address_kind(self) -> BluetoothAddressKind {
        match self.os_address_type {
            Some(value) if value == BluetoothAddressType::Public.0 => BluetoothAddressKind::Public,
            Some(value) if value == BluetoothAddressType::Random.0 => BluetoothAddressKind::Random,
            _ => BluetoothAddressKind::Unknown,
        }
    }

    /// Keep this side record with the batch envelope's collecting source and
    /// policy. Raw Windows wall-clock ticks do not establish synchronization.
    pub fn radio_identifier_observation(self) -> Result<RadioIdentifierObservation, ContractError> {
        let address = RadioAddress48::try_from_u64(self.bluetooth_address)?;
        Ok(RadioIdentifierObservation {
            identifier: RadioIdentifier::Bluetooth { address, kind: self.bluetooth_address_kind() },
            observed_at: RadioObservationTime::WindowsUtc100ns(self.os_timestamp_100ns),
        })
    }
}

pub(super) fn raw_address_type<E>(
    reported: Result<BluetoothAddressType, E>,
) -> Option<i32> {
    reported.ok().map(|value| value.0)
}

#[cfg(test)]
#[path = "sample_tests.rs"]
mod tests;
