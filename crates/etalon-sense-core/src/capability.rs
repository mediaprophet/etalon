//! Validated declarations of sensor data capabilities.
//!
//! These declarations describe data shapes and units only. They do not imply
//! that a backend exists, a device is connected, a stream is trusted, or an
//! operation is authorized.

use crate::{
    quantity::{PhysicalQuantity, SampleEncoding, SampleRate, Unit},
    ContractError,
};

/// Physical or logical class of sensor source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    Wifi,
    Bluetooth,
    Cellular,
    Sdr,
    Radar,
    VisibleCamera,
    NearInfraredCamera,
    ThermalCamera,
    Microphone,
    FieldInstrument,
}

/// Validated description of one data stream a source may provide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SensorCapability {
    modality: Modality,
    quantity: PhysicalQuantity,
    unit: Unit,
    encoding: SampleEncoding,
    max_extents: [u32; 4],
    max_rank: u8,
    max_sample_rate: Option<SampleRate>,
    max_payload_bytes: u64,
}

impl SensorCapability {
    /// Declare a bounded stream. Extents have rank 1–4; unused trailing axes
    /// are zero. No capture backend is selected or inferred.
    pub fn new(
        modality: Modality,
        quantity: PhysicalQuantity,
        unit: Unit,
        encoding: SampleEncoding,
        max_extents: [u32; 4],
        max_sample_rate: Option<SampleRate>,
    ) -> Result<Self, ContractError> {
        if !quantity.accepts_unit(unit) {
            return Err(ContractError::IncompatibleUnit);
        }
        if !quantity.accepts_encoding(encoding) {
            return Err(ContractError::IncompatibleEncoding);
        }
        let rank = max_extents
            .iter()
            .position(|&extent| extent == 0)
            .unwrap_or(4);
        if rank == 0 || max_extents[rank..].iter().any(|&extent| extent != 0) {
            return Err(ContractError::InvalidExtent);
        }

        let mut elements = 1u64;
        for extent in &max_extents[..rank] {
            elements = elements
                .checked_mul(u64::from(*extent))
                .ok_or(ContractError::ArithmeticOverflow)?;
        }
        let max_payload_bytes = elements
            .checked_mul(u64::from(encoding.element_bytes()))
            .ok_or(ContractError::ArithmeticOverflow)?;

        Ok(Self {
            modality,
            quantity,
            unit,
            encoding,
            max_extents,
            max_rank: rank as u8,
            max_sample_rate,
            max_payload_bytes,
        })
    }

    pub const fn modality(self) -> Modality {
        self.modality
    }

    pub const fn quantity(self) -> PhysicalQuantity {
        self.quantity
    }

    pub const fn unit(self) -> Unit {
        self.unit
    }

    pub const fn encoding(self) -> SampleEncoding {
        self.encoding
    }

    pub const fn max_extents(self) -> [u32; 4] {
        self.max_extents
    }

    pub const fn max_rank(self) -> u8 {
        self.max_rank
    }

    pub const fn max_sample_rate(self) -> Option<SampleRate> {
        self.max_sample_rate
    }

    /// Maximum tightly packed payload size for the declared extents.
    pub const fn max_payload_bytes(self) -> u64 {
        self.max_payload_bytes
    }
}

#[cfg(test)]
#[path = "capability_tests.rs"]
mod tests;
