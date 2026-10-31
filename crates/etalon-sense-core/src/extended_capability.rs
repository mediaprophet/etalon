//! Extensible stream descriptions for modalities and quantities outside the
//! built-in catalog. Registry identity and dimensional compatibility are
//! supplied by the host; numeric IDs are not authenticated or interpreted.
use crate::{buffer::SampleLayout, quantity::{SampleEncoding, SampleRate}, ContractError};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtendedCapability {
    modality_id: u64,
    quantity_id: u64,
    unit_id: Option<u64>,
    encoding: SampleEncoding,
    max_extents: [u32; 4],
    rank: u8,
    max_sample_rate: Option<SampleRate>,
    max_payload_bytes: u64,
}

impl ExtendedCapability {
    /// Declare an extensible stream. Unit None is unknown, not unitless. This
    /// checks shape and byte limits, not registry membership or unit meaning.
    pub fn new(modality_id: u64, quantity_id: u64, unit_id: Option<u64>, encoding: SampleEncoding, max_extents: [u32; 4], max_sample_rate: Option<SampleRate>) -> Result<Self, ContractError> {
        if modality_id == 0 || quantity_id == 0 || matches!(unit_id, Some(0)) { return Err(ContractError::InvalidIdentifier); }
        let mut rank = 0usize;
        let mut ended = false;
        let mut elements = 1u64;
        for extent in max_extents {
            if extent == 0 { ended = true; continue; }
            if ended { return Err(ContractError::InvalidExtent); }
            rank += 1;
            elements = elements.checked_mul(u64::from(extent)).ok_or(ContractError::ArithmeticOverflow)?;
        }
        if rank == 0 { return Err(ContractError::InvalidExtent); }
        let max_payload_bytes = elements.checked_mul(u64::from(encoding.element_bytes())).ok_or(ContractError::ArithmeticOverflow)?;
        Ok(Self { modality_id, quantity_id, unit_id, encoding, max_extents, rank: rank as u8, max_sample_rate, max_payload_bytes })
    }

    pub fn validate_layout(&self, layout: &SampleLayout, rate: Option<SampleRate>) -> Result<(), ContractError> {
        if *layout.encoding() != self.encoding { return Err(ContractError::IncompatibleEncoding); }
        if layout.rank() != self.rank { return Err(ContractError::InvalidExtent); }
        for (extent, max) in layout.extents().iter().zip(self.max_extents.iter()) {
            if extent > max { return Err(ContractError::InvalidExtent); }
        }
        if let (Some(actual), Some(max)) = (rate, self.max_sample_rate) {
            if actual.hz() > max.hz() { return Err(ContractError::InvalidRate); }
        }
        Ok(())
    }

    pub const fn modality_id(self) -> u64 { self.modality_id }
    pub const fn quantity_id(self) -> u64 { self.quantity_id }
    pub const fn unit_id(self) -> Option<u64> { self.unit_id }
    pub const fn encoding(self) -> SampleEncoding { self.encoding }
    pub const fn max_extents(self) -> [u32; 4] { self.max_extents }
    pub const fn rank(self) -> u8 { self.rank }
    pub const fn max_sample_rate(self) -> Option<SampleRate> { self.max_sample_rate }
    pub const fn max_payload_bytes(self) -> u64 { self.max_payload_bytes }
}

#[cfg(test)]
#[path = "extended_capability_tests.rs"]
mod tests;
