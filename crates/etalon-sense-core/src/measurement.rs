//! Bounded metadata for a sample batch; payload storage stays caller-owned.
//!
//! Validation covers the declared shape and region, not device identity,
//! authorization, payload contents, or the truth of calibration claims.
use crate::{
    buffer::{SampleLayout, SampleRegion},
    capability::SensorCapability,
    extended_capability::ExtendedCapability,
    quality::{CalibrationRecord, QualityRecord, SourceDescriptor},
    quantity::SampleRate,
    time::ClockStamp,
    ContractError,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    capability: SensorCapability,
    layout: SampleLayout,
    region: SampleRegion,
    source: SourceDescriptor,
    timestamp: Option<ClockStamp>,
    sample_rate: Option<SampleRate>,
    calibration: CalibrationRecord,
    quality: QualityRecord,
}

/// Measurement declaration for a host-registered modality or quantity outside
/// the built-in catalog. Numeric registry IDs and unit semantics are preserved
/// but are not authenticated or interpreted by this crate. Unknown unit stays
/// distinct from a dimensionless unit.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtendedMeasurement {
    capability: ExtendedCapability,
    layout: SampleLayout,
    region: SampleRegion,
    source: SourceDescriptor,
    timestamp: Option<ClockStamp>,
    sample_rate: Option<SampleRate>,
    calibration: CalibrationRecord,
    quality: QualityRecord,
}

impl Measurement {
    /// Validate the batch against its declared capability and buffer region.
    /// An unknown rate remains unknown even when the capability has a limit.
    /// The capability's rank is fixed, while each axis extent is an upper bound.
    pub fn new(
        capability: SensorCapability,
        layout: SampleLayout,
        region: SampleRegion,
        source: SourceDescriptor,
        timestamp: ClockStamp,
        sample_rate: Option<SampleRate>,
        calibration: CalibrationRecord,
        quality: QualityRecord,
    ) -> Result<Self, ContractError> {
        Self::new_with_optional_timestamp(
            capability,
            layout,
            region,
            source,
            Some(timestamp),
            sample_rate,
            calibration,
            quality,
        )
    }

    pub fn new_with_optional_timestamp(
        capability: SensorCapability,
        layout: SampleLayout,
        region: SampleRegion,
        source: SourceDescriptor,
        timestamp: Option<ClockStamp>,
        sample_rate: Option<SampleRate>,
        calibration: CalibrationRecord,
        quality: QualityRecord,
    ) -> Result<Self, ContractError> {
        if *layout.encoding() != capability.encoding() {
            return Err(ContractError::IncompatibleEncoding);
        }
        if layout.rank() != capability.max_rank() {
            return Err(ContractError::InvalidExtent);
        }
        let maximum = capability.max_extents();
        for (extent, limit) in layout.extents().iter().zip(maximum.iter()) {
            if extent > limit {
                return Err(ContractError::InvalidExtent);
            }
        }
        if let (Some(actual), Some(max)) = (sample_rate, capability.max_sample_rate()) {
            if actual.hz() > max.hz() {
                return Err(ContractError::InvalidRate);
            }
        }
        region.validate_layout(&layout)?;
        Ok(Self {
            capability,
            layout,
            region,
            source,
            timestamp,
            sample_rate,
            calibration,
            quality,
        })
    }

    pub const fn capability(&self) -> &SensorCapability {
        &self.capability
    }
    pub const fn layout(&self) -> &SampleLayout {
        &self.layout
    }
    pub const fn region(&self) -> &SampleRegion {
        &self.region
    }
    pub const fn source(&self) -> SourceDescriptor {
        self.source
    }
    pub const fn timestamp(&self) -> Option<ClockStamp> {
        self.timestamp
    }
    pub const fn sample_rate(&self) -> Option<SampleRate> {
        self.sample_rate
    }
    pub const fn calibration(&self) -> CalibrationRecord {
        self.calibration
    }
    pub const fn quality(&self) -> QualityRecord {
        self.quality
    }
}

impl ExtendedMeasurement {
    /// Validate arbitrary registered IDs against the declared byte layout and
    /// region. Registry membership, dimensional compatibility, authorization,
    /// and source identity remain the host's responsibility.
    pub fn new(
        capability: ExtendedCapability,
        layout: SampleLayout,
        region: SampleRegion,
        source: SourceDescriptor,
        timestamp: ClockStamp,
        sample_rate: Option<SampleRate>,
        calibration: CalibrationRecord,
        quality: QualityRecord,
    ) -> Result<Self, ContractError> {
        Self::new_with_optional_timestamp(
            capability,
            layout,
            region,
            source,
            Some(timestamp),
            sample_rate,
            calibration,
            quality,
        )
    }

    pub fn new_with_optional_timestamp(
        capability: ExtendedCapability,
        layout: SampleLayout,
        region: SampleRegion,
        source: SourceDescriptor,
        timestamp: Option<ClockStamp>,
        sample_rate: Option<SampleRate>,
        calibration: CalibrationRecord,
        quality: QualityRecord,
    ) -> Result<Self, ContractError> {
        capability.validate_layout(&layout, sample_rate)?;
        region.validate_layout(&layout)?;
        Ok(Self {
            capability,
            layout,
            region,
            source,
            timestamp,
            sample_rate,
            calibration,
            quality,
        })
    }

    pub const fn capability(&self) -> &ExtendedCapability {
        &self.capability
    }
    pub const fn layout(&self) -> &SampleLayout {
        &self.layout
    }
    pub const fn region(&self) -> &SampleRegion {
        &self.region
    }
    pub const fn source(&self) -> SourceDescriptor {
        self.source
    }
    pub const fn timestamp(&self) -> Option<ClockStamp> {
        self.timestamp
    }
    pub const fn sample_rate(&self) -> Option<SampleRate> {
        self.sample_rate
    }
    pub const fn calibration(&self) -> CalibrationRecord {
        self.calibration
    }
    pub const fn quality(&self) -> QualityRecord {
        self.quality
    }
}

#[cfg(test)]
#[path = "measurement_tests.rs"]
mod tests;
