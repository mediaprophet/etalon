//! Bounded validation for cellular measurements supplied by a host.
//!
//! This module does not query a modem or operating system. Field names preserve
//! their stated units: dBm/dB values are never converted from bars, and a
//! generation value remains a host declaration rather than a verified radio
//! identity. Unknown time remains unknown.

use etalon_sense_core::time::ClockStamp;

/// Host declaration about the serving radio generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadioGeneration {
    Unknown,
    HostDeclared4G,
    HostDeclared5G,
}

/// OS/provider ordinal quality level. No physical unit is implied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrdinalBars {
    pub level: u8,
    pub levels: u8,
}

/// One host-supplied telemetry record. Values retain the units named by their
/// fields. `None` means unavailable; it is not a zero measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HostCellularTelemetry {
    pub source_id: u64,
    pub generation: RadioGeneration,
    pub rsrp_dbm: Option<f64>,
    pub rsrq_db: Option<f64>,
    pub sinr_db: Option<f64>,
    pub rssi_dbm: Option<f64>,
    pub bars: Option<OrdinalBars>,
    pub sequence: u64,
    /// `None` means the host cannot establish the number of lost records.
    pub gap_before: Option<u64>,
    pub timestamp: TelemetryTimestamp,
}

/// Timestamp in an explicitly named Etalon monotonic domain, or unavailable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TelemetryTimestamp {
    Unknown,
    Clock(ClockStamp),
}

/// Validated host record. Validation proves shape and finiteness only; it does
/// not authenticate the host, source, calibration, units, or generation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellularTelemetry(HostCellularTelemetry);

impl CellularTelemetry {
    pub const fn source_id(self) -> u64 {
        self.0.source_id
    }
    pub const fn generation(self) -> RadioGeneration {
        self.0.generation
    }
    pub const fn rsrp_dbm(self) -> Option<f64> {
        self.0.rsrp_dbm
    }
    pub const fn rsrq_db(self) -> Option<f64> {
        self.0.rsrq_db
    }
    pub const fn sinr_db(self) -> Option<f64> {
        self.0.sinr_db
    }
    pub const fn rssi_dbm(self) -> Option<f64> {
        self.0.rssi_dbm
    }
    pub const fn bars(self) -> Option<OrdinalBars> {
        self.0.bars
    }
    pub const fn sequence(self) -> u64 {
        self.0.sequence
    }
    pub const fn gap_before(self) -> Option<u64> {
        self.0.gap_before
    }
    pub const fn timestamp(self) -> TelemetryTimestamp {
        self.0.timestamp
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TelemetryError {
    OutputTooSmall { required: usize },
    InvalidSource,
    NoMeasurement,
    NonFiniteValue,
    InvalidBars,
}

/// Validate a complete host batch into caller-owned fixed storage.
///
/// Output is cleared on any validation failure; no partially validated prefix
/// is exposed. Entries after the returned count are `None`. This does not
/// assign timestamps or repair sequence gaps.
pub fn validate_batch(
    input: &[HostCellularTelemetry],
    output: &mut [Option<CellularTelemetry>],
) -> Result<usize, TelemetryError> {
    output.fill(None);
    if output.len() < input.len() {
        return Err(TelemetryError::OutputTooSmall {
            required: input.len(),
        });
    }
    for record in input {
        if let Err(error) = validate_record(*record) {
            output.fill(None);
            return Err(error);
        }
    }
    for (destination, record) in output.iter_mut().zip(input.iter().copied()) {
        *destination = Some(CellularTelemetry(record));
    }
    Ok(input.len())
}

fn validate_record(record: HostCellularTelemetry) -> Result<(), TelemetryError> {
    if record.source_id == 0 {
        return Err(TelemetryError::InvalidSource);
    }
    if record.rsrp_dbm.is_none()
        && record.rsrq_db.is_none()
        && record.sinr_db.is_none()
        && record.rssi_dbm.is_none()
        && record.bars.is_none()
    {
        return Err(TelemetryError::NoMeasurement);
    }
    for value in [
        record.rsrp_dbm,
        record.rsrq_db,
        record.sinr_db,
        record.rssi_dbm,
    ]
    .into_iter()
    .flatten()
    {
        if !value.is_finite() {
            return Err(TelemetryError::NonFiniteValue);
        }
    }
    if let Some(bars) = record.bars {
        if bars.levels == 0 || bars.level > bars.levels {
            return Err(TelemetryError::InvalidBars);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "telemetry_tests.rs"]
mod tests;
