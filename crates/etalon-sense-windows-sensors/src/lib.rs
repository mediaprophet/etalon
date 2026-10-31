#![forbid(unsafe_code)]
//! Policy-gated Windows light, magnetic, pressure and inertial sensor readings.
//!
//! The OS may return a cached reading. Default device selection, timestamps,
//! coordinate frame and units are OS declarations, not authenticated source
//! identity, freshness, radiometric calibration or forensic evidence. The host
//! controls COM/WinRT initialization, source binding and policy. Calls may
//! block and revocation cannot interrupt an already executing OS call.

mod observation;
pub use observation::{adapt_reading, AdaptError, MappedReading, Mapping};
mod presence;
pub use presence::{adapt_os_presence, MappedOsPresence, OsPresenceMapping};
pub mod compass;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SensorKind {
    AmbientLight,
    Magnetometer,
    Accelerometer,
    Gyrometer,
    Barometer,
    OsHumanPresence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SensorOperation {
    SelectDefault,
    Read,
}

pub trait SensorPolicy {
    fn permit(&mut self, source_id: u64, sensor: SensorKind, operation: SensorOperation) -> bool;

    /// Hosts with value-dependent policy should override this final check.
    fn permit_reading(&mut self, _sensor: SensorKind, _reading: &Reading) -> bool {
        true
    }

    /// The OS classification is unverified. Hosts may reject it before return.
    fn permit_presence(&mut self, _reading: &OsPresenceReading) -> bool {
        true
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SensorError {
    InvalidSource,
    PolicyDenied,
    PolicyPanicked,
    Windows(i32),
    InvalidReading,
}

/// Raw native timestamp is Windows DateTime 100-ns ticks, not an Etalon clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reading {
    pub source_id: u64,
    pub value: SensorValue,
    pub windows_utc_ticks_100ns: i64,
}

/// Unverified OS classification. The raw value is not an identity, proof of
/// presence, or evidence that this sensor uses a particular technology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OsPresenceReading {
    pub source_id: u64,
    pub presence_code: i32,
    pub windows_utc_ticks_100ns: i64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SensorValue {
    /// Native `IlluminanceInLux` value. No radiometric calibration is asserted.
    IlluminanceLux(f32),
    /// Native X/Y/Z values and the OS directional-accuracy enum's raw value.
    /// Axis frame and field units must be supplied by the host's OS convention.
    MagneticField {
        x: f32,
        y: f32,
        z: f32,
        directional_accuracy: i32,
    },
    /// Native acceleration components. Unit and axes follow the OS convention.
    Acceleration { x: f64, y: f64, z: f64 },
    /// Native angular velocity components. Unit and axes follow the OS convention.
    AngularVelocity { x: f64, y: f64, z: f64 },
    /// Native station pressure in hectopascals; no altitude is inferred.
    StationPressureHectopascals(f64),
}

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::{
    AccelerationSensor, AmbientLightSensor, AngularVelocitySensor, MagneticSensor,
    OsPresenceSensor, PressureSensor,
};
