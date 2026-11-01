//! OS-reported magnetic heading, not a position or identity.
use crate::AdaptError;
#[cfg(windows)]
use crate::SensorError;
use etalon_sense_core::{
    evidence::EvidenceOrigin,
    observation::{
        HostPolicyDecision, MeasurementUncertainty, ObservationEnvelope, ObservationKind,
        QuantityUnitIds,
    },
    quality::{CalibrationRecord, CalibrationState, QualityRecord, SourceDescriptor},
};

/// A native cached reading. Magnetic heading is in OS-declared degrees; the
/// raw time is Windows 100-ns ticks and is not a synchronized sensor clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompassReading {
    pub source_id: u64,
    pub magnetic_heading_degrees: f64,
    pub windows_utc_ticks_100ns: i64,
    /// Raw OS accuracy category, absent when unavailable or not supported.
    pub heading_accuracy: Option<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompassOperation {
    SelectDefault,
    Read,
}

/// Hosts decide source binding and applicable policy for each operation and
/// for the exact completed reading. Failures and panics do not admit data.
pub trait CompassPolicy {
    fn permit_operation(&mut self, source_id: u64, operation: CompassOperation) -> bool;
    fn permit_reading(&mut self, reading: &CompassReading) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompassObservation {
    pub reading: CompassReading,
    pub envelope: ObservationEnvelope,
}

/// Host binds quantity/unit identifiers; an envelope does not authenticate the
/// native source, calibration or reading.
pub fn adapt_compass_reading(
    reading: CompassReading,
    source: SourceDescriptor,
    semantics: QuantityUnitIds,
) -> Result<CompassObservation, AdaptError> {
    if reading.source_id == 0 || reading.source_id != source.source_id() {
        return Err(AdaptError::SourceMismatch);
    }
    if !reading.magnetic_heading_degrees.is_finite() {
        return Err(AdaptError::InvalidReading);
    }
    let calibration =
        CalibrationRecord::new(CalibrationState::Missing).map_err(AdaptError::Contract)?;
    let envelope = ObservationEnvelope::new(
        ObservationKind::CachedQuery,
        source,
        None,
        None,
        None,
        None,
        semantics,
        MeasurementUncertainty::Unknown,
        QualityRecord::Unknown,
        calibration,
        EvidenceOrigin::Unknown,
        None,
        HostPolicyDecision::NotEvaluated,
    )
    .map_err(AdaptError::Contract)?;
    Ok(CompassObservation { reading, envelope })
}

#[cfg(windows)]
mod native {
    use super::*;
    use ::windows::Devices::Sensors::Compass;

    fn permit_operation(
        policy: &mut impl CompassPolicy,
        source_id: u64,
        operation: CompassOperation,
    ) -> Result<(), SensorError> {
        if source_id == 0 {
            return Err(SensorError::InvalidSource);
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            policy.permit_operation(source_id, operation)
        })) {
            Ok(true) => Ok(()),
            Ok(false) => Err(SensorError::PolicyDenied),
            Err(_) => Err(SensorError::PolicyPanicked),
        }
    }

    pub struct CompassSensor {
        source_id: u64,
        sensor: Compass,
    }
    impl CompassSensor {
        pub fn select_default(
            source_id: u64,
            policy: &mut impl CompassPolicy,
        ) -> Result<Self, SensorError> {
            permit_operation(policy, source_id, CompassOperation::SelectDefault)?;
            let sensor = Compass::GetDefault().map_err(|e| SensorError::Windows(e.code().0))?;
            permit_operation(policy, source_id, CompassOperation::SelectDefault)?;
            Ok(Self { source_id, sensor })
        }

        /// The OS may return cached data. Calls may block and cannot be
        /// interrupted when policy changes during an in-flight OS call.
        pub fn read(&self, policy: &mut impl CompassPolicy) -> Result<CompassReading, SensorError> {
            permit_operation(policy, self.source_id, CompassOperation::Read)?;
            let reading = self
                .sensor
                .GetCurrentReading()
                .map_err(|e| SensorError::Windows(e.code().0))?;
            let heading = reading
                .HeadingMagneticNorth()
                .map_err(|e| SensorError::Windows(e.code().0))?;
            if !heading.is_finite() {
                return Err(SensorError::InvalidReading);
            }
            let value = CompassReading {
                source_id: self.source_id,
                magnetic_heading_degrees: heading,
                windows_utc_ticks_100ns: reading
                    .Timestamp()
                    .map_err(|e| SensorError::Windows(e.code().0))?
                    .UniversalTime,
                heading_accuracy: reading.HeadingAccuracy().ok().map(|accuracy| accuracy.0),
            };
            permit_operation(policy, self.source_id, CompassOperation::Read)?;
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                policy.permit_reading(&value)
            })) {
                Ok(true) => {}
                Ok(false) => return Err(SensorError::PolicyDenied),
                Err(_) => return Err(SensorError::PolicyPanicked),
            }
            permit_operation(policy, self.source_id, CompassOperation::Read)?;
            Ok(value)
        }
    }
}

#[cfg(windows)]
pub use native::CompassSensor;

#[cfg(test)]
mod tests {
    use super::*;
    use etalon_sense_core::quality::SourceKind;

    #[test]
    fn preserves_heading_timestamp_and_source_binding() {
        let reading = CompassReading {
            source_id: 7,
            magnetic_heading_degrees: 17.25,
            windows_utc_ticks_100ns: -19,
            heading_accuracy: None,
        };
        let source = SourceDescriptor::new(7, SourceKind::Live).unwrap();
        let semantics = QuantityUnitIds::new(None, None).unwrap();
        let mapped = adapt_compass_reading(reading, source, semantics).unwrap();
        assert_eq!(mapped.reading, reading);
        assert_eq!(mapped.envelope.timestamp(), None);
        assert_eq!(
            adapt_compass_reading(
                reading,
                SourceDescriptor::new(8, SourceKind::Live).unwrap(),
                semantics
            ),
            Err(AdaptError::SourceMismatch)
        );
        assert_eq!(
            adapt_compass_reading(
                CompassReading {
                    magnetic_heading_degrees: f64::NAN,
                    ..reading
                },
                source,
                semantics
            ),
            Err(AdaptError::InvalidReading)
        );
    }
}
