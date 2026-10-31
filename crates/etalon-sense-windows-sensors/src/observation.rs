use etalon_sense_core::{
    evidence::EvidenceOrigin,
    observation::{
        HostPolicyDecision, MeasurementUncertainty, ObservationEnvelope, ObservationKind,
        QuantityUnitIds,
    },
    quality::{CalibrationRecord, CalibrationState, QualityRecord, SourceDescriptor},
    ContractError,
};

use crate::{Reading, SensorKind, SensorValue};

/// Semantic IDs and source are host declarations, not verified identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mapping {
    pub source: SourceDescriptor,
    /// Index 0 is illuminance, pressure, or X; indexes 1 and 2 are Y/Z.
    pub semantics: [QuantityUnitIds; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MappedReading {
    pub envelope: ObservationEnvelope,
    pub sensor_kind: SensorKind,
    pub component: u8,
    pub value: f64,
    pub windows_utc_ticks_100ns: i64,
    pub magnetic_directional_accuracy: Option<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptError {
    SourceMismatch,
    InvalidReading,
    OutputTooSmall { required: usize },
    Contract(ContractError),
}

/// Native time is retained separately; it is not silently converted into a
/// synchronized clock. Output is cleared before validation and on failure.
pub fn adapt_reading(
    reading: Reading,
    mapping: Mapping,
    output: &mut [Option<MappedReading>],
) -> Result<usize, AdaptError> {
    output.fill(None);
    if reading.source_id == 0 || reading.source_id != mapping.source.source_id() {
        return Err(AdaptError::SourceMismatch);
    }
    let (kind, values, count, accuracy) = match reading.value {
        SensorValue::IlluminanceLux(lux) if lux.is_finite() && lux >= 0.0 => (
            SensorKind::AmbientLight,
            [f64::from(lux), 0.0, 0.0],
            1,
            None,
        ),
        SensorValue::StationPressureHectopascals(value) if value.is_finite() && value >= 0.0 => {
            (SensorKind::Barometer, [value, 0.0, 0.0], 1, None)
        }
        SensorValue::MagneticField {
            x,
            y,
            z,
            directional_accuracy,
        } if x.is_finite() && y.is_finite() && z.is_finite() => (
            SensorKind::Magnetometer,
            [f64::from(x), f64::from(y), f64::from(z)],
            3,
            Some(directional_accuracy),
        ),
        SensorValue::Acceleration { x, y, z }
            if x.is_finite() && y.is_finite() && z.is_finite() =>
        {
            (SensorKind::Accelerometer, [x, y, z], 3, None)
        }
        SensorValue::AngularVelocity { x, y, z }
            if x.is_finite() && y.is_finite() && z.is_finite() =>
        {
            (SensorKind::Gyrometer, [x, y, z], 3, None)
        }
        _ => return Err(AdaptError::InvalidReading),
    };
    if output.len() < count {
        return Err(AdaptError::OutputTooSmall { required: count });
    }
    let calibration =
        CalibrationRecord::new(CalibrationState::Missing).map_err(AdaptError::Contract)?;
    for index in 0..count {
        let envelope = match ObservationEnvelope::new(
            ObservationKind::CachedQuery,
            mapping.source,
            None,
            None,
            None,
            None,
            mapping.semantics[index],
            MeasurementUncertainty::Unknown,
            QualityRecord::Unknown,
            calibration,
            EvidenceOrigin::Unknown,
            None,
            HostPolicyDecision::NotEvaluated,
        ) {
            Ok(envelope) => envelope,
            Err(error) => {
                output.fill(None);
                return Err(AdaptError::Contract(error));
            }
        };
        output[index] = Some(MappedReading {
            envelope,
            sensor_kind: kind,
            component: index as u8,
            value: values[index],
            windows_utc_ticks_100ns: reading.windows_utc_ticks_100ns,
            magnetic_directional_accuracy: accuracy,
        });
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use etalon_sense_core::quality::SourceKind;

    fn mapping() -> Mapping {
        Mapping {
            source: SourceDescriptor::new(1, SourceKind::Live).unwrap(),
            semantics: [QuantityUnitIds::new(None, None).unwrap(); 3],
        }
    }

    #[test]
    fn magnetic_components_retain_native_timestamp_and_accuracy() {
        let reading = Reading {
            source_id: 1,
            windows_utc_ticks_100ns: 42,
            value: SensorValue::MagneticField {
                x: -1.0,
                y: 2.0,
                z: 3.0,
                directional_accuracy: 99,
            },
        };
        let mut output = [None; 3];
        assert_eq!(adapt_reading(reading, mapping(), &mut output), Ok(3));
        assert_eq!(output[0].unwrap().value, -1.0);
        assert_eq!(output[1].unwrap().component, 1);
        assert_eq!(output[2].unwrap().windows_utc_ticks_100ns, 42);
        assert_eq!(output[2].unwrap().magnetic_directional_accuracy, Some(99));
    }

    #[test]
    fn rejects_invalid_value_and_mismatched_source() {
        let mut reading = Reading {
            source_id: 1,
            windows_utc_ticks_100ns: 1,
            value: SensorValue::IlluminanceLux(-1.0),
        };
        let mut output = [None; 3];
        assert_eq!(
            adapt_reading(reading, mapping(), &mut output),
            Err(AdaptError::InvalidReading)
        );
        reading.source_id = 2;
        assert_eq!(
            adapt_reading(reading, mapping(), &mut output),
            Err(AdaptError::SourceMismatch)
        );
    }

    #[test]
    fn short_output_cleared_before_return() {
        let mut output = [None; 1];
        let reading = Reading {
            source_id: 1,
            windows_utc_ticks_100ns: 0,
            value: SensorValue::MagneticField {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                directional_accuracy: 0,
            },
        };
        assert_eq!(
            adapt_reading(reading, mapping(), &mut output),
            Err(AdaptError::OutputTooSmall { required: 3 })
        );
        assert_eq!(output, [None]);
    }

    #[test]
    fn inertial_components_keep_f64_precision_and_native_timestamp() {
        let reading = Reading {
            source_id: 1,
            windows_utc_ticks_100ns: 37,
            value: SensorValue::AngularVelocity {
                x: 1.0 + f64::EPSILON,
                y: -2.0,
                z: 3.0,
            },
        };
        let mut output = [None; 3];
        assert_eq!(adapt_reading(reading, mapping(), &mut output), Ok(3));
        assert_eq!(
            output[0].unwrap().value.to_bits(),
            (1.0 + f64::EPSILON).to_bits()
        );
        assert_eq!(output[1].unwrap().windows_utc_ticks_100ns, 37);
        assert_eq!(output[2].unwrap().magnetic_directional_accuracy, None);
        assert_eq!(output[2].unwrap().sensor_kind, SensorKind::Gyrometer);
    }

    #[test]
    fn pressure_preserves_precision_and_rejects_nonfinite() {
        let mut reading = Reading {
            source_id: 1,
            windows_utc_ticks_100ns: 37,
            value: SensorValue::StationPressureHectopascals(1.0 + f64::EPSILON),
        };
        let mut output = [None; 1];
        assert_eq!(adapt_reading(reading, mapping(), &mut output), Ok(1));
        assert_eq!(output[0].unwrap().sensor_kind, SensorKind::Barometer);
        assert_eq!(
            output[0].unwrap().value.to_bits(),
            (1.0 + f64::EPSILON).to_bits()
        );
        reading.value = SensorValue::StationPressureHectopascals(f64::NAN);
        assert_eq!(
            adapt_reading(reading, mapping(), &mut output),
            Err(AdaptError::InvalidReading)
        );
        assert_eq!(output, [None]);
    }
}
