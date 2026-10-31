use crate::{
    OsPresenceReading, Reading, SensorError, SensorKind, SensorOperation, SensorPolicy, SensorValue,
};
use ::windows::Devices::Sensors::{
    Accelerometer, Barometer, Gyrometer, HumanPresenceSensor, LightSensor, Magnetometer,
};

fn approve(
    policy: &mut impl SensorPolicy,
    id: u64,
    kind: SensorKind,
    operation: SensorOperation,
) -> Result<(), SensorError> {
    if id == 0 {
        return Err(SensorError::InvalidSource);
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        policy.permit(id, kind, operation)
    })) {
        Ok(true) => Ok(()),
        Ok(false) => Err(SensorError::PolicyDenied),
        Err(_) => Err(SensorError::PolicyPanicked),
    }
}

fn err(error: ::windows::core::Error) -> SensorError {
    SensorError::Windows(error.code().0)
}

fn approve_reading(
    policy: &mut impl SensorPolicy,
    kind: SensorKind,
    value: &Reading,
) -> Result<(), SensorError> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        policy.permit_reading(kind, value)
    })) {
        Ok(true) => Ok(()),
        Ok(false) => Err(SensorError::PolicyDenied),
        Err(_) => Err(SensorError::PolicyPanicked),
    }
}

fn approve_presence(
    policy: &mut impl SensorPolicy,
    value: &OsPresenceReading,
) -> Result<(), SensorError> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        policy.permit_presence(value)
    })) {
        Ok(true) => Ok(()),
        Ok(false) => Err(SensorError::PolicyDenied),
        Err(_) => Err(SensorError::PolicyPanicked),
    }
}

pub struct AmbientLightSensor {
    source_id: u64,
    sensor: LightSensor,
}

impl AmbientLightSensor {
    pub fn select_default(
        source_id: u64,
        policy: &mut impl SensorPolicy,
    ) -> Result<Self, SensorError> {
        approve(
            policy,
            source_id,
            SensorKind::AmbientLight,
            SensorOperation::SelectDefault,
        )?;
        let sensor = LightSensor::GetDefault().map_err(err)?;
        approve(
            policy,
            source_id,
            SensorKind::AmbientLight,
            SensorOperation::SelectDefault,
        )?;
        Ok(Self { source_id, sensor })
    }

    pub fn read(&self, policy: &mut impl SensorPolicy) -> Result<Reading, SensorError> {
        approve(
            policy,
            self.source_id,
            SensorKind::AmbientLight,
            SensorOperation::Read,
        )?;
        let reading = self.sensor.GetCurrentReading().map_err(err)?;
        let lux = reading.IlluminanceInLux().map_err(err)?;
        if !lux.is_finite() || lux < 0.0 {
            return Err(SensorError::InvalidReading);
        }
        let value = Reading {
            source_id: self.source_id,
            value: SensorValue::IlluminanceLux(lux),
            windows_utc_ticks_100ns: reading.Timestamp().map_err(err)?.UniversalTime,
        };
        approve(
            policy,
            self.source_id,
            SensorKind::AmbientLight,
            SensorOperation::Read,
        )?;
        approve_reading(policy, SensorKind::AmbientLight, &value)?;
        approve(
            policy,
            self.source_id,
            SensorKind::AmbientLight,
            SensorOperation::Read,
        )?;
        Ok(value)
    }
}

pub struct MagneticSensor {
    source_id: u64,
    sensor: Magnetometer,
}

impl MagneticSensor {
    pub fn select_default(
        source_id: u64,
        policy: &mut impl SensorPolicy,
    ) -> Result<Self, SensorError> {
        approve(
            policy,
            source_id,
            SensorKind::Magnetometer,
            SensorOperation::SelectDefault,
        )?;
        let sensor = Magnetometer::GetDefault().map_err(err)?;
        approve(
            policy,
            source_id,
            SensorKind::Magnetometer,
            SensorOperation::SelectDefault,
        )?;
        Ok(Self { source_id, sensor })
    }

    pub fn read(&self, policy: &mut impl SensorPolicy) -> Result<Reading, SensorError> {
        approve(
            policy,
            self.source_id,
            SensorKind::Magnetometer,
            SensorOperation::Read,
        )?;
        let reading = self.sensor.GetCurrentReading().map_err(err)?;
        let (x, y, z) = (
            reading.MagneticFieldX().map_err(err)?,
            reading.MagneticFieldY().map_err(err)?,
            reading.MagneticFieldZ().map_err(err)?,
        );
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(SensorError::InvalidReading);
        }
        let value = Reading {
            source_id: self.source_id,
            value: SensorValue::MagneticField {
                x,
                y,
                z,
                directional_accuracy: reading.DirectionalAccuracy().map_err(err)?.0,
            },
            windows_utc_ticks_100ns: reading.Timestamp().map_err(err)?.UniversalTime,
        };
        approve(
            policy,
            self.source_id,
            SensorKind::Magnetometer,
            SensorOperation::Read,
        )?;
        approve_reading(policy, SensorKind::Magnetometer, &value)?;
        approve(
            policy,
            self.source_id,
            SensorKind::Magnetometer,
            SensorOperation::Read,
        )?;
        Ok(value)
    }
}

pub struct AccelerationSensor {
    source_id: u64,
    sensor: Accelerometer,
}

impl AccelerationSensor {
    pub fn select_default(
        source_id: u64,
        policy: &mut impl SensorPolicy,
    ) -> Result<Self, SensorError> {
        approve(
            policy,
            source_id,
            SensorKind::Accelerometer,
            SensorOperation::SelectDefault,
        )?;
        let sensor = Accelerometer::GetDefault().map_err(err)?;
        approve(
            policy,
            source_id,
            SensorKind::Accelerometer,
            SensorOperation::SelectDefault,
        )?;
        Ok(Self { source_id, sensor })
    }

    pub fn read(&self, policy: &mut impl SensorPolicy) -> Result<Reading, SensorError> {
        approve(
            policy,
            self.source_id,
            SensorKind::Accelerometer,
            SensorOperation::Read,
        )?;
        let reading = self.sensor.GetCurrentReading().map_err(err)?;
        let (x, y, z) = (
            reading.AccelerationX().map_err(err)?,
            reading.AccelerationY().map_err(err)?,
            reading.AccelerationZ().map_err(err)?,
        );
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(SensorError::InvalidReading);
        }
        let value = Reading {
            source_id: self.source_id,
            value: SensorValue::Acceleration { x, y, z },
            windows_utc_ticks_100ns: reading.Timestamp().map_err(err)?.UniversalTime,
        };
        approve(
            policy,
            self.source_id,
            SensorKind::Accelerometer,
            SensorOperation::Read,
        )?;
        approve_reading(policy, SensorKind::Accelerometer, &value)?;
        approve(
            policy,
            self.source_id,
            SensorKind::Accelerometer,
            SensorOperation::Read,
        )?;
        Ok(value)
    }
}

pub struct AngularVelocitySensor {
    source_id: u64,
    sensor: Gyrometer,
}

impl AngularVelocitySensor {
    pub fn select_default(
        source_id: u64,
        policy: &mut impl SensorPolicy,
    ) -> Result<Self, SensorError> {
        approve(
            policy,
            source_id,
            SensorKind::Gyrometer,
            SensorOperation::SelectDefault,
        )?;
        let sensor = Gyrometer::GetDefault().map_err(err)?;
        approve(
            policy,
            source_id,
            SensorKind::Gyrometer,
            SensorOperation::SelectDefault,
        )?;
        Ok(Self { source_id, sensor })
    }

    pub fn read(&self, policy: &mut impl SensorPolicy) -> Result<Reading, SensorError> {
        approve(
            policy,
            self.source_id,
            SensorKind::Gyrometer,
            SensorOperation::Read,
        )?;
        let reading = self.sensor.GetCurrentReading().map_err(err)?;
        let (x, y, z) = (
            reading.AngularVelocityX().map_err(err)?,
            reading.AngularVelocityY().map_err(err)?,
            reading.AngularVelocityZ().map_err(err)?,
        );
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(SensorError::InvalidReading);
        }
        let value = Reading {
            source_id: self.source_id,
            value: SensorValue::AngularVelocity { x, y, z },
            windows_utc_ticks_100ns: reading.Timestamp().map_err(err)?.UniversalTime,
        };
        approve(
            policy,
            self.source_id,
            SensorKind::Gyrometer,
            SensorOperation::Read,
        )?;
        approve_reading(policy, SensorKind::Gyrometer, &value)?;
        approve(
            policy,
            self.source_id,
            SensorKind::Gyrometer,
            SensorOperation::Read,
        )?;
        Ok(value)
    }
}

pub struct PressureSensor {
    source_id: u64,
    sensor: Barometer,
}

impl PressureSensor {
    pub fn select_default(
        source_id: u64,
        policy: &mut impl SensorPolicy,
    ) -> Result<Self, SensorError> {
        approve(
            policy,
            source_id,
            SensorKind::Barometer,
            SensorOperation::SelectDefault,
        )?;
        let sensor = Barometer::GetDefault().map_err(err)?;
        approve(
            policy,
            source_id,
            SensorKind::Barometer,
            SensorOperation::SelectDefault,
        )?;
        Ok(Self { source_id, sensor })
    }

    pub fn read(&self, policy: &mut impl SensorPolicy) -> Result<Reading, SensorError> {
        approve(
            policy,
            self.source_id,
            SensorKind::Barometer,
            SensorOperation::Read,
        )?;
        let reading = self.sensor.GetCurrentReading().map_err(err)?;
        let pressure = reading.StationPressureInHectopascals().map_err(err)?;
        if !pressure.is_finite() || pressure < 0.0 {
            return Err(SensorError::InvalidReading);
        }
        let value = Reading {
            source_id: self.source_id,
            value: SensorValue::StationPressureHectopascals(pressure),
            windows_utc_ticks_100ns: reading.Timestamp().map_err(err)?.UniversalTime,
        };
        approve(
            policy,
            self.source_id,
            SensorKind::Barometer,
            SensorOperation::Read,
        )?;
        approve_reading(policy, SensorKind::Barometer, &value)?;
        approve(
            policy,
            self.source_id,
            SensorKind::Barometer,
            SensorOperation::Read,
        )?;
        Ok(value)
    }
}

/// Native cached OS classification; no person identity or sensor technology is inferred.
pub struct OsPresenceSensor {
    source_id: u64,
    sensor: HumanPresenceSensor,
}

impl OsPresenceSensor {
    pub fn select_default(
        source_id: u64,
        policy: &mut impl SensorPolicy,
    ) -> Result<Self, SensorError> {
        approve(
            policy,
            source_id,
            SensorKind::OsHumanPresence,
            SensorOperation::SelectDefault,
        )?;
        let sensor = HumanPresenceSensor::GetDefault().map_err(err)?;
        approve(
            policy,
            source_id,
            SensorKind::OsHumanPresence,
            SensorOperation::SelectDefault,
        )?;
        Ok(Self { source_id, sensor })
    }

    pub fn read(&self, policy: &mut impl SensorPolicy) -> Result<OsPresenceReading, SensorError> {
        approve(
            policy,
            self.source_id,
            SensorKind::OsHumanPresence,
            SensorOperation::Read,
        )?;
        let reading = self.sensor.GetCurrentReading().map_err(err)?;
        let value = OsPresenceReading {
            source_id: self.source_id,
            presence_code: reading.Presence().map_err(err)?.0,
            windows_utc_ticks_100ns: reading.Timestamp().map_err(err)?.UniversalTime,
        };
        approve(
            policy,
            self.source_id,
            SensorKind::OsHumanPresence,
            SensorOperation::Read,
        )?;
        approve_presence(policy, &value)?;
        approve(
            policy,
            self.source_id,
            SensorKind::OsHumanPresence,
            SensorOperation::Read,
        )?;
        Ok(value)
    }
}

#[cfg(test)]
mod policy_tests {
    use super::*;
    struct DenyResults;
    impl SensorPolicy for DenyResults {
        fn permit(&mut self, _: u64, _: SensorKind, _: SensorOperation) -> bool {
            true
        }
        fn permit_reading(&mut self, _: SensorKind, _: &Reading) -> bool {
            false
        }
        fn permit_presence(&mut self, _: &OsPresenceReading) -> bool {
            false
        }
    }

    #[test]
    fn result_specific_denials_are_not_promoted_to_permits() {
        let reading = Reading {
            source_id: 1,
            value: SensorValue::IlluminanceLux(1.0),
            windows_utc_ticks_100ns: 1,
        };
        let presence = OsPresenceReading {
            source_id: 1,
            presence_code: 0,
            windows_utc_ticks_100ns: 1,
        };
        assert_eq!(
            approve_reading(&mut DenyResults, SensorKind::AmbientLight, &reading),
            Err(SensorError::PolicyDenied)
        );
        assert_eq!(
            approve_presence(&mut DenyResults, &presence),
            Err(SensorError::PolicyDenied)
        );
    }
}
