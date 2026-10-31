use super::*;

#[test]
fn converts_explicit_linear_and_offset_units() {
    assert_eq!(
        convert(
            2.0,
            PhysicalQuantity::Frequency,
            Unit::Gigahertz,
            Unit::Megahertz
        ),
        Ok(2000.0)
    );
    assert_eq!(
        convert(
            0.0,
            PhysicalQuantity::Temperature,
            Unit::Celsius,
            Unit::Kelvin
        ),
        Ok(273.15)
    );
    let speed = convert(
        3.6,
        PhysicalQuantity::Velocity,
        Unit::KilometerPerHour,
        Unit::MeterPerSecond,
    )
    .unwrap();
    assert!((speed - 1.0).abs() < 1e-12);
}

#[test]
fn rejects_invalid_or_model_dependent_conversion() {
    assert_eq!(
        convert(
            1.0,
            PhysicalQuantity::MagneticField,
            Unit::Tesla,
            Unit::AmperePerMeter
        ),
        Err(ContractError::UnsupportedOperation)
    );
    assert_eq!(
        convert(
            1.0,
            PhysicalQuantity::Power,
            Unit::DecibelMilliwatt,
            Unit::Watt
        ),
        Err(ContractError::UnsupportedOperation)
    );
    assert_eq!(
        convert(
            f64::NAN,
            PhysicalQuantity::Frequency,
            Unit::Hertz,
            Unit::Hertz
        ),
        Err(ContractError::InvalidValue)
    );
    assert_eq!(
        convert(
            f64::MAX,
            PhysicalQuantity::Frequency,
            Unit::Gigahertz,
            Unit::Hertz
        ),
        Err(ContractError::ArithmeticOverflow)
    );
}
