use super::*;

#[test]
fn nonfinite_values_never_pass() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for (expected, actual) in [(value, 1.0), (1.0, value), (value, value)] {
            let report = compare_f32(&[expected], &[actual], OracleTolerance::default()).unwrap();
            assert!(!report.passed());
            assert_eq!(report.mismatch_count, 1);
            assert_eq!(report.first_mismatch.unwrap().0, 0);
            assert!(report.max_absolute_error.is_infinite());
        }
    }
}

#[test]
fn invalid_tolerances_are_errors_even_for_empty_inputs() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1] {
        for tolerance in [
            OracleTolerance {
                absolute: invalid,
                relative: 0.0,
            },
            OracleTolerance {
                absolute: 0.0,
                relative: invalid,
            },
        ] {
            assert!(matches!(
                compare_f32(&[1.0], &[1.0], tolerance),
                Err(ForgeError::OracleMismatch(_))
            ));
            assert!(matches!(
                compare_f32(&[], &[], tolerance),
                Err(ForgeError::OracleMismatch(_))
            ));
        }
    }
}

#[test]
fn finite_absolute_and_relative_boundaries_are_preserved() {
    let exact = OracleTolerance {
        absolute: 0.0,
        relative: 0.0,
    };
    assert!(compare_f32(&[], &[], exact).unwrap().passed());
    assert!(compare_f32(&[0.0, 1.0], &[0.0, 1.0], exact)
        .unwrap()
        .passed());
    let absolute = OracleTolerance {
        absolute: 0.25,
        relative: 0.0,
    };
    assert!(compare_f32(&[0.0], &[0.25], absolute).unwrap().passed());
    assert!(!compare_f32(&[0.0], &[0.5], absolute).unwrap().passed());
    let relative = OracleTolerance {
        absolute: 0.0,
        relative: 0.25,
    };
    assert!(compare_f32(&[4.0], &[5.0], relative).unwrap().passed());
    assert!(!compare_f32(&[4.0], &[6.0], relative).unwrap().passed());
    assert!(compare_f32(&[1.0], &[], exact).is_err());
}
