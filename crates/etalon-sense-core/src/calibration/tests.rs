use super::*;
use core::num::NonZeroU64;
use crate::{
    adapter::AdapterError, authority::TrustedCaptureClock, evidence::EvidenceOrigin,
    observation::{HostPolicyDecision, MeasurementUncertainty, ObservationEnvelope, ObservationKind, QuantityUnitIds},
    processing::OperationId,
    quality::{CalibrationArtifactId, CalibrationRecord, CalibrationState, QualityRecord, SourceDescriptor, SourceKind},
    time::{ClockStamp, TimeInterval},
};

fn id(value: u64) -> NonZeroU64 { NonZeroU64::new(value).unwrap() }
fn semantics(unit: u64) -> QuantityUnitIds { QuantityUnitIds::new(Some(5), Some(unit)).unwrap() }
fn stamp(ticks: u64) -> ClockStamp { ClockStamp::new(40, 2, ticks, 0).unwrap() }

fn datum(
    value: Option<f64>, source: u64, unit: u64, time: Option<ClockStamp>, uncertainty: MeasurementUncertainty,
) -> CalibrationDatum {
    CalibrationDatum::new(ObservationEnvelope::new(
        ObservationKind::SyntheticFrame, SourceDescriptor::new(source, SourceKind::Synthetic).unwrap(),
        None, None, None, time, semantics(unit), uncertainty, QualityRecord::Unknown,
        CalibrationRecord::new(CalibrationState::Missing).unwrap(), EvidenceOrigin::Unknown,
        None, HostPolicyDecision::NotEvaluated,
    ).unwrap(), value).unwrap()
}

fn pair<'a>(number: u64, raw: f64, reference: f64, conditions: &'a [CalibrationCondition]) -> CalibrationObservation<'a> {
    CalibrationObservation::new(id(number),
        datum(Some(raw), 1, 6, Some(stamp(100)), MeasurementUncertainty::Unknown),
        Some(datum(Some(reference), 2, 7, Some(stamp(100)), MeasurementUncertainty::Unknown)),
        Some(id(9)), conditions,
    ).unwrap()
}

fn specification(fit: TwoPointAffineFit) -> AffineCalibrationSpecification {
    AffineCalibrationSpecification {
        artifact_id: CalibrationArtifactId::new([1; 32]).unwrap(), method: OperationId::new(12, 1).unwrap(),
        source: fit.source, input: fit.input, output: fit.output, environment_id: fit.environment_id,
        coefficients: fit.coefficients, input_range: fit.input_range,
        validity: CalibrationValidity { clock_id: id(40), restart_epoch: 2, interval: TimeInterval::new(0, 1000).unwrap() },
        model_error_bound: Some(0.5),
    }
}

struct Clock(ClockStamp);
impl TrustedCaptureClock for Clock {
    fn now(&mut self) -> Result<ClockStamp, AdapterError> { Ok(self.0) }
}

#[test]
fn recording_preserves_missing_reference_values_time_quality_and_conditions() {
    let conditions = [CalibrationCondition { id: id(20), unit_id: None, value: None }];
    let input = datum(None, 1, 6, None, MeasurementUncertainty::Unknown);
    let observation = CalibrationObservation::new(id(1), input, None, None, &conditions).unwrap();
    let mut output = [None; 2];
    assert_eq!(record_calibration_observations(&[observation], &mut output), Ok(1));
    assert_eq!(output[0], Some(observation));
    assert_eq!(output[1], None);
    assert_eq!(output[0].unwrap().measured().envelope().quality(), QualityRecord::Unknown);
    assert_eq!(output[0].unwrap().measured().envelope().timestamp(), None);
    let before = output;
    assert_eq!(record_calibration_observations(&[observation; 2], &mut output), Err(CalibrationError::DuplicateObservation));
    assert_eq!(output, before);
}

#[test]
fn two_point_fit_has_explicit_coefficients_range_source_and_environment() {
    let fit = fit_affine_two_point(pair(1, 0.0, 1.0, &[]), pair(2, 10.0, 21.0, &[]), 1.0).unwrap();
    assert_eq!(fit.coefficients, AffineCoefficients::new(2.0, 1.0).unwrap());
    assert_eq!(fit.input_range, CalibrationInterval::new(0.0, 10.0).unwrap());
    assert_eq!(fit.source.source_id(), 1);
    assert_eq!(fit.environment_id, id(9));
    assert_eq!(fit.observation_ids, [id(1), id(2)]);
}

#[test]
fn fit_rejects_degenerate_missing_mismatched_and_overflowing_data() {
    let first = pair(1, 0.0, 1.0, &[]);
    assert_eq!(fit_affine_two_point(first, pair(2, 0.0, 21.0, &[]), 0.1), Err(CalibrationError::DegenerateFit));
    assert_eq!(fit_affine_two_point(first, pair(2, 0.01, 21.0, &[]), 0.1), Err(CalibrationError::DegenerateFit));
    assert_eq!(fit_affine_two_point(first, pair(2, 10.0, 21.0, &[]), 0.0), Err(CalibrationError::InvalidRange));
    let missing = CalibrationObservation::new(id(2), first.measured(), None, Some(id(9)), &[]).unwrap();
    assert_eq!(fit_affine_two_point(first, missing, 0.1), Err(CalibrationError::MissingReference));
    let different = CalibrationObservation::new(id(2), first.measured(), first.reference(), Some(id(10)), &[]).unwrap();
    assert_eq!(fit_affine_two_point(first, different, 0.1), Err(CalibrationError::EnvironmentMismatch));
    assert_eq!(fit_affine_two_point(pair(1, -f64::MAX, 1.0, &[]), pair(2, f64::MAX, 2.0, &[]), 1.0),
        Err(CalibrationError::NumericalRange));
}

#[test]
fn application_preserves_raw_envelope_and_propagates_known_error_outward() {
    let observations = [pair(1, 0.0, 1.0, &[]), pair(2, 10.0, 21.0, &[])];
    let fit = fit_affine_two_point(observations[0], observations[1], 1.0).unwrap();
    let profile = AffineCalibrationProfile::new(specification(fit), &[], &observations).unwrap();
    let uncertainty = MeasurementUncertainty::absolute_bound(0.25, Some(6)).unwrap();
    let input = datum(Some(4.0), 1, 6, Some(stamp(100)), uncertainty);
    let mut output = None;
    apply_affine_calibration(&profile, input, Some(id(9)), &[], &mut Clock(stamp(200)), &mut output).unwrap();
    let result = output.unwrap();
    assert_eq!(result.value, 9.0);
    assert_eq!(result.original, input);
    assert_eq!(result.original.envelope().calibration().state(), CalibrationState::Missing);
    assert_eq!(result.quantity_unit, semantics(7));
    assert!(matches!(result.uncertainty, MeasurementUncertainty::Interval { lower, upper, unit_id }
        if lower <= 8.0 && upper >= 10.0 && lower > 7.99 && upper < 10.01 && unit_id == Some(id(7))));
    assert_eq!(profile.observations(), observations.as_slice());
}

#[test]
fn negative_gain_reverses_uncertainty_endpoints() {
    let observations = [pair(1, 0.0, 21.0, &[]), pair(2, 10.0, 1.0, &[])];
    let fit = fit_affine_two_point(observations[0], observations[1], 1.0).unwrap();
    let profile = AffineCalibrationProfile::new(specification(fit), &[], &observations).unwrap();
    let input = datum(Some(4.0), 1, 6, Some(stamp(100)), MeasurementUncertainty::absolute_bound(0.25, Some(6)).unwrap());
    let mut output = None;
    apply_affine_calibration(&profile, input, Some(id(9)), &[], &mut Clock(stamp(200)), &mut output).unwrap();
    assert_eq!(output.unwrap().value, 13.0);
    assert!(matches!(output.unwrap().uncertainty, MeasurementUncertainty::Interval { lower, upper, .. }
        if lower <= 12.0 && upper >= 14.0 && lower > 11.99 && upper < 14.01));
}

#[test]
fn unknown_measurement_or_model_error_never_becomes_zero_uncertainty() {
    let observations = [pair(1, 0.0, 1.0, &[]), pair(2, 10.0, 21.0, &[])];
    let fit = fit_affine_two_point(observations[0], observations[1], 1.0).unwrap();
    for unknown_model in [true, false] {
        let mut spec = specification(fit);
        if unknown_model { spec.model_error_bound = None; }
        let profile = AffineCalibrationProfile::new(spec, &[], &observations).unwrap();
        let uncertainty = if unknown_model { MeasurementUncertainty::absolute_bound(0.25, Some(6)).unwrap() }
            else { MeasurementUncertainty::Unknown };
        let input = datum(Some(4.0), 1, 6, Some(stamp(100)), uncertainty);
        let mut output = None;
        apply_affine_calibration(&profile, input, Some(id(9)), &[], &mut Clock(stamp(200)), &mut output).unwrap();
        assert_eq!(output.unwrap().uncertainty, MeasurementUncertainty::Unknown);
    }
}

#[test]
fn device_environment_units_time_and_full_input_error_range_are_checked() {
    let observations = [pair(1, 0.0, 1.0, &[]), pair(2, 10.0, 21.0, &[])];
    let profile = AffineCalibrationProfile::new(specification(fit_affine_two_point(observations[0], observations[1], 1.0).unwrap()),
        &[], &observations).unwrap();
    let input = datum(Some(4.0), 1, 6, Some(stamp(100)), MeasurementUncertainty::Unknown);
    let mut output = None;
    apply_affine_calibration(&profile, input, Some(id(9)), &[], &mut Clock(stamp(200)), &mut output).unwrap();
    for (datum, environment, clock, expected) in [
        (input, Some(id(8)), stamp(200), CalibrationError::EnvironmentMismatch),
        (datum(Some(4.0), 3, 6, Some(stamp(100)), MeasurementUncertainty::Unknown), Some(id(9)), stamp(200), CalibrationError::SourceMismatch),
        (datum(Some(4.0), 1, 8, Some(stamp(100)), MeasurementUncertainty::Unknown), Some(id(9)), stamp(200), CalibrationError::SemanticsMismatch),
        (datum(Some(11.0), 1, 6, Some(stamp(100)), MeasurementUncertainty::Unknown), Some(id(9)), stamp(200), CalibrationError::OutsideRange),
        (datum(Some(9.9), 1, 6, Some(stamp(100)), MeasurementUncertainty::absolute_bound(0.2, Some(6)).unwrap()), Some(id(9)), stamp(200), CalibrationError::OutsideRange),
        (datum(Some(4.0), 1, 6, None, MeasurementUncertainty::Unknown), Some(id(9)), stamp(200), CalibrationError::UnknownTime),
        (input, Some(id(9)), stamp(1001), CalibrationError::OutsideValidity),
        (input, Some(id(9)), ClockStamp::new(40, 3, 200, 0).unwrap(), CalibrationError::ClockMismatch),
    ] {
        assert_eq!(apply_affine_calibration(&profile, datum, environment, &[], &mut Clock(clock), &mut output), Err(expected));
        assert_eq!(output, None);
    }
}

#[test]
fn conditions_require_known_units_and_whole_intervals_inside_profile() {
    let conditions = [CalibrationCondition { id: id(20), unit_id: Some(id(21)), value: Some(CalibrationInterval::new(20.0, 22.0).unwrap()) }];
    let requirements = [CalibrationConditionRequirement { id: id(20), unit_id: id(21), allowed: CalibrationInterval::new(18.0, 25.0).unwrap() }];
    let observations = [pair(1, 0.0, 1.0, &conditions), pair(2, 10.0, 21.0, &conditions)];
    let profile = AffineCalibrationProfile::new(specification(fit_affine_two_point(observations[0], observations[1], 1.0).unwrap()),
        &requirements, &observations).unwrap();
    let input = datum(Some(4.0), 1, 6, Some(stamp(100)), MeasurementUncertainty::Unknown);
    let mut output = None;
    apply_affine_calibration(&profile, input, Some(id(9)), &conditions, &mut Clock(stamp(200)), &mut output).unwrap();
    assert_eq!(apply_affine_calibration(&profile, input, Some(id(9)), &[], &mut Clock(stamp(200)), &mut output), Err(CalibrationError::MissingCondition));
    let outside = [CalibrationCondition { value: Some(CalibrationInterval::new(24.0, 26.0).unwrap()), ..conditions[0] }];
    assert_eq!(apply_affine_calibration(&profile, input, Some(id(9)), &outside, &mut Clock(stamp(200)), &mut output), Err(CalibrationError::ConditionMismatch));
    let unknown = [CalibrationCondition { value: None, ..conditions[0] }];
    assert_eq!(apply_affine_calibration(&profile, input, Some(id(9)), &unknown, &mut Clock(stamp(200)), &mut output), Err(CalibrationError::MissingCondition));
}

#[test]
fn profile_cannot_extrapolate_beyond_anchors_or_use_duplicate_records() {
    let observations = [pair(1, 0.0, 1.0, &[]), pair(2, 10.0, 21.0, &[])];
    let mut spec = specification(fit_affine_two_point(observations[0], observations[1], 1.0).unwrap());
    spec.input_range = CalibrationInterval::new(-1.0, 11.0).unwrap();
    assert_eq!(AffineCalibrationProfile::new(spec, &[], &observations).err(), Some(CalibrationError::OutsideRange));
    assert_eq!(AffineCalibrationProfile::new(spec, &[], &[observations[0]; 2]).err(), Some(CalibrationError::DuplicateObservation));
    assert_eq!(AffineCoefficients::new(f64::INFINITY, 0.0), Err(CalibrationError::InvalidValue));
    assert_eq!(AffineCoefficients::new(f64::MAX, 0.0).unwrap().map(2.0), Err(CalibrationError::NumericalRange));
}
