use etalon_sense_core::time::ClockStamp;

use super::*;

fn record() -> HostCellularTelemetry {
    HostCellularTelemetry {
        source_id: 5,
        generation: RadioGeneration::HostDeclared5G,
        rsrp_dbm: Some(-91.5),
        rsrq_db: Some(-11.0),
        sinr_db: Some(17.25),
        rssi_dbm: Some(-70.0),
        bars: Some(OrdinalBars { level: 3, levels: 5 }),
        sequence: 12,
        gap_before: Some(0),
        timestamp: TelemetryTimestamp::Clock(ClockStamp::new(7, 2, 100, 4).unwrap()),
    }
}

#[test]
fn validates_host_values_without_converting_units_or_claiming_generation() {
    let input = [record()];
    let mut output = [None; 2];
    assert_eq!(validate_batch(&input, &mut output), Ok(1));
    let value = output[0].unwrap();
    assert_eq!(value.rsrp_dbm(), Some(-91.5));
    assert_eq!(value.rsrq_db(), Some(-11.0));
    assert_eq!(value.sinr_db(), Some(17.25));
    assert_eq!(value.rssi_dbm(), Some(-70.0));
    assert_eq!(value.bars(), Some(OrdinalBars { level: 3, levels: 5 }));
    assert_eq!(value.generation(), RadioGeneration::HostDeclared5G);
    assert_eq!(output[1], None);
}

#[test]
fn rejects_nonfinite_metric_and_clears_the_entire_output() {
    let mut invalid = record();
    invalid.sinr_db = Some(f64::NAN);
    let mut output = [Some(CellularTelemetry(record())); 2];
    assert_eq!(validate_batch(&[record(), invalid], &mut output), Err(TelemetryError::NonFiniteValue));
    assert_eq!(output, [None, None]);
}

#[test]
fn rejects_invalid_source_and_ordinal_bars() {
    let mut invalid = record();
    invalid.source_id = 0;
    assert_eq!(validate_batch(&[invalid], &mut [None]), Err(TelemetryError::InvalidSource));
    let mut invalid = record();
    invalid.bars = Some(OrdinalBars { level: 6, levels: 5 });
    assert_eq!(validate_batch(&[invalid], &mut [None]), Err(TelemetryError::InvalidBars));
}

#[test]
fn output_capacity_failure_clears_supplied_storage_and_does_not_truncate() {
    let mut output = [Some(CellularTelemetry(record())); 1];
    assert_eq!(validate_batch(&[record(), record()], &mut output), Err(TelemetryError::OutputTooSmall { required: 2 }));
    assert_eq!(output, [None]);
}

#[test]
fn unknown_generation_and_timestamp_stay_unknown() {
    let mut value = record();
    value.generation = RadioGeneration::Unknown;
    value.timestamp = TelemetryTimestamp::Unknown;
    value.bars = None;
    let mut output = [None];
    assert_eq!(validate_batch(&[value], &mut output), Ok(1));
    assert_eq!(output[0].unwrap().generation(), RadioGeneration::Unknown);
    assert_eq!(output[0].unwrap().timestamp(), TelemetryTimestamp::Unknown);
}
