use super::*;
use etalon_sense_core::{
    processing::{ProcessingRequest, ProcessorPreference},
    processing_runtime::{dispatch_live, DispatchConstraints},
};

fn layout(count: u32, stride: u64, encoding: SampleEncoding) -> SampleLayout {
    SampleLayout::new(1, [count, 0, 0, 0], [stride, 0, 0, 0], encoding).unwrap()
}

fn run(
    provider: &mut CpuIqPowerProvider,
    input_layout: &SampleLayout,
    input: &[u8],
    output_layout: &SampleLayout,
    output: &mut [u8],
) -> Result<(), ContractError> {
    let request = ProcessingRequest::new(
        OperationId::new(IQ_POWER_OPERATION_ID, IQ_POWER_REVISION).unwrap(),
        ProcessorPreference::Require(ProcessorKind::Cpu),
        0,
        None,
    )
    .unwrap();
    let mut providers: [&mut dyn LiveProcessingProvider; 1] = [provider];
    dispatch_live(
        request,
        DispatchConstraints::new(AllocationClass::HotZeroHeap, 0),
        &mut providers,
        input_layout,
        input,
        output_layout,
        output,
        &mut [],
    )?;
    Ok(())
}

#[test]
fn extreme_complex_values_have_exact_power() {
    let mut provider = CpuIqPowerProvider::new(1, 1, 2).unwrap();
    let input = [0x00, 0x80, 0x00, 0x80, 0xff, 0x7f, 0xff, 0xff];
    let mut output = [0; 16];
    run(
        &mut provider,
        &layout(2, 4, SampleEncoding::ComplexI16),
        &input,
        &layout(2, 8, SampleEncoding::F64),
        &mut output,
    )
    .unwrap();
    assert_eq!(
        f64::from_le_bytes(output[..8].try_into().unwrap()),
        2_147_483_648.0
    );
    assert_eq!(
        f64::from_le_bytes(output[8..].try_into().unwrap()),
        1_073_676_290.0
    );
}

#[test]
fn padded_input_is_rejected_and_output_cleared() {
    let mut provider = CpuIqPowerProvider::new(1, 1, 2).unwrap();
    let mut output = [0xff; 16];
    let err = run(
        &mut provider,
        &layout(2, 8, SampleEncoding::ComplexI16),
        &[0; 12],
        &layout(2, 8, SampleEncoding::F64),
        &mut output,
    )
    .unwrap_err();
    assert_eq!(err, ContractError::IncompatibleEncoding);
    assert_eq!(output, [0; 16]);
}

#[test]
fn stopped_provider_is_not_selected() {
    let mut provider = CpuIqPowerProvider::new(1, 1, 1).unwrap();
    provider.stop();
    let mut output = [0xff; 8];
    let err = run(
        &mut provider,
        &layout(1, 4, SampleEncoding::ComplexI16),
        &[0; 4],
        &layout(1, 8, SampleEncoding::F64),
        &mut output,
    )
    .unwrap_err();
    assert_eq!(err, ContractError::UnsupportedOperation);
    assert_eq!(output, [0; 8]);
}

#[test]
fn declared_extent_limit_prevents_dispatch() {
    assert!(matches!(
        CpuIqPowerProvider::new(1, 1, 0),
        Err(ContractError::InvalidExtent)
    ));
    assert!(matches!(
        CpuIqPowerProvider::new(1, 1, MAX_PAIRS + 1),
        Err(ContractError::InvalidExtent)
    ));
    let mut provider = CpuIqPowerProvider::new(1, 1, 1).unwrap();
    let mut output = [0xff; 16];
    let err = run(
        &mut provider,
        &layout(2, 4, SampleEncoding::ComplexI16),
        &[0; 8],
        &layout(2, 8, SampleEncoding::F64),
        &mut output,
    )
    .unwrap_err();
    assert_eq!(err, ContractError::UnsupportedOperation);
    assert_eq!(output, [0; 16]);
}
