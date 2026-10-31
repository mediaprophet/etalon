use super::*;

#[test]
fn native_kernel_handles_extreme_signed_iq_exactly() {
    let mut result = [0; 16];
    write_power(&[0, 128, 0, 128, 255, 127, 255, 255], &mut result);
    assert_eq!(
        f64::from_le_bytes(result[..8].try_into().unwrap()),
        2_147_483_648.0
    );
    assert_eq!(
        f64::from_le_bytes(result[8..].try_into().unwrap()),
        1_073_676_290.0
    );
}

#[cfg(not(target_arch = "hexagon"))]
#[test]
fn host_cannot_construct_a_native_dsp_provider() {
    assert!(matches!(
        HexagonIqPowerProvider::new(1, 1, 16),
        Err(HexagonError::UnsupportedPlatform)
    ));
}

#[cfg(target_arch = "hexagon")]
#[test]
fn native_dsp_uses_exact_target_without_cpu_fallback() {
    use etalon_sense_core::{
        processing::{ProcessingRequest, ProcessorPreference},
        processing_runtime::DispatchConstraints,
    };
    use etalon_sense_dsp::dispatch_dsp;
    let mut provider = HexagonIqPowerProvider::new(7, 1, 1).unwrap();
    let target = DspTarget::new(1, 2, 7).unwrap();
    let mut bindings = [provider.bind(target).unwrap()];
    let request = ProcessingRequest::new(
        OperationId::new(IQ_POWER_OPERATION_ID, IQ_POWER_REVISION).unwrap(),
        ProcessorPreference::Require(ProcessorKind::Dsp),
        0,
        None,
    )
    .unwrap();
    let input =
        SampleLayout::new(1, [1, 0, 0, 0], [4, 0, 0, 0], SampleEncoding::ComplexI16).unwrap();
    let output = SampleLayout::new(1, [1, 0, 0, 0], [8, 0, 0, 0], SampleEncoding::F64).unwrap();
    let mut bytes = [0; 8];
    let receipt = dispatch_dsp(
        target,
        request,
        DispatchConstraints::new(AllocationClass::HotZeroHeap, 0),
        &mut bindings,
        &input,
        &[3, 0, 4, 0],
        &output,
        &mut bytes,
        &mut [],
    )
    .unwrap();
    assert_eq!(receipt.target, target);
    assert_eq!(receipt.dispatch.processor, ProcessorKind::Dsp);
    assert_eq!(f64::from_le_bytes(bytes), 25.0);
}
