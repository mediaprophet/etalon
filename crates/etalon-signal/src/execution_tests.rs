use super::*;

const IQ_BYTES: [u8; 4] = [2, 0, 4, 0];

fn device(value: u64) -> DeviceId {
    DeviceId::new(value).unwrap()
}

#[derive(Clone, Copy)]
enum ProviderBehavior {
    Correct,
    WrongCount,
    NonFinite,
    OutOfRange,
    PartialWriteFullCount,
    FailAfterWriting,
}

struct TestProvider {
    id: DeviceId,
    allocation_class: ExecutionAllocationClass,
    behavior: ProviderBehavior,
    calls: usize,
}

impl RegisteredSignalProvider for TestProvider {
    fn device_id(&self) -> DeviceId {
        self.id
    }

    fn execution_allocation_class(&self) -> ExecutionAllocationClass {
        self.allocation_class
    }

    fn i16_iq_power_into(
        &mut self,
        _bytes: &[u8],
        _byte_order: ByteOrder,
        _scale: f32,
        _workspace: &mut [f32],
        staged_output: &mut [f64],
    ) -> Result<usize, SignalError> {
        self.calls += 1;
        match self.behavior {
            ProviderBehavior::Correct => {
                staged_output[0] = 5.0;
                Ok(1)
            }
            ProviderBehavior::WrongCount => {
                staged_output[0] = 5.0;
                Ok(0)
            }
            ProviderBehavior::NonFinite => {
                staged_output[0] = f64::NAN;
                Ok(1)
            }
            ProviderBehavior::OutOfRange => {
                staged_output[0] = 1.0e20;
                Ok(1)
            }
            ProviderBehavior::PartialWriteFullCount => {
                staged_output[0] = 5.0;
                Ok(staged_output.len())
            }
            ProviderBehavior::FailAfterWriting => {
                staged_output[0] = 123.0;
                Err(SignalError::NumericRange)
            }
        }
    }
}

/// Deliberately omits `execution_allocation_class` to exercise the trait's
/// fail-closed `Unknown` default.
struct DefaultClassProvider(TestProvider);

impl RegisteredSignalProvider for DefaultClassProvider {
    fn device_id(&self) -> DeviceId {
        self.0.id
    }

    fn i16_iq_power_into(
        &mut self,
        bytes: &[u8],
        byte_order: ByteOrder,
        scale: f32,
        workspace: &mut [f32],
        staged_output: &mut [f64],
    ) -> Result<usize, SignalError> {
        self.0
            .i16_iq_power_into(bytes, byte_order, scale, workspace, staged_output)
    }
}

#[test]
fn cpu_route_executes_decode_and_iq_power_pipeline() {
    let mut workspace = [0.0; 2];
    let mut staged = [0.0; 1];
    let mut output = [-1.0; 1];
    let written = execute_i16_iq_power_into(
        ExecutionTarget::Cpu,
        None,
        &IQ_BYTES,
        ByteOrder::LittleEndian,
        0.5,
        &mut workspace,
        &mut staged,
        &mut output,
    )
    .unwrap();
    assert_eq!(written, 1);
    assert_eq!(workspace, [1.0, 2.0]);
    assert_eq!(output, [5.0]);
}

#[test]
fn invalid_shape_does_not_modify_public_output() {
    let mut workspace = [0.0; 1];
    let mut staged = [0.0; 1];
    let mut output = [77.0; 1];
    assert_eq!(
        execute_i16_iq_power_into(
            ExecutionTarget::Cpu,
            None,
            &IQ_BYTES,
            ByteOrder::LittleEndian,
            0.5,
            &mut workspace,
            &mut staged,
            &mut output,
        ),
        Err(ExecutionError::Capacity)
    );
    assert_eq!(output, [77.0]);
}

#[test]
fn registered_target_requires_matching_provider_and_never_falls_back() {
    let expected = device(2);
    let mut workspace = [0.0; 2];
    let mut staged = [0.0; 1];
    let mut output = [77.0; 1];
    assert_eq!(
        execute_i16_iq_power_into(
            ExecutionTarget::RegisteredDevice(expected),
            None,
            &IQ_BYTES,
            ByteOrder::LittleEndian,
            0.5,
            &mut workspace,
            &mut staged,
            &mut output,
        ),
        Err(ExecutionError::DeviceUnavailable)
    );
    assert_eq!(output, [77.0]);

    let mut provider = TestProvider {
        id: device(1),
        allocation_class: ExecutionAllocationClass::Unknown,
        behavior: ProviderBehavior::Correct,
        calls: 0,
    };
    assert_eq!(
        execute_i16_iq_power_into(
            ExecutionTarget::RegisteredDevice(expected),
            Some(&mut provider),
            &IQ_BYTES,
            ByteOrder::LittleEndian,
            0.5,
            &mut workspace,
            &mut staged,
            &mut output,
        ),
        Err(ExecutionError::DeviceIdMismatch)
    );
    assert_eq!(provider.calls, 0);
    assert_eq!(output, [77.0]);
}

#[test]
fn invalid_provider_count_and_nonfinite_result_are_rejected_transactionally() {
    for behavior in [
        ProviderBehavior::WrongCount,
        ProviderBehavior::NonFinite,
        ProviderBehavior::OutOfRange,
    ] {
        let mut provider = TestProvider {
            id: device(3),
            allocation_class: ExecutionAllocationClass::Unknown,
            behavior,
            calls: 0,
        };
        let mut workspace = [0.0; 2];
        let mut staged = [0.0; 1];
        let mut output = [77.0; 1];
        assert_eq!(
            execute_i16_iq_power_into(
                ExecutionTarget::RegisteredDevice(device(3)),
                Some(&mut provider),
                &IQ_BYTES,
                ByteOrder::LittleEndian,
                0.5,
                &mut workspace,
                &mut staged,
                &mut output,
            ),
            Err(ExecutionError::InvalidProviderResult)
        );
        assert_eq!(provider.calls, 1);
        assert_eq!(output, [77.0]);
    }
}

#[test]
fn provider_cannot_claim_full_count_after_leaving_slots_unwritten() {
    let mut provider = TestProvider {
        id: device(5),
        allocation_class: ExecutionAllocationClass::Unknown,
        behavior: ProviderBehavior::PartialWriteFullCount,
        calls: 0,
    };
    let bytes = [2, 0, 4, 0, 2, 0, 4, 0];
    let mut workspace = [0.0; 4];
    let mut staged = [0.0; 2];
    let mut output = [77.0; 2];
    assert_eq!(
        execute_i16_iq_power_into(
            ExecutionTarget::RegisteredDevice(device(5)),
            Some(&mut provider),
            &bytes,
            ByteOrder::LittleEndian,
            0.5,
            &mut workspace,
            &mut staged,
            &mut output,
        ),
        Err(ExecutionError::InvalidProviderResult)
    );
    assert_eq!(provider.calls, 1);
    assert_eq!(output, [77.0; 2]);
}

#[test]
fn provider_failure_after_partial_staging_keeps_public_output_unchanged() {
    let mut provider = TestProvider {
        id: device(4),
        allocation_class: ExecutionAllocationClass::Unknown,
        behavior: ProviderBehavior::FailAfterWriting,
        calls: 0,
    };
    let mut workspace = [0.0; 2];
    let mut staged = [0.0; 1];
    let mut output = [77.0; 1];
    assert_eq!(
        execute_i16_iq_power_into(
            ExecutionTarget::RegisteredDevice(device(4)),
            Some(&mut provider),
            &IQ_BYTES,
            ByteOrder::LittleEndian,
            0.5,
            &mut workspace,
            &mut staged,
            &mut output,
        ),
        Err(ExecutionError::Signal(SignalError::NumericRange))
    );
    assert_eq!(output, [77.0]);
}

#[test]
fn strict_admission_rejects_unknown_provider_before_dispatch() {
    let mut provider = DefaultClassProvider(TestProvider {
        id: device(8),
        allocation_class: ExecutionAllocationClass::Unknown,
        behavior: ProviderBehavior::Correct,
        calls: 0,
    });
    let mut workspace = [0.0; 2];
    let mut staged = [11.0; 1];
    let mut output = [77.0; 1];
    assert_eq!(
        execute_i16_iq_power_into_with_admission(
            ExecutionTarget::RegisteredDevice(device(8)),
            Some(&mut provider),
            &IQ_BYTES,
            ByteOrder::LittleEndian,
            0.5,
            &mut workspace,
            &mut staged,
            &mut output,
            AllocationAdmission::RequireHotZeroHeap,
        ),
        Err(ExecutionError::AllocationClassRejected)
    );
    assert_eq!(provider.0.calls, 0);
    assert_eq!(staged, [11.0]);
    assert_eq!(output, [77.0]);
}

#[test]
fn explicit_hot_provider_and_cpu_are_admitted_but_cold_provider_is_not() {
    let mut provider = TestProvider {
        id: device(9),
        allocation_class: ExecutionAllocationClass::HotZeroHeap,
        behavior: ProviderBehavior::Correct,
        calls: 0,
    };
    let mut workspace = [0.0; 2];
    let mut staged = [0.0; 1];
    let mut output = [0.0; 1];
    assert_eq!(
        execute_i16_iq_power_into_with_admission(
            ExecutionTarget::RegisteredDevice(device(9)),
            Some(&mut provider),
            &IQ_BYTES,
            ByteOrder::LittleEndian,
            0.5,
            &mut workspace,
            &mut staged,
            &mut output,
            AllocationAdmission::RequireHotZeroHeap,
        ),
        Ok(1)
    );
    assert_eq!(provider.calls, 1);

    let mut cpu_workspace = [0.0; 2];
    let mut cpu_staged = [0.0; 1];
    let mut cpu_output = [0.0; 1];
    assert_eq!(
        execute_i16_iq_power_into_with_admission(
            ExecutionTarget::Cpu,
            None,
            &IQ_BYTES,
            ByteOrder::LittleEndian,
            0.5,
            &mut cpu_workspace,
            &mut cpu_staged,
            &mut cpu_output,
            AllocationAdmission::RequireHotZeroHeap,
        ),
        Ok(1)
    );
    assert_eq!(cpu_output, [5.0]);

    provider.allocation_class = ExecutionAllocationClass::ColdBoundedBuffers;
    provider.calls = 0;
    output[0] = 77.0;
    assert_eq!(
        execute_i16_iq_power_into_with_admission(
            ExecutionTarget::RegisteredDevice(device(9)),
            Some(&mut provider),
            &IQ_BYTES,
            ByteOrder::LittleEndian,
            0.5,
            &mut workspace,
            &mut staged,
            &mut output,
            AllocationAdmission::RequireHotZeroHeap,
        ),
        Err(ExecutionError::AllocationClassRejected)
    );
    assert_eq!(provider.calls, 0);
    assert_eq!(output, [77.0]);
}

#[test]
fn bounded_buffer_admission_accepts_cold_provider_but_not_unknown() {
    let mut provider = TestProvider {
        id: device(10),
        allocation_class: ExecutionAllocationClass::ColdBoundedBuffers,
        behavior: ProviderBehavior::Correct,
        calls: 0,
    };
    let mut workspace = [0.0; 2];
    let mut staged = [0.0; 1];
    let mut output = [0.0; 1];
    assert_eq!(
        execute_i16_iq_power_into_with_admission(
            ExecutionTarget::RegisteredDevice(device(10)),
            Some(&mut provider),
            &IQ_BYTES,
            ByteOrder::LittleEndian,
            0.5,
            &mut workspace,
            &mut staged,
            &mut output,
            AllocationAdmission::AllowColdBoundedBuffers,
        ),
        Ok(1)
    );
    assert_eq!(output, [5.0]);

    let mut unknown = DefaultClassProvider(TestProvider {
        id: device(11),
        allocation_class: ExecutionAllocationClass::Unknown,
        behavior: ProviderBehavior::Correct,
        calls: 0,
    });
    output[0] = 77.0;
    assert_eq!(
        execute_i16_iq_power_into_with_admission(
            ExecutionTarget::RegisteredDevice(device(11)),
            Some(&mut unknown),
            &IQ_BYTES,
            ByteOrder::LittleEndian,
            0.5,
            &mut workspace,
            &mut staged,
            &mut output,
            AllocationAdmission::AllowColdBoundedBuffers,
        ),
        Err(ExecutionError::AllocationClassRejected)
    );
    assert_eq!(unknown.0.calls, 0);
    assert_eq!(output, [77.0]);
}

#[test]
fn power_upper_bound_covers_f32_extrema_rounding_and_rejects_invalid_scale() {
    assert_eq!(i16_iq_power_upper_bound(0.0), Ok(0.0));
    for scale in [
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        0.1,
        0.5,
        1.0,
        2.0,
        f32::MAX,
    ] {
        let upper = i16_iq_power_upper_bound(scale).unwrap();
        assert!(upper.is_finite());
        let max_i = (32768.0f64 * f64::from(scale.abs())) as f32;
        let rounded_power = 2.0 * f64::from(max_i) * f64::from(max_i);
        assert!(rounded_power <= upper, "scale={scale:?}");
        let exact_domain_power = 2.0 * (32768.0 * f64::from(scale.abs())).powi(2);
        assert!(upper >= exact_domain_power, "scale={scale:?}");
    }
    assert_eq!(
        i16_iq_power_upper_bound(f32::NAN),
        Err(SignalError::InvalidParameter)
    );
    assert_eq!(
        i16_iq_power_upper_bound(f32::INFINITY),
        Err(SignalError::InvalidParameter)
    );
}

#[test]
fn cpu_extreme_pair_stays_within_shared_upper_bound() {
    let bytes = [0, 128, 0, 128]; // I = Q = i16::MIN
    let scale = 0.1f32;
    let mut workspace = [0.0; 2];
    let mut staged = [0.0; 1];
    let mut output = [0.0; 1];
    assert_eq!(
        execute_i16_iq_power_into(
            ExecutionTarget::Cpu,
            None,
            &bytes,
            ByteOrder::LittleEndian,
            scale,
            &mut workspace,
            &mut staged,
            &mut output,
        ),
        Ok(1)
    );
    assert!(output[0] <= i16_iq_power_upper_bound(scale).unwrap());
}
