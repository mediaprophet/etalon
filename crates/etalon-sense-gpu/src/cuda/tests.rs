use super::*;

#[test]
fn cuda_iq_admission_rejects_hot_path_and_insufficient_budget() {
    assert!(matches!(
        admit_prepared(1, GpuAllocationClass::HotZeroHeap, u64::MAX),
        Err(PreparedCudaPowerError::HotZeroHeapUnsupported)
    ));
    assert!(matches!(
        admit_prepared(65, GpuAllocationClass::ColdBoundedBuffers, 519),
        Err(PreparedCudaPowerError::BudgetExceeded)
    ));
    assert!(admit_prepared(65, GpuAllocationClass::ColdBoundedBuffers, 520).is_ok());
    assert!(admit_prepared(
        crate::MAX_IQ_PAIRS + 1,
        GpuAllocationClass::ColdBoundedBuffers,
        u64::MAX
    )
    .is_err());
}

#[test]
#[ignore = "requires NVIDIA CUDA and nvcc"]
fn cuda_signed_iq_provider_matches_signal_reference() {
    let ordinal = std::env::var("ETALON_CUDA_DEVICE")
        .map(|value| value.parse().unwrap())
        .unwrap_or(0);
    let context = CudaContext::new(ordinal).unwrap();
    eprintln!("CUDA sensing qualification: {:?}", context.device_info());
    let id = DeviceId::new(42).unwrap();
    let mut provider = PreparedCudaIqPower::new(context, id, 65, 520).unwrap();
    assert_eq!(provider.device_id(), id);
    assert_eq!(
        provider.execution_allocation_class(),
        ExecutionAllocationClass::ColdBoundedBuffers
    );
    for order in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
        for scale in [0.0, 1.0, -0.5, 1.0 / 32768.0] {
            for count in [65, 1, 64, 3] {
                let mut input = Vec::new();
                for index in 0..count {
                    let (i, q) = if index % 2 == 0 {
                        (i16::MIN, i16::MAX)
                    } else {
                        (-3, 4)
                    };
                    for value in [i, q] {
                        input.extend_from_slice(&match order {
                            ByteOrder::LittleEndian => value.to_le_bytes(),
                            ByteOrder::BigEndian => value.to_be_bytes(),
                        });
                    }
                }
                let mut workspace = vec![0.0; count * 2];
                let mut reference = vec![0.0; count];
                etalon_signal::pipeline::i16_iq_power_into(
                    &input,
                    order,
                    scale,
                    &mut workspace,
                    &mut reference,
                )
                .unwrap();
                let mut output = vec![f64::NAN; count];
                assert_eq!(
                    provider
                        .compute_into(&input, order, scale, &mut workspace, &mut output)
                        .unwrap(),
                    count
                );
                for (actual, expected) in output.iter().zip(reference) {
                    let tolerance = expected.abs() * 4.0 * f64::from(f32::EPSILON)
                        + 4.0 * f64::from(f32::from_bits(1));
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{actual} != {expected}"
                    );
                }
                assert!(workspace.iter().all(|value| *value == 0.0));
            }
        }
    }
    let completed = provider.completed_runs();
    let mut output = [99.0];
    let mut workspace = [99.0; 2];
    assert!(provider
        .compute_into(
            &[0; 4],
            ByteOrder::LittleEndian,
            f32::NAN,
            &mut workspace,
            &mut output
        )
        .is_err());
    assert_eq!(output, [0.0]);
    assert_eq!(workspace, [0.0; 2]);
    assert_eq!(provider.completed_runs(), completed);
    provider
        .compute_into(
            &[0; 4],
            ByteOrder::LittleEndian,
            1.0,
            &mut workspace,
            &mut output,
        )
        .unwrap();
    assert_eq!(output, [0.0]);
    assert!(!provider.is_poisoned());
}
