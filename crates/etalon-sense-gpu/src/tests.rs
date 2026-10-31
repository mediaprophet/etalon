use super::*;

#[test]
fn request_preflight_is_bounded_and_requires_exact_buffers() {
    assert_eq!(validate_request(&[0; 8], 0.5, 4, 2, 2), Ok(2));
    assert_eq!(validate_request(&[], 0.5, 0, 0, 2), Err(GpuError::EmptyInput));
    assert_eq!(validate_request(&[0; 6], 0.5, 3, 1, 2), Err(GpuError::LengthMismatch));
    assert_eq!(validate_request(&[0; 12], 0.5, 6, 3, 2), Err(GpuError::CapacityExceeded));
    assert_eq!(validate_request(&[0; 8], f32::NAN, 4, 2, 2), Err(GpuError::InvalidScale));
    assert_eq!(validate_request(&[0; 8], 0.5, 3, 2, 2), Err(GpuError::LengthMismatch));
}

#[test]
fn parameter_block_encodes_scale_order_count_and_zero_padding() {
    let params = make_params(0.25, ByteOrder::BigEndian, 17);
    assert_eq!(&params[0..4], &0.25f32.to_le_bytes());
    assert_eq!(&params[4..8], &1u32.to_le_bytes());
    assert_eq!(&params[8..12], &17u32.to_le_bytes());
    assert_eq!(&params[12..16], &[0; 4]);
}

#[test]
fn capacities_use_checked_multiplication_and_shader_guards_tail_invocations() {
    assert_eq!(checked_bytes(4, BYTES_PER_INPUT_PAIR), Ok(16));
    assert_eq!(persistent_buffer_bytes(100), Ok(1_216));
    assert_eq!(checked_bytes(usize::MAX, 2), Err(GpuError::CapacityExceeded));
    let shader = include_str!("iq_power.wgsl");
    assert!(shader.contains("if (index >= params.count)"));
    assert!(shader.contains("params.big_endian != 0u"));
}

#[test]
fn gpu_results_reject_nan_negative_and_values_above_shared_i16_bound() {
    let zero_scale_bound = i16_iq_power_upper_bound(0.0).unwrap();
    assert_eq!(validate_power_values(&[0.0], zero_scale_bound), Ok(()));
    assert_eq!(
        validate_power_values(&[f32::from_bits(1)], zero_scale_bound),
        Err(GpuError::NumericRange)
    );
    let bound = i16_iq_power_upper_bound(0.5).unwrap();
    assert_eq!(validate_power_values(&[f32::NAN], bound), Err(GpuError::NumericRange));
    assert_eq!(validate_power_values(&[-1.0], bound), Err(GpuError::NumericRange));
    assert_eq!(validate_power_values(&[f32::MAX], bound), Err(GpuError::NumericRange));

    let mut staged = [123.0f64; 2];
    assert_eq!(
        stage_power_values(&[1.0, f32::NAN], &mut staged, bound),
        Err(GpuError::NumericRange)
    );
    assert_eq!(staged, [123.0; 2]);
}

#[test]
fn completion_slot_accepts_only_the_current_generation() {
    let completion = MapCompletion::new();
    let first = completion.begin().unwrap();
    completion.complete(first, true);
    assert_eq!(completion.result(first), Some(true));

    let second = completion.begin().unwrap();
    assert!(second > first);
    assert_eq!(completion.result(second), None);
    completion.complete(first, false);
    assert_eq!(completion.result(second), None);
    completion.complete(second, false);
    assert_eq!(completion.result(second), Some(false));
}

#[test]
fn invalidated_callback_cannot_complete_or_reuse_a_later_request() {
    let completion = MapCompletion::new();
    let cancelled = completion.begin().unwrap();
    completion.invalidate(cancelled);
    completion.complete(cancelled, true);
    assert_eq!(completion.result(cancelled), None);

    let current = completion.begin().unwrap();
    completion.complete(cancelled, true);
    assert_eq!(completion.result(current), None);
    completion.complete(current, true);
    assert_eq!(completion.result(current), Some(true));
}
