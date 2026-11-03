use super::*;

#[test]
fn tensor_dimensions_reject_mismatch_zero_and_overflow() {
    assert!(validate_matrix(6, 3, 2).is_ok());
    assert!(validate_matrix(5, 3, 2).is_err());
    assert!(validate_matrix(0, 0, 2).is_err());
    assert!(validate_matrix(0, 2, 0).is_err());
    assert!(matrix_elements(usize::MAX, 2).is_err());
    #[cfg(target_pointer_width = "64")]
    {
        assert!(matrix_elements(65_536, 65_536).is_err());
        assert!(checked_elements(u32::MAX as usize + 1).is_err());
    }
}

fn values(length: usize, seed: usize) -> Vec<f32> {
    (0..length)
        .map(|index| (((index * 7 + seed) % 31) as f32 - 15.0) / 32.0)
        .collect()
}

fn gemv_reference(matrix: &[f32], vector: &[f32]) -> Vec<f32> {
    matrix
        .chunks_exact(vector.len())
        .map(|row| row.iter().zip(vector).map(|(a, b)| a * b).sum())
        .collect()
}

fn gemm_reference(
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
) -> Vec<f32> {
    (0..rows * columns)
        .map(|index| {
            let row = index / columns;
            let column = index % columns;
            (0..inner)
                .map(|k| left[row * inner + k] * right[k * columns + column])
                .sum()
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-5 + expected.abs() * 2e-5,
            "element {index}: actual {actual}, expected {expected}"
        );
    }
}

#[test]
#[ignore = "requires nvcc and NVIDIA hardware; select ordinal with ETALON_CUDA_DEVICE"]
fn cuda_resident_tensor_kernels_match_cpu_and_reuse_weights() {
    let ordinal = std::env::var("ETALON_CUDA_DEVICE")
        .map(|value| {
            value
                .parse::<i32>()
                .expect("CUDA ordinal must be an integer")
        })
        .unwrap_or(0);
    let executor = CudaTensorExecutor::new(CudaContext::new(ordinal).unwrap()).unwrap();
    eprintln!("CUDA tensor qualification: {:?}", executor.device_info());

    let matrix = values(259 * 17, 3);
    let mut gemv = executor.prepare_gemv(&matrix, 259, 17).unwrap();
    let right = values(7 * 19, 5);
    let mut gemm = executor.prepare_gemm(&right, 17, 7, 19).unwrap();
    let w1 = values(13 * 11, 2);
    let w2 = values(263 * 13, 9);
    let mut ffn = executor.prepare_fused_ffn(&w1, &w2, 11, 13, 263).unwrap();
    assert!(executor
        .prepare_gemv(&matrix[..matrix.len() - 1], 259, 17)
        .is_err());
    assert!(executor.prepare_gemm(&right, 0, 7, 19).is_err());
    assert!(executor
        .prepare_fused_ffn(&w1, &w2[..w2.len() - 1], 11, 13, 263)
        .is_err());
    drop(executor);

    for seed in [1, 7, 19] {
        let vector = values(17, seed);
        let mut output = vec![f32::NAN; gemv.output_len()];
        gemv.execute(&vector, &mut output).unwrap();
        assert_close(&output, &gemv_reference(&matrix, &vector));
        assert!(gemv.execute(&vector[..16], &mut output).is_err());
        assert!(output.iter().all(|&value| value == 0.0));
        gemv.execute(&vector, &mut output).unwrap();
        assert_close(&output, &gemv_reference(&matrix, &vector));

        let left = values(17 * 7, seed);
        let mut output = vec![f32::NAN; gemm.output_len()];
        gemm.execute(&left, &mut output).unwrap();
        assert_close(&output, &gemm_reference(&left, &right, 17, 7, 19));
        let last = output.len() - 1;
        assert!(gemm.execute(&left, &mut output[..last]).is_err());
        assert!(output[..last].iter().all(|&value| value == 0.0));

        let input = values(11, seed);
        let hidden: Vec<f32> = gemv_reference(&w1, &input)
            .into_iter()
            .map(|value| {
                0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value.powi(3))).tanh())
            })
            .collect();
        let expected = gemv_reference(&w2, &hidden);
        let mut output = vec![f32::NAN; ffn.output_len()];
        ffn.execute(&input, &mut output).unwrap();
        assert_close(&output, &expected);
        assert!(ffn.execute(&input[..10], &mut output).is_err());
        assert!(output.iter().all(|&value| value == 0.0));
    }
}
