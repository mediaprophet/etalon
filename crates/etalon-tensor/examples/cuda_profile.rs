#[cfg(any(not(feature = "cuda"), target_family = "wasm"))]
fn main() {
    eprintln!("cuda_profile requires --features cuda and a native CUDA device");
}

#[cfg(all(feature = "cuda", not(target_family = "wasm")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use etalon_tensor::cuda::{CudaContext, CudaPreparedTensor, CudaTensorExecutor};
    use std::time::Instant;

    fn values(length: usize, seed: usize) -> Vec<f32> {
        (0..length)
            .map(|i| (((i * 7 + seed) % 31) as f32 - 15.0) / 32.0)
            .collect()
    }

    fn gemv(matrix: &[f32], input: &[f32]) -> Vec<f32> {
        matrix
            .chunks_exact(input.len())
            .map(|row| row.iter().zip(input).map(|(a, b)| a * b).sum())
            .collect()
    }

    fn measure(
        label: &str,
        mut prepared: CudaPreparedTensor,
        input: &[f32],
        expected: &[f32],
        iterations: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut output = vec![0.0; expected.len()];
        for _ in 0..5 {
            prepared.execute(input, &mut output)?;
        }
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            prepared.execute(input, &mut output)?;
            samples.push(start.elapsed().as_secs_f64() * 1e6);
            for (index, (&actual, &expected)) in output.iter().zip(expected).enumerate() {
                if (actual - expected).abs() > 2e-5 + expected.abs() * 2e-5 || !actual.is_finite() {
                    return Err(format!("{label}[{index}]: {actual} != {expected}").into());
                }
            }
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "{label}: median_us={:.3}, min_us={:.3}, iterations={iterations}, verified={}",
            samples[iterations / 2],
            samples[0],
            output.len()
        );
        Ok(())
    }

    let mut args = std::env::args().skip(1);
    let workload = args.next().unwrap_or_else(|| "all".into());
    if !matches!(
        workload.as_str(),
        "all" | "gemv" | "gemm" | "ffn" | "ffn-tail" | "ffn-small"
    ) {
        return Err("expected all, gemv, gemm, ffn, ffn-tail or ffn-small".into());
    }
    let iterations = args
        .next()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(100);
    if iterations == 0 || args.next().is_some() {
        return Err("usage: cuda_profile [workload] [positive iterations]".into());
    }
    let ordinal = match std::env::var("ETALON_CUDA_DEVICE") {
        Ok(value) => value.parse::<i32>()?,
        Err(std::env::VarError::NotPresent) => 0,
        Err(error) => return Err(error.into()),
    };
    let executor = CudaTensorExecutor::new(CudaContext::new(ordinal)?)?;
    println!("CUDA profile: {:?}", executor.device_info());

    if workload == "all" || workload == "gemv" {
        let matrix = values(512 * 256, 3);
        let input = values(256, 7);
        measure(
            "gemv 512x256",
            executor.prepare_gemv(&matrix, 512, 256)?,
            &input,
            &gemv(&matrix, &input),
            iterations,
        )?;
    }
    if workload == "all" || workload == "gemm" {
        let (rows, inner, columns) = (64, 128, 64);
        let left = values(rows * inner, 3);
        let right = values(inner * columns, 7);
        let expected: Vec<f32> = (0..rows * columns)
            .map(|index| {
                (0..inner)
                    .map(|k| {
                        left[index / columns * inner + k] * right[k * columns + index % columns]
                    })
                    .sum()
            })
            .collect();
        measure(
            "gemm 64x128x64",
            executor.prepare_gemm(&right, rows, inner, columns)?,
            &left,
            &expected,
            iterations,
        )?;
    }
    for (name, input_size, hidden_size, output_size) in [
        ("ffn", 256, 512, 512),
        ("ffn-tail", 127, 267, 263),
        ("ffn-small", 11, 13, 263),
    ] {
        if workload != "all" && workload != name {
            continue;
        }
        let input = values(input_size, 7);
        let w1 = values(hidden_size * input_size, 2);
        let w2 = values(output_size * hidden_size, 9);
        let hidden: Vec<f32> = gemv(&w1, &input)
            .into_iter()
            .map(|x| 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh()))
            .collect();
        measure(
            &format!("{name} {input_size}x{hidden_size}x{output_size}"),
            executor.prepare_fused_ffn(&w1, &w2, input_size, hidden_size, output_size)?,
            &input,
            &gemv(&w2, &hidden),
            iterations,
        )?;
    }
    Ok(())
}
