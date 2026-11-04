use std::time::Instant;

pub const DEFAULT_BENCH_BYTES: usize = 256 * 1024 * 1024;

pub fn validate_shape(n: usize, budget: usize) -> Result<(), String> {
    let bytes = n
        .checked_mul(n)
        .and_then(|v| v.checked_mul(16))
        .and_then(|v| n.checked_mul(64).and_then(|tail| v.checked_add(tail)))
        .and_then(|v| v.checked_add(32 * 1024 * 1024))
        .ok_or("benchmark size overflow")?;
    if n == 0 || n > u32::MAX as usize || bytes > budget {
        return Err("benchmark shape exceeds byte budget".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingSource {
    HostWallClock,
    SubmitAndCompletionWallClock,
    DeviceTimestamp,
}

pub fn timestamp_nanoseconds(begin: u64, end: u64, period_ns: f32) -> Option<u64> {
    if !period_ns.is_finite() || period_ns <= 0.0 {
        return None;
    }
    let value = end.checked_sub(begin)? as f64 * period_ns as f64;
    if value.is_finite() && value < u64::MAX as f64 {
        Some(value as u64)
    } else {
        None
    }
}

/// Native-Rust (`rayon`) GEMV timing (ms per GEMV) — the CPU compute path, fairly multi-threaded.
pub fn bench_cpu_gemv(n: usize, byte_budget: usize) -> Result<f64, String> {
    validate_shape(n, byte_budget)?;
    use rayon::prelude::*;
    let weight = vec![0.05f32; n * n];
    let input = vec![0.1f32; n];
    let mut out = vec![0.0f32; n];
    let gemv = |out: &mut [f32]| {
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let row = &weight[i * n..(i + 1) * n];
            *o = row.iter().zip(input.iter()).map(|(a, b)| a * b).sum();
        });
    };
    gemv(&mut out); // warmup
    let iters = 5u32;
    let t0 = Instant::now();
    for _ in 0..iters {
        gemv(&mut out);
    }
    Ok(t0.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

#[inline]
pub fn gflops(n: usize, ms: f64) -> f64 {
    if !ms.is_finite() || ms <= 0.0 {
        0.0
    } else {
        (2.0 * n as f64 * n as f64) / (ms / 1e3) / 1e9
    }
}
