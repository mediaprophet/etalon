use std::time::Instant;
const GEMV_BENCH_WGSL: &str = include_str!("gemv_bench.wgsl");

#[inline]
fn params_bytes(n_in: u32, n_out: u32) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[0..4].copy_from_slice(&n_in.to_le_bytes());
    b[4..8].copy_from_slice(&n_out.to_le_bytes());
    b
}

/// Persistent-pipeline GEMV timing on one wgpu device (ms per dispatch). No readback — we poll to
/// completion so the timing reflects execution, with submit overhead amortized over K dispatches.
pub(super) fn bench_gpu_gemv(
    device: &etalon::Device,
    queue: &etalon::Queue,
    n: usize,
) -> Result<f64, String> {
    let weight_bytes = (n as u64)
        .checked_mul(n as u64)
        .and_then(|v| v.checked_mul(4))
        .ok_or("GPU benchmark size overflow")?;
    let limits = device.limits();
    if weight_bytes > limits.max_buffer_size
        || weight_bytes > limits.max_storage_buffer_binding_size as u64
    {
        return Err("GPU benchmark exceeds device buffer limits".into());
    }
    let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
        label: Some("gemv_bench"),
        source: etalon::ShaderSource::Wgsl(GEMV_BENCH_WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
        label: Some("gemv_bench_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("gemv"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mk = |contents: &[u8], usage: etalon::BufferUsages| {
        let b = device.create_buffer(&etalon::BufferDescriptor {
            label: None,
            size: contents.len().max(4) as u64,
            usage,
            mapped_at_creation: false,
        });
        if !contents.is_empty() {
            queue.write_buffer(&b, 0, contents);
        }
        b
    };
    let input = mk(
        bytemuck::cast_slice(&vec![0.1f32; n]),
        etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
    );
    let weight = mk(
        bytemuck::cast_slice(&vec![0.05f32; n * n]),
        etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
    );
    let params = mk(
        &params_bytes(n as u32, n as u32),
        etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
    );
    let out = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: (n * 4).max(4) as u64,
        usage: etalon::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&etalon::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            etalon::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            etalon::BindGroupEntry {
                binding: 1,
                resource: weight.as_entire_binding(),
            },
            etalon::BindGroupEntry {
                binding: 2,
                resource: params.as_entire_binding(),
            },
            etalon::BindGroupEntry {
                binding: 3,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let wg_x = (n as u32).div_ceil(64).max(1);
    let (k, s) = (16u32, 5u32);
    let submit_batch = || -> Result<(), String> {
        let mut enc =
            device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&etalon::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            for _ in 0..k {
                pass.dispatch_workgroups(wg_x, 1, 1);
            }
        }
        queue.submit(Some(enc.finish()));
        device
            .poll(etalon::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll: {e}"))?;
        Ok(())
    };
    submit_batch()?; // warmup
    let t0 = Instant::now();
    for _ in 0..s {
        submit_batch()?;
    }
    Ok(t0.elapsed().as_secs_f64() * 1e3 / (k * s) as f64)
}

/// Host→device upload bandwidth (GB/s) for a `bytes`-sized buffer — the transfer axis (D31).
/// Times `write_buffer` + a flushing submit + `poll(Wait)` so the upload is realized. For a discrete
/// GPU this is the PCIe cost; for an iGPU it's the wgpu staging path (not the true near-zero of a
/// unified pool — a relative signal, flagged honestly).
pub(super) fn bench_upload_gbps(
    device: &etalon::Device,
    queue: &etalon::Queue,
    bytes: usize,
) -> Result<f64, String> {
    let data = vec![0u8; bytes];
    let buf = device.create_buffer(&etalon::BufferDescriptor {
        label: Some("upload_probe"),
        size: bytes as u64,
        usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let flush = || -> Result<(), String> {
        let enc = device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
        queue.submit(Some(enc.finish()));
        device
            .poll(etalon::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll: {e}"))?;
        Ok(())
    };
    queue.write_buffer(&buf, 0, &data); // warmup
    flush()?;
    let iters = 3u32;
    let t0 = Instant::now();
    for _ in 0..iters {
        queue.write_buffer(&buf, 0, &data);
        flush()?;
    }
    let secs = t0.elapsed().as_secs_f64() / iters as f64;
    Ok(if secs <= 0.0 {
        0.0
    } else {
        bytes as f64 / secs / 1e9
    })
}
