use super::*;
#[cfg(all(target_os = "macos", feature = "native-metal-gpu"))]
struct MetalMegaGpuPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}
#[cfg(all(target_os = "macos", feature = "native-metal-gpu"))]
static METAL_MEGA_WGSL_PIPELINE: OnceLock<MetalMegaGpuPipeline> = OnceLock::new();

#[cfg(all(target_os = "macos", feature = "native-metal-gpu"))]
pub(super) fn try_gpu_single_fence_argmax(logits: &[f32]) -> Option<u32> {
    let ctx = device_source::get()?;
    let mega = METAL_MEGA_WGSL_PIPELINE.get_or_init(|| {
        const SHADER: &str = r#"
            @group(0) @binding(0) var<storage, read> logits: array<f32>;
            @group(0) @binding(1) var<storage, read_write> best_out: array<u32>;
            @compute @workgroup_size(1)
            fn main() {
                let n = arrayLength(&logits);
                var best_idx: u32 = 0u;
                var best_val: f32 = logits[0];
                for (var i: u32 = 1u; i < n; i = i + 1u) {
                    if (logits[i] > best_val) { best_val = logits[i]; best_idx = i; }
                }
                best_out[0] = best_idx;
            }
        "#;
        let sm = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("metal_mega_argmax_wgsl"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            count: None,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        };
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("metal_mega_bgl"),
                entries: &[entry(0, true), entry(1, false)],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("metal_mega_pl"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("metal_mega_pipeline"),
                layout: Some(&pl),
                module: &sm,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        MetalMegaGpuPipeline { pipeline, bgl }
    });

    let logits_bytes: &[u8] = bytemuck::cast_slice(logits);
    let mk_buf = |label, size, usage| {
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let logits_buf = mk_buf(
        "metal_logits",
        (logits_bytes.len() as u64).max(4),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    ctx.queue.write_buffer(&logits_buf, 0, logits_bytes);
    let out_buf = mk_buf(
        "metal_best",
        4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let read_buf = mk_buf(
        "metal_read",
        4,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("metal_mega_bg"),
        layout: &mega.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: logits_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("metal_mega_enc"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("metal_mega_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&mega.pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, 4);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = read_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().ok()?.ok()?;
    let mapped = slice.get_mapped_range().ok()?;
    let token = u32::from_le_bytes([mapped[0], mapped[1], mapped[2], mapped[3]]);
    drop(mapped);
    read_buf.unmap();
    Some(token)
}
