pub trait ForgeAttentionHost {
    fn device(&self) -> &wgpu::Device;
    fn submit(&self, command: wgpu::CommandBuffer);
    fn poll_wait(&self);
    fn wait_for_map(
        &self,
        receiver: futures_channel::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>,
    ) -> bool;
}
pub fn dispatch_forge_attention<H: ForgeAttentionHost>(
    host: &H,
    source: &str,
    params: &[u8],
    q_len: usize,
    kv_len: usize,
    head_count: u32,
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    out: &mut [f32],
) -> bool {
    let module = host
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ForgeCausalAttentionShader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = host
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ForgeCausalAttentionPipeline"),
            layout: None,
            module: &module,
            entry_point: Some("causal_attention"),
            compilation_options: Default::default(),
            cache: None,
        });

    use wgpu::util::DeviceExt;
    let q_buf = host
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ForgeAttnQ"),
            contents: bytemuck::cast_slice(&query[..q_len]),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let k_buf = host
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ForgeAttnK"),
            contents: bytemuck::cast_slice(&keys[..kv_len]),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let v_buf = host
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ForgeAttnV"),
            contents: bytemuck::cast_slice(&values[..kv_len]),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let out_bytes = (q_len * std::mem::size_of::<f32>()) as u64;
    let out_buf = host.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("ForgeAttnOut"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_buf = host
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ForgeAttnParams"),
            contents: params,
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = host.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ForgeAttnBG"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: k_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: v_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let staging = host.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("ForgeAttnStaging"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = host
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ForgeAttnEncoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ForgeAttnPass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(head_count, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
    host.submit(encoder.finish());

    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    host.poll_wait();

    if host.wait_for_map(rx) {
        let data = slice
            .get_mapped_range()
            .expect("wgpu buffer map_range failed");
        let floats: &[f32] = bytemuck::cast_slice(&data);
        out[..q_len].copy_from_slice(&floats[..q_len]);
        drop(data);
        staging.unmap();
        return true;
    }
    staging.unmap();
    false
}
