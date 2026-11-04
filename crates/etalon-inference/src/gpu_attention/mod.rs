use crate::ordinary::cpu::rms_norm_inplace;
use crate::ordinary::gpu_params::AttentionGpuParams;
use crate::ordinary::RMS_NORM_EPS;
pub struct PassResources<'a> {
    pub input: &'a wgpu::Buffer,
    pub weight: &'a wgpu::Buffer,
    pub output: &'a wgpu::Buffer,
    pub params: &'a wgpu::Buffer,
    pub mask: &'a wgpu::Buffer,
    pub kv: &'a wgpu::Buffer,
    pub staging: &'a wgpu::Buffer,
    pub bind_layout: &'a wgpu::BindGroupLayout,
    pub pipeline: &'a wgpu::ComputePipeline,
}
pub fn upload_input(
    queue: &wgpu::Queue,
    input: &wgpu::Buffer,
    hidden: &[f32],
    n_embd: usize,
    batch: usize,
    norm: Option<&[f32]>,
) {
    if let Some(weight) = norm.filter(|weight| n_embd <= 4096 && weight.len() >= n_embd) {
        let mut token = [0f32; 4096];
        for t in 0..batch {
            let start = t * n_embd;
            token[..n_embd].copy_from_slice(&hidden[start..start + n_embd]);
            rms_norm_inplace(&mut token[..n_embd], &weight[..n_embd], RMS_NORM_EPS);
            queue.write_buffer(
                input,
                (start * 4) as u64,
                bytemuck::cast_slice(&token[..n_embd]),
            );
        }
    } else {
        queue.write_buffer(input, 0, bytemuck::cast_slice(&hidden[..n_embd * batch]));
    }
}
pub fn encode_pass(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resources: &PassResources<'_>,
    params: &AttentionGpuParams,
    mask: &[u32],
    n_workgroups: u32,
    readback_elems: usize,
    timestamps: Option<wgpu::ComputePassTimestampWrites<'_>>,
) -> Option<wgpu::CommandEncoder> {
    let input_buf = resources.input;
    let weight_binding = resources.weight;
    let output_buf = resources.output;
    let params_buf = resources.params;
    let mask_buf = resources.mask;
    let kv_buf = resources.kv;
    let staging = resources.staging;
    let output_bytes = (readback_elems as u64).checked_mul(4)?;
    let layer_bytes = (params.layer_stride as u64).checked_mul(4)?.max(4);
    let layer_offset = (params.layer_idx as u64)
        .checked_mul(params.layer_stride as u64)?
        .checked_mul(4)?;
    if output_bytes > output_buf.size()
        || output_bytes > staging.size()
        || layer_offset.checked_add(layer_bytes)? > kv_buf.size()
    {
        return None;
    }
    queue.write_buffer(params_buf, 0, bytemuck::bytes_of(params));
    queue.write_buffer(mask_buf, 0, bytemuck::cast_slice(mask));
    // Bind one layer slice of the KV arena (full arena exceeds 128 MiB wgpu binding cap).
    let layer_f32s = params.layer_stride as usize;
    let layer_bytes = (layer_f32s * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let layer_offset = (params.layer_idx as usize * layer_f32s * std::mem::size_of::<f32>())
        as wgpu::BufferAddress;
    let kv_binding = wgpu::BufferBinding {
        buffer: kv_buf,
        offset: layer_offset,
        size: std::num::NonZeroU64::new(layer_bytes.max(4)),
    };
    let (wg_x, wg_y) = if params.proj_kind == 0 && params.num_tokens_in_batch > 1 {
        (params.n_head, params.num_tokens_in_batch)
    } else {
        (n_workgroups.max(1), 1)
    };

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FusedAttentionBindGroup"),
        layout: resources.bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weight_binding.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(kv_binding),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: mask_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("FusedAttentionEncoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FusedAttentionPass"),
            timestamp_writes: timestamps,
        });
        cpass.set_pipeline(resources.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    let readback_elems = readback_elems;
    if readback_elems > 0 {
        let out_bytes = (readback_elems * 4) as wgpu::BufferAddress;
        encoder.copy_buffer_to_buffer(output_buf, 0, staging, 0, out_bytes);
    }
    Some(encoder)
}

#[cfg(not(target_arch = "wasm32"))]
pub mod forge;
