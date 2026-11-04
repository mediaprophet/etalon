use super::*;
use crate::ordinary::gpu_params::*;
impl<P: ProjectionGpuProvider> ProjectionGpuEngine<'_, P> {
    pub fn dispatch_fused_transformer_block(
        &self,
        tensor: &ProjectionTensor<'_>,
        input_activations: &[f32],
    ) -> Vec<f32> {
        let rows = tensor.shape.get(0).copied().unwrap_or(4096);
        let cols = tensor.shape.get(1).copied().unwrap_or(4096);

        if let Some(result) = self.host.backend_fused_block(tensor, input_activations) {
            return result;
        }
        // ── wgpu / WGSL fallback (all platforms — Vulkan on Linux/NVIDIA,
        //    Metal on macOS when mmap not loaded, D3D12 on Windows fallback) ──
        let input_bytes = bytemuck::cast_slice(input_activations);
        let input_buf = self
            .host
            .gpu_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("Input"),
                size: input_bytes.len().max(4) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.host
            .gpu_queue()
            .write_buffer(&input_buf, 0, input_bytes);

        // Upload real weights from mmap when available, else use a zero buffer.
        let weights_size = (rows * cols * 4) as wgpu::BufferAddress;
        let weights_buf = self
            .host
            .gpu_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("Weights"),
                size: weights_size.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        if let Some(mmap) = self.host.model_bytes() {
            let offset = (self.host.tensor_data_offset() + tensor.byte_offset) as usize;
            let end = (offset + rows * cols * 4).min(mmap.len());
            if end > offset {
                let f32_bytes = &mmap[offset..end];
                self.host
                    .gpu_queue()
                    .write_buffer(&weights_buf, 0, f32_bytes);
            }
        }

        let output_size = (rows * 4).max(4) as wgpu::BufferAddress;
        let output_buf = self
            .host
            .gpu_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output"),
                size: output_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

        // Upload GemmGpuParams for fused_transformer.wgsl (binding 2).
        let gemm_params = GemmGpuParams {
            n_in: cols as u32,
            n_out: rows as u32,
            weight_ggml_type: if tensor.is_quantized_q4_k { 12 } else { 14 },
            weight_row_elems: cols as u32,
            weight_byte_len: (rows * cols * 4) as u32,
            n_batch: 1,
            in_row_stride: 0,
            out_row_stride: 0,
        };
        let params_buf = self
            .host
            .gpu_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("TransformerParams"),
                size: std::mem::size_of::<GemmGpuParams>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.host
            .gpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&gemm_params));

        // The bind group MUST be built from the SAME pipeline it is dispatched
        // under (below). Both `pipeline` (Fused Transformer) and `mock_pipeline`
        // (Mock Fused Contraction) are created with `layout: None`, i.e. wgpu
        // *exclusive* auto-derived layouts — a bind group from one is rejected by
        // a dispatch that set the other ("exclusive pipelines don't match"). The
        // real path (`gguf_mmap.is_some()`) dispatches `self.pipeline`; the mock
        // path (no model, i.e. tests) dispatches `self.mock_pipeline`, so it must
        // take the mock pipeline's own group-0 layout.
        let use_mock = self.host.model_bytes().is_none();
        #[cfg(not(target_arch = "wasm32"))]
        let bind_group_layout = if use_mock {
            self.mock_pipeline.get_bind_group_layout(0)
        } else {
            self.pipeline_bind_layout.clone()
        };
        #[cfg(target_arch = "wasm32")]
        let bind_group_layout = if use_mock {
            self.mock_pipeline.get_bind_group_layout(0)
        } else {
            self.pipeline.get_bind_group_layout(0)
        };
        let bind_group = self
            .host
            .gpu_device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weights_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .host
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: self.host.pass_writes(Profile::Both),
            });
            // Same selector as the bind-group layout above — they must agree.
            let pipeline = if use_mock {
                &self.mock_pipeline
            } else {
                &self.pipeline
            };
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((rows as u32 + 63) / 64, 1, 1);
        }

        let staging_buf = self
            .host
            .gpu_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging"),
                size: output_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
        self.host.resolve(&mut encoder);
        self.host.submit(encoder.finish());
        self.host.accumulate(Phase::FusedBlock);

        let result = self
            .host
            .readback_owned(&staging_buf, output_size, true)
            .expect("fused block readback failed");

        self.host.record_operations(rows * cols);
        result
    }
}
