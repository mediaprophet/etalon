use super::*;
use crate::ordinary::gpu_params::*;
impl<P: ProjectionGpuProvider> ProjectionGpuEngine<'_, P> {
    pub fn dispatch_quantized_token_embedding(
        &self,
        raw_embd: &[u8],
        ggml_type: u32,
        n_embd: u32,
        weight_tensor: &ProjectionTensor<'_>,
    ) -> Option<Vec<f32>> {
        if ggml_type != etalon_model::ggml::GGML_TYPE_Q6_K || raw_embd.is_empty() || n_embd == 0 {
            return None;
        }

        let n_output = weight_tensor
            .shape
            .first()
            .copied()
            .unwrap_or(n_embd as usize) as u32;
        let n_embd_u = n_embd;
        let weights_elems = (n_output as usize).saturating_mul(n_embd as usize);

        let params = EmbeddingGpuParams {
            n_embd: n_embd_u,
            ggml_type,
            n_output,
            raw_byte_len: raw_embd.len() as u32,
        };

        // WGSL storage uses u32 words; pad mmap slice to 4-byte alignment.
        let word_bytes = raw_embd.len().div_ceil(4) * 4;
        let embd_buf = self
            .host
            .gpu_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("QuantizedEmbeddingBytes"),
                size: word_bytes.max(4) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        if raw_embd.len() == word_bytes {
            self.host.gpu_queue().write_buffer(&embd_buf, 0, raw_embd);
        } else {
            const MAX_EMB_ROW_PAD: usize = 8192;
            if word_bytes > MAX_EMB_ROW_PAD {
                return None;
            }
            let mut padded = [0u8; MAX_EMB_ROW_PAD];
            padded[..raw_embd.len()].copy_from_slice(raw_embd);
            self.host
                .gpu_queue()
                .write_buffer(&embd_buf, 0, &padded[..word_bytes]);
        }

        let params_buf = self
            .host
            .gpu_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("EmbeddingParams"),
                size: std::mem::size_of::<EmbeddingGpuParams>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.host
            .gpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let weights_size = (weights_elems * 4).max(4) as wgpu::BufferAddress;
        let weights_buf = self
            .host
            .gpu_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("EmbeddingWeights"),
                size: weights_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        if let Some(mmap) = self.host.model_bytes() {
            let offset = (self.host.tensor_data_offset() + weight_tensor.byte_offset) as usize;
            let end = (offset + weights_elems * 4).min(mmap.len());
            if end > offset {
                self.host
                    .gpu_queue()
                    .write_buffer(&weights_buf, 0, &mmap[offset..end]);
            }
        }

        let output_size = (n_output as usize * 4).max(4) as wgpu::BufferAddress;
        let output_buf = self
            .host
            .gpu_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("EmbeddingOutput"),
                size: output_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

        #[cfg(not(target_arch = "wasm32"))]
        let bind_layout = self.embedding_bind_layout.clone();
        #[cfg(target_arch = "wasm32")]
        let bind_layout = self.embedding_pipeline.get_bind_group_layout(0);
        let bind_group = self
            .host
            .gpu_device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("QuantizedEmbeddingBindGroup"),
                layout: &bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: embd_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: weights_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            self.host
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("QuantizedEmbeddingEncoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("QuantizedEmbeddingPass"),
                timestamp_writes: self.host.pass_writes(Profile::Both),
            });
            cpass.set_pipeline(&self.embedding_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((n_output + 63) / 64, 1, 1);
        }

        let staging_buf = self
            .host
            .gpu_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("EmbeddingStaging"),
                size: output_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
        self.host.resolve(&mut encoder);
        self.host.submit(encoder.finish());
        self.host.accumulate(Phase::Embedding);

        let result = self.host.readback_owned(&staging_buf, output_size, false)?;

        self.host.record_operations(weights_elems);
        Some(result)
    }
}
