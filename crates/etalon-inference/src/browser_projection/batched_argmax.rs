use super::*;
use crate::ordinary::gpu_params::GemmGpuParams;
impl<P: BrowserProjectionProvider> BrowserProjectionEngine<'_, P> {
    pub async fn dispatch_output_argmax_batched_async(
        &self,
        index: &P::Index,
        hidden: &[f32],
        emb_dim: usize,
        chunk_logits: &mut [f32],
        max_chunks: u32,
        sieve_mask: Option<&P::Mask>,
    ) -> Option<StreamingArgmaxResult> {
        let resources = self.host.resources();
        if sieve_mask.is_some() {
            return None;
        }
        let info = P::logits(index)?;
        let (n_in, vocab_size) = crate::ordinary::matmul_dims(&info);
        if n_in == 0 || vocab_size == 0 || n_in > emb_dim || n_in > hidden.len() {
            return None;
        }
        if chunk_logits.len() < VOCAB_CHUNK_ROWS || !self.host.buffers_ready() {
            return None;
        }
        let resident_buf = resources.mc8_logits_resident_buf.as_ref()?;
        let row_bytes = resources.mc8_logits_row_bytes as u64;
        let input_buf = resources.gemm_input_buf.as_ref().unwrap();
        let output_buf = resources.gemm_output_buf.as_ref().unwrap();
        let params_buf = resources.gemm_params_buf.as_ref().unwrap();
        let staging = resources.gemm_output_staging.as_ref().unwrap();

        let full_chunks = vocab_size.div_ceil(VOCAB_CHUNK_ROWS);
        let n_chunks = if max_chunks == 0 {
            full_chunks
        } else {
            (max_chunks as usize).min(full_chunks)
        };
        let top1_plan = P::top1_plan(vocab_size, n_chunks as u32)?;
        if !self.host.prepare_browser_top1(top1_plan) {
            return None;
        }

        // Write input hidden + params once
        let params = GemmGpuParams {
            n_in: n_in as u32,
            n_out: VOCAB_CHUNK_ROWS as u32,
            weight_ggml_type: info.ggml_type,
            weight_row_elems: info.dims[0] as u32,
            weight_byte_len: (VOCAB_CHUNK_ROWS as u64 * row_bytes) as u32,
            n_batch: 1,
            in_row_stride: 0,
            out_row_stride: 0,
        };
        self.host
            .gpu_queue()
            .write_buffer(input_buf, 0, bytemuck::cast_slice(&hidden[..n_in]));
        self.host
            .gpu_queue()
            .write_buffer(params_buf, 0, bytemuck::bytes_of(&params));

        let use_mmv_q8_0 = info.ggml_type == etalon_model::ggml::GGML_TYPE_Q8_0 && (n_in % 32 == 0);
        let logits_pipeline: &wgpu::ComputePipeline = if use_mmv_q8_0 {
            &resources.mmv_q8_0_pipeline
        } else {
            &resources.pipeline
        };
        let bind_layout = logits_pipeline.get_bind_group_layout(0);
        let mut encoder =
            self.host
                .gpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("BatchedArgmaxEncoder"),
                });

        // Each chunk reuses the bounded logits buffer, then immediately reduces
        // it to compact block candidates before the next GEMM overwrites it.
        for chunk_idx in 0..n_chunks {
            let row_start = chunk_idx * VOCAB_CHUNK_ROWS;
            let chunk_rows = VOCAB_CHUNK_ROWS.min(vocab_size - row_start);
            let weight = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: resident_buf,
                offset: row_start as u64 * row_bytes,
                size: std::num::NonZeroU64::new(chunk_rows as u64 * row_bytes),
            });
            let bind_group = self
                .host
                .gpu_device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("BatchedArgmaxBind"),
                    layout: &bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: weight,
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: dynamic_uniform_binding(params_buf),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: output_buf.as_entire_binding(),
                        },
                    ],
                });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(logits_pipeline);
                cpass.set_bind_group(0, &bind_group, &[0]);
                if use_mmv_q8_0 {
                    cpass.dispatch_workgroups((chunk_rows as u32 + 3) / 4, 1, 1);
                } else {
                    cpass.dispatch_workgroups((chunk_rows as u32 + 63) / 64, 1, 1);
                }
            }
            if !self.host.encode_browser_top1_chunk(
                &mut encoder,
                output_buf,
                staging,
                top1_plan,
                chunk_idx,
            ) {
                return None;
            }
        }

        self.host.submit(encoder.finish());
        self.host.read_browser_top1(staging, top1_plan).await
    }
}
