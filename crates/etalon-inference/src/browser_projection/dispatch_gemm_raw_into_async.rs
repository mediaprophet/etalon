use super::*;
use crate::ordinary::gpu_params::GemmGpuParams;
use etalon_model::inference::TensorInfo as GgufTensorInfo;
impl<P: BrowserProjectionProvider> BrowserProjectionEngine<'_, P> {
    pub async fn dispatch_gemm_raw_into_async(
        &self,
        info: &GgufTensorInfo,
        raw: &[u8],
        input: &[f32],
        out: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> bool {
        let resources = self.host.resources();
        if n_in > input.len() || n_out > out.len() {
            return false;
        }

        let weight_bytes = raw.len();
        if crate::ordinary::quant_support::ggml_gpu_attention_shader_supported(info.ggml_type)
            && n_in <= MAX_STACK_GEMM_IN
            && n_out <= resources.gemm_max_out_dim as usize
            && weight_bytes <= resources.max_tensor_bytes
            && resources.gemm_input_buf.is_some()
        {
            let params = GemmGpuParams {
                n_in: n_in as u32,
                n_out: n_out as u32,
                weight_ggml_type: info.ggml_type,
                weight_row_elems: info.dims[0] as u32,
                weight_byte_len: raw.len() as u32,
                n_batch: 1,
                in_row_stride: 0,
                out_row_stride: 0,
            };
            let input_buf = resources.gemm_input_buf.unwrap();
            let weight_buf = resources.gemm_weight_buf.unwrap();
            let output_buf = resources.gemm_output_buf.unwrap();
            let params_buf = resources.gemm_params_buf.unwrap();
            let staging = resources.gemm_output_staging.unwrap();

            self.host
                .gpu_queue()
                .write_buffer(input_buf, 0, bytemuck::cast_slice(&input[..n_in]));
            self.host
                .write_weight_words(raw, resources.max_tensor_bytes);
            self.host
                .gpu_queue()
                .write_buffer(params_buf, 0, bytemuck::bytes_of(&params));

            let bind_layout = resources.pipeline.get_bind_group_layout(0);
            let bind_group = self
                .host
                .gpu_device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("LayerGemmBindGroup"),
                    layout: &bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: weight_buf.as_entire_binding(),
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

            let mut encoder =
                self.host
                    .gpu_device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("LayerGemmEncoder"),
                    });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: self.host.timestamps(),
                });
                cpass.set_pipeline(resources.pipeline);
                cpass.set_bind_group(0, &bind_group, &[0]);
                cpass.dispatch_workgroups((n_out as u32 + 63) / 64, 1, 1);
            }
            let out_bytes = (n_out * 4) as wgpu::BufferAddress;
            encoder.copy_buffer_to_buffer(output_buf, 0, staging, 0, out_bytes);
            self.host.resolve(&mut encoder);
            if !self.host.submit(encoder.finish()) {
                return false;
            }
            self.host.accumulate_gemm();

            let slice = staging.slice(..out_bytes);
            if self.host.await_map(slice).await {
                let data = slice
                    .get_mapped_range()
                    .expect("wgpu buffer map_range failed");
                let floats: &[f32] = bytemuck::cast_slice(&data);
                out[..n_out].copy_from_slice(&floats[..n_out]);
                drop(data);
                staging.unmap();
                return true;
            }
            let _ = staging.unmap();
        }

        self.host.gemm(info, raw, input, out, n_in, n_out)
    }
}
