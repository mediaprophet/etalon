use super::*;
use crate::ordinary::gpu_params::GemmGpuParams;
use etalon_model::inference::TensorInfo as GgufTensorInfo;
impl<P: BrowserProjectionProvider> BrowserProjectionEngine<'_, P> {
    pub async fn dispatch_gemm_resident_chunk_async(
        &self,
        weight_ggml_type: u32,
        weight_row_elems: u32,
        weight: wgpu::BindingResource<'_>,
        weight_byte_len: u32,
        input: &[f32],
        out: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> bool {
        let resources = self.host.resources();
        if n_in > input.len() || n_out > out.len() {
            return false;
        }
        if n_in > MAX_STACK_GEMM_IN || n_out > resources.gemm_max_out_dim as usize {
            return false;
        }
        let input_buf = match resources.gemm_input_buf {
            Some(b) => b,
            None => return false,
        };
        let output_buf = match resources.gemm_output_buf {
            Some(b) => b,
            None => return false,
        };
        let params_buf = match resources.gemm_params_buf {
            Some(b) => b,
            None => return false,
        };
        let staging = match resources.gemm_output_staging {
            Some(b) => b,
            None => return false,
        };
        let params = GemmGpuParams {
            n_in: n_in as u32,
            n_out: n_out as u32,
            weight_ggml_type,
            weight_row_elems,
            weight_byte_len,
            n_batch: 1,
            in_row_stride: 0,
            out_row_stride: 0,
        };
        self.host
            .gpu_queue()
            .write_buffer(input_buf, 0, bytemuck::cast_slice(&input[..n_in]));
        self.host
            .gpu_queue()
            .write_buffer(params_buf, 0, bytemuck::bytes_of(&params));

        let use_mmv = weight_ggml_type == etalon_model::ggml::GGML_TYPE_Q8_0 && (n_in % 32 == 0);
        let active_pipeline = if use_mmv {
            resources.mmv_q8_0_pipeline
        } else {
            resources.pipeline
        };
        let bind_layout = active_pipeline.get_bind_group_layout(0);
        let bind_group = self.host.gpu_device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ResidentLogitsBind"),
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
                    // resources.pipeline uses MC8GemmBGL, whose binding 2 is a DYNAMIC uniform — bind it
                    // as a 256-sized dynamic slice and pass offset 0 below (matches encode_gemm_bufs_offset).
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
                    label: Some("ResidentLogitsEncoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: self.host.timestamps(),
            });
            cpass.set_pipeline(active_pipeline);
            cpass.set_bind_group(0, &bind_group, &[0]);
            if use_mmv {
                cpass.dispatch_workgroups((n_out as u32 + 3) / 4, 1, 1);
            } else {
                cpass.dispatch_workgroups((n_out as u32 + 63) / 64, 1, 1);
            }
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
        false
    }
}
