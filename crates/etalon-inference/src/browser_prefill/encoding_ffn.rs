use super::encoding::*;
use super::*;
impl<P: PrefillProvider> PrefillEngine<'_, P> {
    pub fn encode_fused_ffn_expansion(
        &self,
        pipeline: &mut P::Pipeline,
        layer: u32,
        input: &wgpu::Buffer,
        in_off: wgpu::BufferAddress,
        in_bytes: wgpu::BufferAddress,
        output: &wgpu::Buffer,
        out_off: wgpu::BufferAddress,
        out_bytes: wgpu::BufferAddress,
        n_out: u32,
        n_batch: u32,
        gemm_dyn_offset: u32,
    ) -> bool {
        let r = self.host.encoding_resources();
        let params_buf = r.gemm_params_buf.as_ref().unwrap();
        let key = bg_hash(&[
            3,
            layer as u64,
            in_off,
            in_bytes,
            out_off,
            out_bytes,
            input as *const _ as u64,
            output as *const _ as u64,
        ]);
        let bind = {
            let cached = r
                .mc8_bg_cache
                .lock()
                .ok()
                .and_then(|c| c.get(&key).cloned());
            match cached {
                Some(bg) => bg,
                None => {
                    let bg = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("MC8FfnFusedBind"),
                        layout: &r.mc8_ffn_fused_bind_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buf_slice(input, in_off, in_bytes),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.weight_binding(Mc8WeightRole::Gate, layer),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.weight_binding(Mc8WeightRole::Up, layer),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: dynamic_uniform_binding(params_buf),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: buf_slice(output, out_off, out_bytes),
                            },
                        ],
                    });
                    if let Ok(mut c) = r.mc8_bg_cache.lock() {
                        c.insert(key, bg.clone());
                    }
                    bg
                }
            }
        };
        let mut cpass = P::encoder(pipeline).begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&r.mc8_ffn_fused_pipeline);
        cpass.set_bind_group(0, &bind, &[gemm_dyn_offset]);
        cpass.dispatch_workgroups((n_out + 63) / 64, n_batch.max(1), 1);
        true
    }
}
