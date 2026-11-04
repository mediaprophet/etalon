use super::encoding::*;
use super::*;
use crate::ordinary::gpu_params::*;
use crate::ordinary::RMS_NORM_EPS;
impl<P: PrefillProvider> PrefillEngine<'_, P> {
    pub fn encode_elem_offset(
        &self,
        pipeline: &mut P::Pipeline,
        op: u32,
        n: u32,
        batch: u32,
        a: &wgpu::Buffer,
        a_off: wgpu::BufferAddress,
        a_bytes: wgpu::BufferAddress,
        b: &wgpu::Buffer,
        b_off: wgpu::BufferAddress,
        b_bytes: wgpu::BufferAddress,
        out: &wgpu::Buffer,
        out_off: wgpu::BufferAddress,
        out_bytes: wgpu::BufferAddress,
        a_row_stride: u32,
        b_row_stride: u32,
        out_row_stride: u32,
        a_slot: u32,
        b_slot: u32,
        out_slot: u32,
        elem_dyn_offset: u32,
    ) {
        let r = self.host.encoding_resources();
        let b_dispatch = batch.max(1);
        let (pipe, wg_x, wg_y) = match op {
            ELEM_OP_RMS_NORM => (&r.elem_rms_norm_pipeline, 1u32, b_dispatch),
            ELEM_OP_SILU_MUL => (&r.elem_silu_mul_pipeline, (n + 63) / 64, b_dispatch),
            ELEM_OP_ADD_RESIDUAL => (&r.elem_add_residual_pipeline, (n + 63) / 64, b_dispatch),
            _ => return,
        };
        let params_buf = r.elem_params_buf.as_ref().unwrap();
        let key = bg_hash(&[
            2,
            op as u64,
            a_off,
            a_bytes,
            b_off,
            b_bytes,
            out_off,
            out_bytes,
            a as *const _ as u64,
            b as *const _ as u64,
            out as *const _ as u64,
            a_row_stride as u64,
            b_row_stride as u64,
            out_row_stride as u64,
            a_slot as u64,
            b_slot as u64,
            out_slot as u64,
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
                        label: Some("ElemBindOff"),
                        layout: &r.mc8_elem_bind_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buf_slice(a, a_off, a_bytes),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: buf_slice(b, b_off, b_bytes),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: buf_slice(out, out_off, out_bytes),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: dynamic_uniform_binding(params_buf),
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
        cpass.set_pipeline(pipe);
        cpass.set_bind_group(0, &bind, &[elem_dyn_offset]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    pub fn encode_residual_add_offset(
        &self,
        pipeline: &mut P::Pipeline,
        base: &wgpu::Buffer,
        base_off: wgpu::BufferAddress,
        delta: &wgpu::Buffer,
        delta_off: wgpu::BufferAddress,
        dst: &wgpu::Buffer,
        dst_off: wgpu::BufferAddress,
        scratch: &wgpu::Buffer,
        scratch_off: wgpu::BufferAddress,
        dim: u32,
    ) {
        let bytes = (dim as usize * 4) as wgpu::BufferAddress;
        self.encode_elem_offset(
            pipeline,
            ELEM_OP_ADD_RESIDUAL,
            dim,
            1,
            base,
            base_off,
            bytes,
            delta,
            delta_off,
            bytes,
            scratch,
            scratch_off,
            bytes,
            0,
            0,
            0,
            0,
            0,
            0,
            0, // elem_dyn_offset — caller must pre-upload arena
        );
        P::encoder(pipeline).copy_buffer_to_buffer(scratch, scratch_off, dst, dst_off, bytes);
    }
    pub fn encode_elem(
        &self,
        pipeline: &mut P::Pipeline,
        op: u32,
        n: u32,
        batch: u32,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        out: &wgpu::Buffer,
    ) {
        let r = self.host.encoding_resources();
        let b_dispatch = batch.max(1);
        let params = ElemGpuParams {
            n,
            batch: b_dispatch,
            op,
            eps: RMS_NORM_EPS,
            a_row_stride: 0,
            b_row_stride: 0,
            out_row_stride: 0,
            a_slot: 0,
            b_slot: 0,
            out_slot: 0,
            _pad: 0,
        };
        let mut arena = Mc8UniformArena {
            bytes: [0u8; MC8_MAX_GEMM_UNIFORM_SLOTS * MC8_UNIFORM_ALIGN],
            slots: 0,
        };
        let elem_dyn_offset = arena.push(&params);
        let params_buf = r.elem_params_buf.as_ref().unwrap();
        arena.upload(self.host.gpu_queue(), params_buf);
        let (pipe, wg_x, wg_y) = match op {
            ELEM_OP_RMS_NORM => (&r.elem_rms_norm_pipeline, 1u32, b_dispatch),
            ELEM_OP_SILU_MUL => (&r.elem_silu_mul_pipeline, (n + 63) / 64, b_dispatch),
            ELEM_OP_ADD_RESIDUAL => (&r.elem_add_residual_pipeline, (n + 63) / 64, b_dispatch),
            _ => return,
        };
        let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ElemBind"),
            layout: &r.mc8_elem_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dynamic_uniform_binding(params_buf),
                },
            ],
        });
        let mut cpass = P::encoder(pipeline).begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipe);
        cpass.set_bind_group(0, &bind, &[elem_dyn_offset]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    pub fn encode_residual_add_gpu(
        &self,
        pipeline: &mut P::Pipeline,
        base: &wgpu::Buffer,
        delta: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        scratch: &wgpu::Buffer,
        dim: u32,
    ) {
        self.encode_elem(pipeline, ELEM_OP_ADD_RESIDUAL, dim, 1, base, delta, scratch);
        let bytes = (dim as usize * 4) as wgpu::BufferAddress;
        P::encoder(pipeline).copy_buffer_to_buffer(scratch, 0, dst, 0, bytes);
    }
}
