use super::{PrefixParams, RadixParams};
#[cfg(not(target_arch = "wasm32"))]
use crate::layout::NQuinWire;
use crate::{
    readback::{check_buffer, check_dispatch, read_word, validate_capacity, MapRead},
    GpuQueryError,
};

pub struct GpuRadixSorter {
    pub hist_pipeline: etalon::ComputePipeline,
    pub prefix_pipeline: etalon::ComputePipeline,
    pub scatter_pipeline: etalon::ComputePipeline,
    pub hist_layout: etalon::BindGroupLayout,
    pub prefix_layout: etalon::BindGroupLayout,
    pub scatter_layout: etalon::BindGroupLayout,
}

impl GpuRadixSorter {
    pub fn new(device: &etalon::Device) -> Self {
        let hist_shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("etalon-radix-hist-shader"),
            source: etalon::ShaderSource::Wgsl(
                include_str!("../shaders/graph_radix_hist.wgsl").into(),
            ),
        });
        let prefix_shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("etalon-radix-prefix-shader"),
            source: etalon::ShaderSource::Wgsl(
                include_str!("../shaders/graph_radix_prefix.wgsl").into(),
            ),
        });
        let scatter_shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("etalon-radix-scatter-shader"),
            source: etalon::ShaderSource::Wgsl(
                include_str!("../shaders/graph_radix_scatter.wgsl").into(),
            ),
        });

        let hist_pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("etalon-radix-hist-pipeline"),
            layout: None,
            module: &hist_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let prefix_pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("etalon-radix-prefix-pipeline"),
            layout: None,
            module: &prefix_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scatter_pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("etalon-radix-scatter-pipeline"),
            layout: None,
            module: &scatter_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let hist_layout = hist_pipeline.get_bind_group_layout(0);
        let prefix_layout = prefix_pipeline.get_bind_group_layout(0);
        let scatter_layout = scatter_pipeline.get_bind_group_layout(0);

        Self {
            hist_pipeline,
            prefix_pipeline,
            scatter_pipeline,
            hist_layout,
            prefix_layout,
            scatter_layout,
        }
    }

    /// Sort predicates on native backends. This cold convenience API allocates GPU and CPU buffers.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn sort_by_predicate(
        &self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        quins: &[NQuinWire],
    ) -> Option<Vec<u32>> {
        if !quins.is_empty() {
            sort_sizes(quins.len(), &device.limits()).ok()?;
        }
        let keys: Vec<u64> = quins.iter().map(|q| q.predicate).collect();
        self.sort_indices(device, queue, &keys)
    }

    /// Native blocking convenience wrapper with a five-second readback timeout.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn sort_indices(
        &self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        keys: &[u64],
    ) -> Option<Vec<u32>> {
        if keys.is_empty() {
            return Some(Vec::new());
        }
        let mut read = MapRead::new(self.submit_sort(device, queue, keys).ok()?);
        let mut out = vec![0; keys.len()];
        read.wait_blocking(device).ok()?;
        self.copy_sorted(&read.buffer, keys, &mut out).ok()?;
        Some(out)
    }

    /// Stable 64-bit GPU sort into caller-owned output; returns the number of indices written.
    ///
    /// This cold operation allocates bounded GPU/CPU scratch. `out` must hold `keys.len()` rows.
    /// Browser WebGPU progresses on its event loop. Native callers must keep polling the device
    /// elsewhere in their runtime until this future completes. Dropping the future cancels mapping.
    pub async fn sort_indices_async(
        &self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        keys: &[u64],
        out: &mut [u32],
    ) -> Result<usize, GpuQueryError> {
        validate_capacity(keys.len(), out.len())?;
        if keys.is_empty() {
            return Ok(0);
        }
        let mut read = MapRead::new(self.submit_sort(device, queue, keys)?);
        (&mut read).await?;
        self.copy_sorted(&read.buffer, keys, out)
    }

    fn copy_sorted(
        &self,
        stage: &etalon::Buffer,
        keys: &[u64],
        out: &mut [u32],
    ) -> Result<usize, GpuQueryError> {
        let raw = stage
            .slice(..)
            .get_mapped_range()
            .map_err(|_| GpuQueryError::Mapping)?;
        copy_sorted_rows(&raw, keys, out)
    }

    fn submit_sort(
        &self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        keys: &[u64],
    ) -> Result<etalon::Buffer, GpuQueryError> {
        let n = keys.len();
        let limits = device.limits();
        let (key_bytes, idx_bytes, wg, total_bins, hist_size) = sort_sizes(n, &limits)?;
        let mut packed = vec![0u32; n * 2];
        for (i, &x) in keys.iter().enumerate() {
            packed[i * 2] = x as u32;
            packed[i * 2 + 1] = (x >> 32) as u32;
        }
        let idx: Vec<u32> = (0..n as u32).collect();

        let usage_st = etalon::BufferUsages::STORAGE
            | etalon::BufferUsages::COPY_SRC
            | etalon::BufferUsages::COPY_DST;

        let k0 = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-k0"),
            size: key_bytes,
            usage: usage_st,
            mapped_at_creation: false,
        });
        let k1 = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-k1"),
            size: key_bytes,
            usage: usage_st,
            mapped_at_creation: false,
        });
        let i0 = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-i0"),
            size: idx_bytes,
            usage: usage_st,
            mapped_at_creation: false,
        });
        let i1 = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-i1"),
            size: idx_bytes,
            usage: usage_st,
            mapped_at_creation: false,
        });

        queue.write_buffer(&k0, 0, bytemuck::cast_slice(&packed));
        queue.write_buffer(&i0, 0, bytemuck::cast_slice(&idx));

        let hist_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-hist"),
            size: hist_size,
            usage: usage_st,
            mapped_at_creation: false,
        });
        let off_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-off"),
            size: hist_size,
            usage: usage_st,
            mapped_at_creation: false,
        });

        let prefix_params = PrefixParams {
            total_bins,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let prefix_param_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-prefix-params"),
            size: core::mem::size_of::<PrefixParams>() as u64,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&prefix_param_buf, 0, bytemuck::bytes_of(&prefix_params));

        let prefix_bg = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("etalon-radix-prefix-bg"),
            layout: &self.prefix_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: hist_buf.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: off_buf.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 2,
                    resource: prefix_param_buf.as_entire_binding(),
                },
            ],
        });

        // Pre-create per-pass uniform buffers for shifts 0, 8, 16, 24, 32, 40, 48, 56
        // so all 8 LSD passes are recorded into a single CommandEncoder.
        let param_bufs: [etalon::Buffer; 8] = core::array::from_fn(|pass| {
            let params = RadixParams {
                n: n as u32,
                shift: (pass * 8) as u32,
                num_wgs: wg,
                _pad1: 0,
            };
            let buf = device.create_buffer(&etalon::BufferDescriptor {
                label: Some("etalon-radix-pass-params"),
                size: core::mem::size_of::<RadixParams>() as u64,
                usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::bytes_of(&params));
            buf
        });

        let out_stage = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-out-stage"),
            size: idx_bytes,
            usage: etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut src_k = &k0;
        let mut src_i = &i0;
        let mut dst_k = &k1;
        let mut dst_i = &i1;

        let mut enc = device.create_command_encoder(&etalon::CommandEncoderDescriptor {
            label: Some("etalon-radix-single-submission-enc"),
        });

        for param_buf in &param_bufs {
            let hist_bg = device.create_bind_group(&etalon::BindGroupDescriptor {
                label: Some("etalon-radix-hist-bg"),
                layout: &self.hist_layout,
                entries: &[
                    etalon::BindGroupEntry {
                        binding: 0,
                        resource: src_k.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 1,
                        resource: hist_buf.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 2,
                        resource: param_buf.as_entire_binding(),
                    },
                ],
            });

            let sc_bg = device.create_bind_group(&etalon::BindGroupDescriptor {
                label: Some("etalon-radix-scatter-bg"),
                layout: &self.scatter_layout,
                entries: &[
                    etalon::BindGroupEntry {
                        binding: 0,
                        resource: src_k.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 1,
                        resource: src_i.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 2,
                        resource: dst_k.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 3,
                        resource: dst_i.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 4,
                        resource: off_buf.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 5,
                        resource: param_buf.as_entire_binding(),
                    },
                ],
            });

            let mut pass = enc.begin_compute_pass(&etalon::ComputePassDescriptor {
                label: Some("etalon-radix-lsd-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.hist_pipeline);
            pass.set_bind_group(0, &hist_bg, &[]);
            pass.dispatch_workgroups(wg, 1, 1);

            pass.set_pipeline(&self.prefix_pipeline);
            pass.set_bind_group(0, &prefix_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);

            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &sc_bg, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
            drop(pass);

            std::mem::swap(&mut src_k, &mut dst_k);
            std::mem::swap(&mut src_i, &mut dst_i);
        }

        enc.copy_buffer_to_buffer(src_i, 0, &out_stage, 0, idx_bytes);
        queue.submit(Some(enc.finish()));

        Ok(out_stage)
    }
}

fn copy_sorted_rows(raw: &[u8], keys: &[u64], out: &mut [u32]) -> Result<usize, GpuQueryError> {
    validate_capacity(keys.len(), out.len())?;
    let mut previous = None;
    for i in 0..keys.len() {
        let row = read_word(raw, i)?;
        let key = *keys.get(row as usize).ok_or(GpuQueryError::InvalidOutput)?;
        let current = (key, row);
        if previous.is_some_and(|prev| current <= prev) {
            return Err(GpuQueryError::InvalidOutput);
        }
        previous = Some(current);
    }
    for (i, row) in out[..keys.len()].iter_mut().enumerate() {
        *row = read_word(raw, i)?;
    }
    Ok(keys.len())
}

fn sort_sizes(
    n: usize,
    limits: &etalon::Limits,
) -> Result<(u64, u64, u32, u32, u64), GpuQueryError> {
    let n = u32::try_from(n).map_err(|_| GpuQueryError::InputTooLarge)?;
    n.checked_mul(2).ok_or(GpuQueryError::InputTooLarge)?;
    let key_bytes = u64::from(n) * 8;
    let idx_bytes = u64::from(n) * 4;
    let wg = n.div_ceil(64);
    check_dispatch(limits, wg, 64, 1024, 5)?;
    let total_bins = wg.checked_mul(256).ok_or(GpuQueryError::InputTooLarge)?;
    let hist_size = u64::from(total_bins) * 4;
    for size in [key_bytes, idx_bytes, hist_size] {
        check_buffer(limits, size, true)?;
    }
    Ok((key_bytes, idx_bytes, wg, total_bins, hist_size))
}

#[cfg(test)]
#[path = "gpu_tests.rs"]
mod tests;
