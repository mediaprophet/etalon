use super::{NQuinWire, QuinField, SieveParams, NQUIN_WARP_SCAN_WGSL};
use crate::{
    readback::{check_buffer, check_dispatch, read_word, validate_capacity, MapRead},
    GpuQueryError,
};

/// GPU warp-transposed `uvec4` equality sieve over 48-byte `NQuinWire` buffers.
pub struct GpuQuinSieve {
    pub pipeline: etalon::ComputePipeline,
    pub bind_group_layout: etalon::BindGroupLayout,
}

/// Alias matching Runbook 05 naming (`GpuNQuinSieve`).
pub type GpuNQuinSieve = GpuQuinSieve;

impl GpuQuinSieve {
    pub fn new(device: &etalon::Device) -> Self {
        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("etalon-nquin-warp-scan-shader"),
            source: etalon::ShaderSource::Wgsl(NQUIN_WARP_SCAN_WGSL.into()),
        });

        let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("etalon-nquin-warp-scan-pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Record a warp-transposed `uvec4` equality scan dispatch into `cpass`.
    pub fn record_dispatch<'a>(
        &'a self,
        cpass: &mut etalon::ComputePass<'a>,
        bind_group: &'a etalon::BindGroup,
        quin_count: u32,
    ) {
        if quin_count == 0 {
            return;
        }
        let workgroups = quin_count.div_ceil(32);
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Native blocking convenience wrapper with a five-second readback timeout.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn sieve_eq_indices(
        &self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        quins: &[NQuinWire],
        field: QuinField,
        needle: u64,
    ) -> Option<Vec<u32>> {
        if quins.is_empty() {
            return Some(Vec::new());
        }
        let mut read = MapRead::new(
            self.submit_sieve(device, queue, quins, field, needle)
                .ok()?,
        );
        read.wait_blocking(device).ok()?;
        let mut out = vec![0; quins.len()];
        let count = self
            .copy_matches(&read.buffer, quins, field, needle, &mut out)
            .ok()?;
        out.truncate(count);
        Some(out)
    }

    /// GPU equality scan into caller-owned output, sorted by row index.
    ///
    /// This cold operation allocates bounded GPU scratch. If the result exceeds `out`, returns
    /// `OutputTooSmall` without copying rows. Browser WebGPU progresses on its event loop.
    /// Native callers must keep polling the device elsewhere in their runtime until completion.
    /// Dropping the future cancels mapping and releases its private readback buffer.
    pub async fn sieve_eq_indices_async(
        &self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        quins: &[NQuinWire],
        field: QuinField,
        needle: u64,
        out: &mut [u32],
    ) -> Result<usize, GpuQueryError> {
        if quins.is_empty() {
            return Ok(0);
        }
        let mut read = MapRead::new(self.submit_sieve(device, queue, quins, field, needle)?);
        (&mut read).await?;
        self.copy_matches(&read.buffer, quins, field, needle, out)
    }

    fn copy_matches(
        &self,
        stage: &etalon::Buffer,
        quins: &[NQuinWire],
        field: QuinField,
        needle: u64,
        out: &mut [u32],
    ) -> Result<usize, GpuQueryError> {
        let raw = stage
            .slice(..)
            .get_mapped_range()
            .map_err(|_| GpuQueryError::Mapping)?;
        let count = read_word(&raw, 0)? as usize;
        if count > quins.len() {
            return Err(GpuQueryError::InvalidOutput);
        }
        validate_capacity(count, out.len())?;
        for (i, row) in out[..count].iter_mut().enumerate() {
            *row = read_word(&raw, i + 2)?;
            let quin = quins
                .get(*row as usize)
                .ok_or(GpuQueryError::InvalidOutput)?;
            if field.extract(quin) != needle {
                return Err(GpuQueryError::InvalidOutput);
            }
        }
        out[..count].sort_unstable();
        if out[..count].windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(GpuQueryError::InvalidOutput);
        }
        Ok(count)
    }

    fn submit_sieve(
        &self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        quins: &[NQuinWire],
        field: QuinField,
        needle: u64,
    ) -> Result<etalon::Buffer, GpuQueryError> {
        let n = quins.len();
        let (quin_bytes, match_bytes) = sieve_sizes(n, &device.limits())?;
        let quin_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-nquin-sieve-quins"),
            size: quin_bytes,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&quin_buf, 0, bytemuck::cast_slice(quins));

        let matches_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-nquin-sieve-matches"),
            size: match_bytes,
            usage: etalon::BufferUsages::STORAGE
                | etalon::BufferUsages::COPY_SRC
                | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-nquin-sieve-count"),
            size: 4,
            usage: etalon::BufferUsages::STORAGE
                | etalon::BufferUsages::COPY_SRC
                | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&count_buf, 0, &[0u8; 4]);

        let params = SieveParams {
            n: n as u32,
            field: field.as_u32(),
            needle_lo: needle as u32,
            needle_hi: (needle >> 32) as u32,
        };
        let param_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-nquin-sieve-params"),
            size: core::mem::size_of::<SieveParams>() as u64,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&param_buf, 0, bytemuck::bytes_of(&params));

        let bg = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("etalon-nquin-sieve-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: quin_buf.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: matches_buf.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 2,
                    resource: count_buf.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 3,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });

        let out_stage = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-nquin-sieve-stage"),
            size: match_bytes + 8,
            usage: etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&etalon::CommandEncoderDescriptor {
            label: Some("etalon-nquin-sieve-enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&etalon::ComputePassDescriptor {
                label: Some("etalon-nquin-sieve-pass"),
                timestamp_writes: None,
            });
            self.record_dispatch(&mut pass, &bg, n as u32);
        }
        enc.copy_buffer_to_buffer(&count_buf, 0, &out_stage, 0, 4);
        enc.copy_buffer_to_buffer(&matches_buf, 0, &out_stage, 8, match_bytes);
        queue.submit(Some(enc.finish()));

        Ok(out_stage)
    }
}

fn sieve_sizes(n: usize, limits: &etalon::Limits) -> Result<(u64, u64), GpuQueryError> {
    let n = u32::try_from(n).map_err(|_| GpuQueryError::InputTooLarge)?;
    n.checked_mul(3).ok_or(GpuQueryError::InputTooLarge)?;
    let quin_bytes = u64::from(n).div_ceil(32) * 32 * 48;
    let match_bytes = u64::from(n) * 4;
    check_dispatch(limits, n.div_ceil(32), 32, 32 * 13 * 4, 3)?;
    check_buffer(limits, quin_bytes, true)?;
    check_buffer(limits, match_bytes, true)?;
    check_buffer(limits, match_bytes + 8, false)?;
    Ok((quin_bytes, match_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sieve_sizes_pad_tails_and_reject_limits() {
        let mut limits = etalon::Limits::default();
        assert_eq!(sieve_sizes(1, &limits), Ok((1536, 4)));
        assert_eq!(sieve_sizes(33, &limits), Ok((3072, 132)));
        limits.max_compute_workgroups_per_dimension = 1;
        assert_eq!(
            sieve_sizes(33, &limits),
            Err(GpuQueryError::UnsupportedLimits)
        );
        assert_eq!(
            sieve_sizes(u32::MAX as usize, &limits),
            Err(GpuQueryError::InputTooLarge)
        );
    }
}
