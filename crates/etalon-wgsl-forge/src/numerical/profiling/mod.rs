//! Caller-owned two-slot pass timer. Serialize use and request TIMESTAMP_QUERY first.
use crate::numerical::ForgeError;

pub struct TimestampResources {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl TimestampResources {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("forge-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("forge-timestamp-resolve"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("forge-timestamp-staging"),
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        }
    }
}

impl TimestampResources {
    pub fn pass_writes(&self, begin: bool, end: bool) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: begin.then_some(0),
            end_of_pass_write_index: end.then_some(1),
        }
    }
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..2, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.staging, 0, 16);
    }
    pub fn read_ticks(&self, device: &wgpu::Device) -> Result<[u64; 2], ForgeError> {
        let result = (|| {
            let slice = self.staging.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| ForgeError::DeviceLost(format!("timestamp poll failed: {e:?}")))?;
            receiver
                .recv()
                .map_err(|e| ForgeError::GpuValidation(e.to_string()))?
                .map_err(|e| ForgeError::GpuValidation(e.to_string()))?;
            let bytes = slice
                .get_mapped_range()
                .map_err(|e| ForgeError::GpuValidation(format!("timestamp map failed: {e:?}")))?;
            let ticks: &[u64] = bytemuck::cast_slice(&bytes);
            Ok([ticks[0], ticks[1]])
        })();
        self.staging.unmap();
        result
    }
}
