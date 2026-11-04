use super::*;
use std::sync::mpsc;

struct Unmap<'a>(&'a wgpu::Buffer);
impl Drop for Unmap<'_> {
    fn drop(&mut self) {
        self.0.unmap();
    }
}

impl WgpuComputeContext {
    /// Reads back a buffer as f32s via `view`.
    pub fn read_buffer_f32(&self, view: &BufferView) -> Result<Vec<f32>, ForgeError> {
        self.ensure_live()?;
        let size = view.length_bytes as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forge-output-staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forge-output-copy"),
            });
        encoder.copy_buffer_to_buffer(
            self.slab_for(view.usage),
            view.offset as u64,
            &staging,
            0,
            size,
        );
        self.submit_commands(encoder.finish())?;
        let _unmap = Unmap(&staging);
        let bytes = map_read(self, &staging)?;

        let elements = view.length_bytes / size_of::<f32>();
        let output = bytemuck::cast_slice::<u8, f32>(&bytes)[..elements].to_vec();
        drop(bytes);
        self.ensure_live()?;
        Ok(output)
    }
}

fn map_read(
    context: &WgpuComputeContext,
    buffer: &wgpu::Buffer,
) -> Result<wgpu::BufferView, ForgeError> {
    context.ensure_live()?;
    let device = &context.device;
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| ForgeError::DeviceLost(format!("device poll failed during map: {e:?}")))?;
    receiver
        .recv()
        .map_err(|error| ForgeError::GpuValidation(error.to_string()))?
        .map_err(|error| ForgeError::GpuValidation(error.to_string()))?;
    context.ensure_live()?;
    // wgpu 30: get_mapped_range() returns Result — propagate as ForgeError.
    slice
        .get_mapped_range()
        .map_err(|e| ForgeError::GpuValidation(format!("map_range failed: {e:?}")))
}
