use std::sync::OnceLock;

#[derive(Clone)]
pub struct MetalDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}
static SOURCE: OnceLock<fn() -> Option<MetalDevice>> = OnceLock::new();
pub fn install(source: fn() -> Option<MetalDevice>) -> Result<(), fn() -> Option<MetalDevice>> {
    SOURCE.set(source)
}
pub(super) fn get() -> Option<MetalDevice> {
    SOURCE.get().and_then(|source| source())
}
