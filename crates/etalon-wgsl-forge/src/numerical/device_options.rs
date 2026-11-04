use std::sync::OnceLock;
use wgpu::Backends;
pub type SharedDevice = (
    wgpu::Device,
    wgpu::Queue,
    etalon_resources::device_caps::GpuAdapterCaps,
);
#[derive(Clone, Copy)]
pub struct DevicePolicy {
    pub backend: fn() -> Option<Backends>,
    pub experimental: fn() -> bool,
    pub shared: fn() -> Option<SharedDevice>,
}
static POLICY: OnceLock<DevicePolicy> = OnceLock::new();
pub fn install(policy: DevicePolicy) -> Result<(), DevicePolicy> {
    POLICY.set(policy)
}
pub fn backend_override() -> Option<Backends> {
    if let Some(p) = POLICY.get() {
        return (p.backend)();
    }
    let raw = std::env::var("ETALON_WGPU_BACKEND").ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "vulkan" | "vk" => Some(Backends::VULKAN),
        "dx12" | "d3d12" => Some(Backends::DX12),
        "metal" => Some(Backends::METAL),
        "gl" | "gles" => Some(Backends::GL),
        "primary" => Some(Backends::PRIMARY),
        "all" => Some(Backends::all()),
        _ => None,
    }
}
pub fn experimental_features_allowed() -> bool {
    if let Some(p) = POLICY.get() {
        return (p.experimental)();
    }
    matches!(
        std::env::var("ETALON_WGPU_EXPERIMENTAL_FEATURES")
            .ok()
            .as_deref(),
        Some("1" | "true" | "yes")
    )
}
pub fn shared_device() -> Option<SharedDevice> {
    (POLICY.get()?.shared)()
}
