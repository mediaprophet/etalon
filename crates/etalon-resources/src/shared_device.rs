use crate::device_caps::{GpuAdapterCaps, GpuFeatureCaps};

#[derive(Debug)]
pub struct DeviceContext {
    pub lease: crate::device_lease::DeviceLease,
    pub device: etalon::Device,
    pub queue: etalon::Queue,
    pub instance: etalon::Instance,
    pub adapter: etalon::Adapter,
    pub adapter_caps: GpuAdapterCaps,
    pub enabled_features: GpuFeatureCaps,
    pub timestamps_supported: bool,
    pub timestamp_period_ns: f32,
}
impl DeviceContext {
    pub async fn request(
        instance: etalon::Instance,
        adapter: etalon::Adapter,
        descriptor: &etalon::DeviceDescriptor<'_>,
    ) -> Result<Self, String> {
        let adapter_caps = GpuAdapterCaps::from_adapter(&adapter);
        let (device, queue) = adapter
            .request_device(descriptor)
            .await
            .map_err(|e| e.to_string())?;
        let enabled_features = GpuFeatureCaps::from_features(device.features());
        let timestamps_supported = enabled_features.timestamp_query;
        let timestamp_period_ns = if timestamps_supported {
            queue.get_timestamp_period()
        } else {
            0.0
        };
        Ok(Self {
            lease: crate::device_lease::DeviceLease::new(&device, &queue)?,
            device,
            queue,
            instance,
            adapter,
            adapter_caps,
            enabled_features,
            timestamps_supported,
            timestamp_period_ns,
        })
    }
}
