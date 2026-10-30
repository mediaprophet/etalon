#![allow(dead_code)]
use std::pin::Pin;
use std::sync::Arc;

use etalon::custom::{
    AdapterInterface, ComputePipelineInterface, DeviceInterface, DispatchAdapter, DispatchBlas,
    DispatchDevice, DispatchQueue, DispatchShaderModule, DispatchSurface, InstanceInterface,
    QueueInterface, RequestAdapterFuture, ShaderModuleInterface,
};

#[derive(Debug, Clone)]
pub struct Counter(Arc<()>);

impl Counter {
    pub fn new() -> Self {
        Self(Arc::new(()))
    }

    pub fn count(&self) -> usize {
        Arc::strong_count(&self.0)
    }
}

#[derive(Debug)]
pub struct CustomInstance(pub Counter);

impl InstanceInterface for CustomInstance {
    fn new(__desc: etalon::InstanceDescriptor) -> Self
    where
        Self: Sized,
    {
        Self(Counter::new())
    }

    unsafe fn create_surface(
        &self,
        _target: etalon::SurfaceTargetUnsafe,
    ) -> Result<DispatchSurface, etalon::CreateSurfaceError> {
        unimplemented!()
    }

    fn request_adapter(
        &self,
        _options: &etalon::RequestAdapterOptions<'_, '_>,
    ) -> std::pin::Pin<Box<dyn RequestAdapterFuture>> {
        Box::pin(std::future::ready(Ok(DispatchAdapter::custom(
            CustomAdapter(self.0.clone()),
        ))))
    }

    fn poll_all_devices(&self, _force_wait: bool) -> bool {
        unimplemented!()
    }

    fn wgsl_language_features(&self) -> etalon::WgslLanguageFeatures {
        unimplemented!()
    }

    fn enumerate_adapters(
        &self,
        _backends: etalon::Backends,
    ) -> Pin<Box<dyn etalon::custom::EnumerateAdapterFuture>> {
        unimplemented!()
    }
}

#[derive(Debug)]
struct CustomAdapter(Counter);

impl AdapterInterface for CustomAdapter {
    fn request_device(
        &self,
        desc: &etalon::DeviceDescriptor<'_>,
    ) -> Pin<Box<dyn etalon::custom::RequestDeviceFuture>> {
        assert_eq!(desc.label, Some("device"));
        let res: Result<_, etalon::RequestDeviceError> = Ok((
            DispatchDevice::custom(CustomDevice(self.0.clone())),
            DispatchQueue::custom(CustomQueue(self.0.clone())),
        ));
        Box::pin(std::future::ready(res))
    }

    fn cooperative_matrix_properties(&self) -> Vec<etalon::CooperativeMatrixProperties> {
        Vec::new()
    }

    fn is_surface_supported(&self, _surface: &DispatchSurface) -> bool {
        unimplemented!()
    }

    fn features(&self) -> etalon::Features {
        unimplemented!()
    }

    fn limits(&self) -> etalon::Limits {
        unimplemented!()
    }

    fn downlevel_capabilities(&self) -> etalon::DownlevelCapabilities {
        unimplemented!()
    }

    fn get_info(&self) -> etalon::AdapterInfo {
        unimplemented!()
    }

    fn get_texture_format_features(
        &self,
        _format: etalon::TextureFormat,
    ) -> etalon::TextureFormatFeatures {
        unimplemented!()
    }

    fn get_presentation_timestamp(&self) -> etalon::PresentationTimestamp {
        unimplemented!()
    }
}

#[derive(Debug)]
struct CustomDevice(Counter);

impl DeviceInterface for CustomDevice {
    fn features(&self) -> etalon::Features {
        unimplemented!()
    }

    fn limits(&self) -> etalon::Limits {
        unimplemented!()
    }

    fn adapter_info(&self) -> etalon::AdapterInfo {
        unimplemented!()
    }

    fn create_shader_module(
        &self,
        desc: etalon::ShaderModuleDescriptor<'_>,
        _shader_bound_checks: etalon::ShaderRuntimeChecks,
    ) -> DispatchShaderModule {
        assert_eq!(desc.label, Some("shader"));
        DispatchShaderModule::custom(CustomShaderModule(self.0.clone()))
    }

    unsafe fn create_shader_module_passthrough(
        &self,
        _desc: &etalon::ShaderModuleDescriptorPassthrough<'_>,
    ) -> DispatchShaderModule {
        unimplemented!()
    }

    fn create_bind_group_layout(
        &self,
        _desc: &etalon::BindGroupLayoutDescriptor<'_>,
    ) -> etalon::custom::DispatchBindGroupLayout {
        unimplemented!()
    }

    fn create_bind_group(
        &self,
        _desc: &etalon::BindGroupDescriptor<'_>,
    ) -> etalon::custom::DispatchBindGroup {
        unimplemented!()
    }

    fn create_pipeline_layout(
        &self,
        _desc: &etalon::PipelineLayoutDescriptor<'_>,
    ) -> etalon::custom::DispatchPipelineLayout {
        unimplemented!()
    }

    fn create_render_pipeline(
        &self,
        _desc: &etalon::RenderPipelineDescriptor<'_>,
    ) -> etalon::custom::DispatchRenderPipeline {
        unimplemented!()
    }

    fn create_mesh_pipeline(
        &self,
        _desc: &etalon::MeshPipelineDescriptor<'_>,
    ) -> etalon::custom::DispatchRenderPipeline {
        unimplemented!()
    }

    fn create_compute_pipeline(
        &self,
        desc: &etalon::ComputePipelineDescriptor<'_>,
    ) -> etalon::custom::DispatchComputePipeline {
        let module = desc.module.as_custom::<CustomShaderModule>().unwrap();
        etalon::custom::DispatchComputePipeline::custom(CustomComputePipeline(module.0.clone()))
    }

    unsafe fn create_pipeline_cache(
        &self,
        _desc: &etalon::PipelineCacheDescriptor<'_>,
    ) -> etalon::custom::DispatchPipelineCache {
        unimplemented!()
    }

    fn create_buffer(
        &self,
        _desc: &etalon::BufferDescriptor<'_>,
    ) -> etalon::custom::DispatchBuffer {
        unimplemented!()
    }

    fn create_texture(
        &self,
        _desc: &etalon::TextureDescriptor<'_>,
    ) -> etalon::custom::DispatchTexture {
        unimplemented!()
    }

    fn create_external_texture(
        &self,
        _desc: &etalon::ExternalTextureDescriptor<'_>,
        _planes: &[&etalon::TextureView],
    ) -> etalon::custom::DispatchExternalTexture {
        unimplemented!()
    }

    fn create_blas(
        &self,
        _desc: &etalon::CreateBlasDescriptor<'_>,
        _sizes: etalon::BlasGeometrySizeDescriptors,
    ) -> (Option<u64>, etalon::custom::DispatchBlas) {
        unimplemented!()
    }

    fn create_tlas(
        &self,
        _desc: &etalon::CreateTlasDescriptor<'_>,
    ) -> etalon::custom::DispatchTlas {
        unimplemented!()
    }

    fn create_sampler(
        &self,
        _desc: &etalon::SamplerDescriptor<'_>,
    ) -> etalon::custom::DispatchSampler {
        unimplemented!()
    }

    fn create_query_set(
        &self,
        _desc: &etalon::QuerySetDescriptor<'_>,
    ) -> etalon::custom::DispatchQuerySet {
        unimplemented!()
    }

    fn create_command_encoder(
        &self,
        _desc: &etalon::CommandEncoderDescriptor<'_>,
    ) -> etalon::custom::DispatchCommandEncoder {
        unimplemented!()
    }

    fn create_render_bundle_encoder(
        &self,
        _desc: &etalon::RenderBundleEncoderDescriptor<'_>,
    ) -> etalon::custom::DispatchRenderBundleEncoder {
        unimplemented!()
    }

    fn set_device_lost_callback(
        &self,
        _device_lost_callback: etalon::custom::BoxDeviceLostCallback,
    ) {
        unimplemented!()
    }

    fn on_uncaptured_error(&self, _handler: Arc<dyn etalon::UncapturedErrorHandler>) {
        unimplemented!()
    }

    fn push_error_scope(&self, _filter: etalon::ErrorFilter) -> u32 {
        unimplemented!()
    }

    fn pop_error_scope(&self, _index: u32) -> Pin<Box<dyn etalon::custom::PopErrorScopeFuture>> {
        unimplemented!()
    }

    unsafe fn start_graphics_debugger_capture(&self) {
        unimplemented!()
    }

    unsafe fn stop_graphics_debugger_capture(&self) {
        unimplemented!()
    }

    fn poll(
        &self,
        _maintain: etalon::wgt::PollType<u64>,
    ) -> Result<etalon::PollStatus, etalon::PollError> {
        unimplemented!()
    }

    fn get_internal_counters(&self) -> etalon::InternalCounters {
        unimplemented!()
    }

    fn generate_allocator_report(&self) -> Option<etalon::AllocatorReport> {
        unimplemented!()
    }

    fn destroy(&self) {
        unimplemented!()
    }
}

#[derive(Debug)]
pub struct CustomShaderModule(pub Counter);

impl ShaderModuleInterface for CustomShaderModule {
    fn get_compilation_info(&self) -> Pin<Box<dyn etalon::custom::ShaderCompilationInfoFuture>> {
        unimplemented!()
    }
}

#[derive(Debug)]
struct CustomQueue(Counter);

impl QueueInterface for CustomQueue {
    fn write_buffer(
        &self,
        _buffer: &etalon::custom::DispatchBuffer,
        _offset: etalon::BufferAddress,
        _data: &[u8],
    ) {
        unimplemented!()
    }

    fn create_staging_buffer(
        &self,
        _size: etalon::BufferSize,
    ) -> Option<etalon::custom::DispatchQueueWriteBuffer> {
        unimplemented!()
    }

    fn validate_write_buffer(
        &self,
        _buffer: &etalon::custom::DispatchBuffer,
        _offset: etalon::BufferAddress,
        _size: etalon::BufferSize,
    ) -> Option<()> {
        unimplemented!()
    }

    fn write_staging_buffer(
        &self,
        _buffer: &etalon::custom::DispatchBuffer,
        _offset: etalon::BufferAddress,
        _staging_buffer: etalon::custom::DispatchQueueWriteBuffer,
    ) {
        unimplemented!()
    }

    fn write_texture(
        &self,
        _texture: etalon::TexelCopyTextureInfo<'_>,
        _data: &[u8],
        _data_layout: etalon::TexelCopyBufferLayout,
        _size: etalon::Extent3d,
    ) {
        unimplemented!()
    }

    fn submit(
        &self,
        _command_buffers: &mut dyn Iterator<Item = etalon::custom::DispatchCommandBuffer>,
    ) -> Option<u64> {
        unimplemented!()
    }

    fn get_timestamp_period(&self) -> f32 {
        unimplemented!()
    }

    fn on_submitted_work_done(&self, _callback: etalon::custom::BoxSubmittedWorkDoneCallback) {
        unimplemented!()
    }

    #[cfg(all(target_arch = "wasm32", feature = "web"))]
    fn copy_external_image_to_texture(
        &self,
        _source: &etalon::CopyExternalImageSourceInfo,
        _dest: etalon::CopyExternalImageDestInfo<&etalon::Texture>,
        _size: etalon::Extent3d,
    ) {
        unimplemented!()
    }

    fn compact_blas(&self, _blas: &DispatchBlas) -> (Option<u64>, DispatchBlas) {
        unimplemented!()
    }

    fn present(&self, _detail: &etalon::custom::DispatchSurfaceOutputDetail) {
        unimplemented!()
    }
}

#[derive(Debug)]
pub struct CustomComputePipeline(pub Counter);

impl ComputePipelineInterface for CustomComputePipeline {
    fn get_bind_group_layout(&self, _index: u32) -> etalon::custom::DispatchBindGroupLayout {
        unimplemented!()
    }
}
