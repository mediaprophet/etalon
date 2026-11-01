use super::*;

impl WebSurface {
    pub(crate) async fn get_capabilities_async(
        &self,
        adapter: &WebAdapter,
        device: &WebDevice,
    ) -> Result<wgt::SurfaceCapabilities, crate::SurfaceCapabilitiesError> {
        if device.adapter_ident != adapter.ident {
            return Err(crate::SurfaceCapabilitiesError::DeviceMismatch);
        }
        let supported = match device.rgba16float_canvas_support.get() {
            Some(&supported) => supported,
            None => {
                let supported = probe_rgba16float(&device.inner).await?;
                let _ = device.rgba16float_canvas_support.set(supported);
                supported
            }
        };
        Ok(self.canvas_capabilities(supported))
    }
}

fn probe_failed(message: impl Into<String>) -> crate::SurfaceCapabilitiesError {
    crate::SurfaceCapabilitiesError::ProbeFailed {
        message: message.into(),
    }
}

async fn probe_rgba16float(
    device: &webgpu_sys::GpuDevice,
) -> Result<bool, crate::SurfaceCapabilitiesError> {
    let canvas = web_sys::OffscreenCanvas::new(1, 1)
        .map_err(|error| probe_failed(format!("Cannot create probe canvas: {error:?}")))?;
    let context = canvas
        .get_context("webgpu")
        .map_err(|error| probe_failed(format!("Cannot create probe context: {error:?}")))?
        .ok_or_else(|| probe_failed("Probe canvas has no WebGPU context"))?;
    let context: webgpu_sys::GpuCanvasContext = context.unchecked_into();
    let config = webgpu_sys::GpuCanvasConfiguration::new(
        device,
        map_texture_format(wgt::TextureFormat::Rgba16Float),
    );
    device.push_error_scope(webgpu_sys::GpuErrorFilter::Validation);
    device.push_error_scope(webgpu_sys::GpuErrorFilter::Internal);
    device.push_error_scope(webgpu_sys::GpuErrorFilter::OutOfMemory);
    let configured = context.configure(&config).is_ok();
    // Configuration validation must not acquire a presentation texture.
    context.unconfigure();

    // Pop every scope before yielding, and attach rejection handlers even if cancelled.
    let memory = wasm_bindgen_futures::JsFuture::from(device.pop_error_scope());
    let internal = wasm_bindgen_futures::JsFuture::from(device.pop_error_scope());
    let validation = wasm_bindgen_futures::JsFuture::from(device.pop_error_scope());
    let memory = memory.await;
    let internal = internal.await;
    let validation = validation.await;
    for (label, result) in [("out-of-memory", memory), ("internal", internal)] {
        let error = result.map_err(|error| {
            probe_failed(format!("Canvas probe {label} scope rejected: {error:?}"))
        })?;
        if let Some(error) = error.into_option() {
            return Err(probe_failed(format!(
                "Canvas probe {label} error: {}",
                error.message()
            )));
        }
    }
    let validation = validation.map_err(|error| {
        probe_failed(format!("Canvas probe validation scope rejected: {error:?}"))
    })?;
    Ok(configured && validation.into_option().is_none())
}
