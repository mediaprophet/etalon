use serde_json::{json, Value};
use wasm_bindgen::{prelude::*, JsCast};

use crate::browser::{bounded, failure, Gpu};

pub(crate) async fn canvas(gpu: &Gpu) -> Result<Value, Value> {
    let canvas = web_sys::OffscreenCanvas::new(1, 1)
        .map_err(|error| failure("canvas-create", &format!("{error:?}")))?;
    let surface = gpu
        .instance
        .create_surface(etalon::SurfaceTarget::OffscreenCanvas(canvas.clone()))
        .map_err(|error| failure("surface-create", &error.to_string()))?;
    let baseline = surface.get_capabilities(&gpu.adapter);
    let required = [
        etalon::TextureFormat::Rgba8Unorm,
        etalon::TextureFormat::Bgra8Unorm,
    ];
    if !required
        .iter()
        .all(|format| baseline.formats.contains(format))
        || baseline
            .formats
            .contains(&etalon::TextureFormat::Rgba16Float)
    {
        return Err(failure(
            "canvas-baseline",
            "Synchronous browser capabilities must be conservative",
        ));
    }
    let config = surface
        .get_default_config(&gpu.adapter, 1, 1)
        .ok_or_else(|| failure("canvas-config", "No baseline canvas configuration"))?;
    surface.configure(&gpu.device, &config);
    let before = canvas_configuration(&canvas)?;
    let expected_format = match config.format {
        etalon::TextureFormat::Rgba8Unorm => "rgba8unorm",
        etalon::TextureFormat::Bgra8Unorm => "bgra8unorm",
        _ => return Err(failure("canvas-config", "Unexpected baseline format")),
    };
    if before.1 != expected_format {
        return Err(failure(
            "canvas-config",
            "Browser did not retain the requested baseline format",
        ));
    }
    let verified = surface
        .get_capabilities_async(&gpu.adapter, &gpu.device)
        .await
        .map_err(|error| failure("canvas-probe", &error.to_string()))?;
    if !required
        .iter()
        .all(|format| verified.formats.contains(format))
    {
        return Err(failure(
            "canvas-formats",
            "Async capabilities lost a baseline format",
        ));
    }
    let cached = surface
        .get_capabilities_async(&gpu.adapter, &gpu.device)
        .await
        .map_err(|error| failure("canvas-cache", &error.to_string()))?;
    if cached.formats != verified.formats
        || surface.get_capabilities(&gpu.adapter).formats != baseline.formats
    {
        return Err(failure(
            "canvas-cache",
            "Device probe changed conservative or cached capabilities",
        ));
    }
    let after = canvas_configuration(&canvas)?;
    if before != after {
        return Err(failure(
            "canvas-preservation",
            "Capability validation changed the configured canvas device or format",
        ));
    }

    // A second handle to the same hardware must not inherit this device's probe cache.
    let other = gpu
        .instance
        .request_webgpu_adapter(
            &etalon::RequestAdapterOptions::default(),
            etalon::FeatureLevel::Compatibility,
        )
        .await
        .map_err(|error| failure("canvas-other-adapter", &error.to_string()))?;
    if !matches!(
        surface.get_capabilities_async(&other, &gpu.device).await,
        Err(etalon::SurfaceCapabilitiesError::DeviceMismatch)
    ) {
        return Err(failure(
            "canvas-device-mismatch",
            "Canvas query accepted a device from another adapter handle",
        ));
    }
    let roundtrip = crate::readback::buffer_case(&gpu.device, &gpu.queue, false).await?;
    Ok(json!({
        "baseline_formats": baseline.formats.iter().map(|format| format!("{format:?}")).collect::<Vec<_>>(),
        "verified_formats": verified.formats.iter().map(|format| format!("{format:?}")).collect::<Vec<_>>(),
        "rgba16float_configuration_accepted": verified.formats.contains(&etalon::TextureFormat::Rgba16Float),
        "configured_surface_preserved": true, "device_mismatch_rejected": true,
        "presentation_tested": false, "presentation_texture_acquired": false,
        "post_query_gpu_roundtrip": roundtrip,
        "message": "Canvas configuration and device/format preservation verified; GPU buffer readback remains healthy. Presentation is not exercised."
    }))
}

fn canvas_configuration(canvas: &web_sys::OffscreenCanvas) -> Result<(JsValue, String), Value> {
    let context = canvas
        .get_context("webgpu")
        .map_err(|error| failure("canvas-context", &format!("{error:?}")))?
        .ok_or_else(|| failure("canvas-context", "WebGPU canvas context unavailable"))?;
    let get_configuration = js_sys::Reflect::get(&context, &JsValue::from_str("getConfiguration"))
        .and_then(|value| value.dyn_into::<js_sys::Function>())
        .map_err(|error| failure("canvas-configuration-query", &format!("{error:?}")))?;
    let config = get_configuration
        .call0(&context)
        .map_err(|error| failure("canvas-configuration-query", &format!("{error:?}")))?;
    let device = js_sys::Reflect::get(&config, &JsValue::from_str("device"))
        .map_err(|error| failure("canvas-configuration-device", &format!("{error:?}")))?;
    let format = js_sys::Reflect::get(&config, &JsValue::from_str("format"))
        .ok()
        .and_then(|value| value.as_string())
        .ok_or_else(|| failure("canvas-configuration-format", "Missing canvas format"))?;
    if !device.is_object() {
        return Err(failure(
            "canvas-configuration-device",
            "Missing configured device",
        ));
    }
    Ok((device, format))
}

/// Browser-boundary fixture entry point; assertions execute through the Rust backend.
#[wasm_bindgen]
pub async fn inspect_adapter_contract(mode: String) -> Result<String, JsValue> {
    inspect(&mode)
        .await
        .map(|value| value.to_string())
        .map_err(|error| JsValue::from_str(&error.to_string()))
}

async fn inspect(mode: &str) -> Result<Value, Value> {
    if !matches!(mode, "missing" | "zero" | "description") {
        return Err(failure("contract-mode", "Unknown adapter fixture"));
    }
    let instance = etalon::Instance::new(etalon::InstanceDescriptor {
        backends: etalon::Backends::BROWSER_WEBGPU,
        ..etalon::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = instance
        .request_webgpu_adapter(
            &etalon::RequestAdapterOptions::default(),
            etalon::FeatureLevel::Core,
        )
        .await
        .map_err(|error| failure("contract-adapter", &error.to_string()))?;
    let limits = adapter.limits();
    let actual = [
        limits.max_storage_buffers_in_vertex_stage,
        limits.max_storage_buffers_in_fragment_stage,
        limits.max_storage_textures_in_vertex_stage,
        limits.max_storage_textures_in_fragment_stage,
        limits.max_bind_groups_plus_vertex_buffers,
        limits.max_color_attachment_bytes_per_sample,
    ];
    let expected = if mode == "zero" {
        [0; 6]
    } else {
        [
            limits.max_storage_buffers_per_shader_stage,
            limits.max_storage_buffers_per_shader_stage,
            limits.max_storage_textures_per_shader_stage,
            limits.max_storage_textures_per_shader_stage,
            24,
            32,
        ]
    };
    if actual != expected {
        return Err(failure(
            "contract-limit-mapping",
            &format!("Mapped {actual:?}; expected {expected:?}"),
        ));
    }
    let info = adapter.get_info();
    let expected_name = if mode == "description" {
        "fixture-description"
    } else {
        "fixture-vendor fixture-architecture fixture-device"
    };
    if info.name != expected_name || info.vendor != 0 || info.device != 0 ||
        info.transient_saves_memory.is_some() || !info.driver.is_empty() ||
        info.driver_info != "WebGPU vendor=fixture-vendor; architecture=fixture-architecture; device=fixture-device" ||
        info.device_type != etalon::DeviceType::Other {
        return Err(failure("contract-adapter-info", &format!("Unexpected mapped identity: {info:?}")));
    }
    let mut required = limits.clone();
    required.max_buffer_size = limits.max_buffer_size.checked_add(1).ok_or_else(|| {
        failure(
            "contract-limit-overflow",
            "Cannot construct oversized buffer limit",
        )
    })?;
    let rejected = match adapter
        .request_device(&etalon::DeviceDescriptor {
            label: Some("intentionally oversized adapter contract"),
            required_limits: required,
            ..Default::default()
        })
        .await
    {
        Ok((device, _)) => {
            device.destroy();
            return Err(failure(
                "contract-limit-accepted",
                "Oversized explicit required limit was accepted",
            ));
        }
        Err(error) => error.to_string(),
    };
    if !rejected.contains("max_buffer_size")
        || !rejected.contains("incompatible with adapter limit")
    {
        return Err(failure("contract-limit-rejection", &rejected));
    }
    Ok(json!({"mode": mode, "mapped_optional_limits": actual,
        "identity_name": info.name, "identity_details": info.driver_info,
        "numeric_vendor": info.vendor, "numeric_device": info.device,
        "transient_saves_memory": info.transient_saves_memory,
        "oversized_limit_rejected": true, "device_error": bounded(&rejected)}))
}
