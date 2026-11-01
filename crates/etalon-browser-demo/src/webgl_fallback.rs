use std::sync::{Arc, Mutex};

use js_sys::{Function, Reflect};
use serde_json::{json, Value};
use wasm_bindgen::{prelude::*, JsCast};

use crate::browser::{bounded, failure};

struct ContextOwner(web_sys::OffscreenCanvas);
impl Drop for ContextOwner {
    fn drop(&mut self) {
        let Ok(Some(context)) = self.0.get_context("webgl2") else {
            return;
        };
        let Ok(get) = Reflect::get(&context, &"getExtension".into()) else {
            return;
        };
        let Ok(get) = get.dyn_into::<Function>() else {
            return;
        };
        let Ok(extension) = get.call1(&context, &"WEBGL_lose_context".into()) else {
            return;
        };
        if let Ok(lose) = Reflect::get(&extension, &"loseContext".into()) {
            if let Some(lose) = lose.dyn_ref::<Function>() {
                let _ = lose.call0(&extension);
            }
        }
    }
}

struct DeviceOwner(etalon::Device);
impl Drop for DeviceOwner {
    fn drop(&mut self) {
        self.0.destroy();
    }
}

#[wasm_bindgen]
pub async fn webgl_render_fallback() -> String {
    let started = js_sys::Date::now();
    let mut report = match run().await {
        Ok(value) => value,
        Err(error) => json!({"status":error["status"],"details":error}),
    };
    report["schema"] = json!("etalon.webgl-fallback.v1");
    report["library_version"] = json!(env!("CARGO_PKG_VERSION"));
    report["source_revision"] =
        json!(option_env!("ETALON_DEMO_REVISION").unwrap_or("local-unidentified"));
    report["backend"] = json!("WebGL2");
    report["execution"] = json!("Etalon Rust/WASM");
    report["compute_supported"] = json!(false);
    report["webgpu_qualified"] = json!(false);
    report["duration_ms"] = json!(js_sys::Date::now() - started);
    report.to_string()
}

async fn run() -> Result<Value, Value> {
    let canvas = ContextOwner(web_sys::OffscreenCanvas::new(64, 64).map_err(|_| {
        failure(
            "webgl-canvas-failed",
            "Cannot create bounded offscreen canvas",
        )
    })?);
    let instance = etalon::Instance::new(etalon::InstanceDescriptor {
        backends: etalon::Backends::GL,
        ..etalon::InstanceDescriptor::new_without_display_handle()
    });
    let surface = instance.create_surface(etalon::SurfaceTarget::OffscreenCanvas(canvas.0.clone()))
        .map_err(|error| json!({"status":"unsupported","code":"webgl-surface-unavailable","message":bounded(&error.to_string())}))?;
    let adapter = instance.request_adapter(&etalon::RequestAdapterOptions {
        compatible_surface: Some(&surface),
        ..Default::default()
    }).await.map_err(|error| json!({"status":"unsupported","code":"webgl-adapter-unavailable","message":bounded(&error.to_string())}))?;
    let info = adapter.get_info();
    if info.backend != etalon::Backend::Gl {
        return Err(failure("webgl-backend-mismatch", "Expected the GL backend"));
    }
    let (device, queue) = adapter
        .request_device(&etalon::DeviceDescriptor {
            label: Some("bounded WebGL2 rendering fallback"),
            required_limits: etalon::Limits::downlevel_webgl2_defaults(),
            ..Default::default()
        })
        .await
        .map_err(|error| failure("webgl-device-failed", &bounded(&error.to_string())))?;
    let device = DeviceOwner(device);
    let health = Arc::new(Mutex::new(None));
    let errors = health.clone();
    device.0.on_uncaptured_error(Arc::new(move |error| {
        *errors.lock().unwrap() = Some(bounded(&error.to_string()));
    }));
    let lost = health.clone();
    device.0.set_device_lost_callback(move |reason, message| {
        *lost.lock().unwrap() = Some(bounded(&format!("{reason:?}: {message}")));
    });
    let scope = device.0.push_error_scope(etalon::ErrorFilter::Validation);
    let pixels = crate::render::run_webgl(&device.0, &queue).await;
    if let Some(error) = scope.pop().await {
        return Err(failure(
            "webgl-validation-failed",
            &bounded(&error.to_string()),
        ));
    }
    if let Some(error) = health.lock().unwrap().take() {
        return Err(failure("webgl-device-error", &error));
    }
    let pixels = pixels?;
    Ok(
        json!({"status":"pass","render_readback":true,"image_rgba":pixels,
        "details":{"adapter":bounded(&info.name),"driver":bounded(&info.driver),
        "driver_info":bounded(&info.driver_info),"compared_pixels":4096,"position_assertions":6,
        "width":64,"height":64,"readback_bytes":16384,
        "scope":"WGSL triangle rendering and pixel readback; no compute emulation"}}),
    )
}
