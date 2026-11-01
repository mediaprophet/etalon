use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

use crate::{compute, extensions, readback, render};

const CASES: &[(&str, &str, bool)] = &[
    (
        "nquin-wire-sieve-sort",
        "NQuin ABI, parity, sieve and stable sort (WASM CPU)",
        false,
    ),
    (
        "forge-determinism",
        "WGSL Forge deterministic emission and Naga validation (WASM CPU)",
        false,
    ),
    ("empty-error-scope", "Empty validation error scope", true),
    ("buffer-roundtrip", "Buffer upload and readback", true),
    ("buffer-zero-init", "Unwritten buffer reads as zero", true),
    ("compute-1", "Vector compute: 1 element", true),
    ("compute-63", "Vector compute: 63 elements", true),
    ("compute-64", "Vector compute: 64 elements", true),
    ("compute-65", "Vector compute: 65 elements", true),
    ("compute-257", "Vector compute: 257 elements", true),
    (
        "compute-changed-inputs",
        "Vector compute after changing inputs",
        true,
    ),
    (
        "validation-negative",
        "Invalid buffer usage is rejected",
        true,
    ),
    (
        "render-readback",
        "Offscreen triangle and pixel assertions",
        true,
    ),
];

#[derive(Default, Debug)]
struct Health {
    lost: Option<String>,
    uncaptured: Option<String>,
}

struct Gpu {
    device: etalon::Device,
    queue: etalon::Queue,
}

#[wasm_bindgen]
pub struct Session {
    gpu: Option<Gpu>,
    gpu_error: Option<Value>,
    info: String,
    health: Arc<Mutex<Health>>,
    running: std::cell::Cell<bool>,
    destroyed: std::cell::Cell<bool>,
    image: std::cell::RefCell<Vec<u8>>,
}

#[wasm_bindgen]
pub async fn create_session() -> Session {
    let health = Arc::new(Mutex::new(Health::default()));
    let (gpu, gpu_error, mut info) = match request_gpu(&health).await {
        Ok((gpu, info)) => (Some(gpu), None, info),
        Err(error) => (
            None,
            Some(error.clone()),
            json!({
                "gpu_status": error["status"], "gpu_reason": error,
            }),
        ),
    };
    info["source_revision"] =
        json!(option_env!("ETALON_DEMO_REVISION").unwrap_or("local-unidentified"));
    info["execution"] = json!("Etalon Rust/WASM; GPU cases delegate to browser WebGPU; extension cases execute on WASM CPU");
    info["bounds"] = json!({"cases": CASES.len(), "max_buffer_bytes": 16384, "image_width": 64, "image_height": 64});
    Session {
        gpu,
        gpu_error,
        info: info.to_string(),
        health,
        running: std::cell::Cell::new(false),
        destroyed: std::cell::Cell::new(false),
        image: std::cell::RefCell::new(Vec::new()),
    }
}

async fn request_gpu(health: &Arc<Mutex<Health>>) -> Result<(Gpu, Value), Value> {
    let navigator = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("navigator"))
        .unwrap_or(JsValue::UNDEFINED);
    let exposed_gpu =
        js_sys::Reflect::get(&navigator, &JsValue::from_str("gpu")).unwrap_or(JsValue::UNDEFINED);
    if exposed_gpu.is_null() || exposed_gpu.is_undefined() {
        return Err(json!({
            "status": "unsupported", "code": "adapter-unavailable",
            "message": "navigator.gpu is unavailable in this browser context",
        }));
    }
    let instance = etalon::Instance::new(etalon::InstanceDescriptor {
        backends: etalon::Backends::BROWSER_WEBGPU,
        ..etalon::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = instance.request_adapter(&etalon::RequestAdapterOptions {
        power_preference: etalon::PowerPreference::LowPower, ..Default::default()
    }).await.map_err(|e| json!({"status": "unsupported", "code": "adapter-unavailable", "message": bounded(&e.to_string())}))?;
    let adapter_info = adapter.get_info();
    let (device, queue) = adapter.request_device(&etalon::DeviceDescriptor {
        label: Some("Etalon browser self-test"),
        required_features: etalon::Features::empty(),
        required_limits: etalon::Limits::downlevel_defaults(),
        ..Default::default()
    }).await.map_err(|e| json!({"status": "fail", "code": "device-request-failed", "message": bounded(&e.to_string())}))?;
    let lost_health = health.clone();
    device.set_device_lost_callback(move |reason, message| {
        if let Ok(mut state) = lost_health.lock() {
            state.lost = Some(bounded(&format!("{reason:?}: {message}")));
        }
    });
    let error_health = health.clone();
    device.on_uncaptured_error(Arc::new(move |error: etalon::Error| {
        if let Ok(mut state) = error_health.lock() {
            state.uncaptured = Some(bounded(&error.to_string()));
        }
    }));
    let info = json!({
        "gpu_status": "available", "backend": format!("{:?}", adapter_info.backend),
        "adapter": bounded(&adapter_info.name), "vendor": adapter_info.vendor, "device": adapter_info.device,
        "device_type": format!("{:?}", adapter_info.device_type),
        "driver": bounded(&adapter_info.driver), "driver_info": bounded(&adapter_info.driver_info),
        "available_features": bounded(&format!("{:?}", adapter.features())),
        "enabled_features": format!("{:?}", device.features()),
        "adapter_limits": limits_json(&adapter.limits()), "device_limits": limits_json(&device.limits()),
    });
    Ok((Gpu { device, queue }, info))
}

#[wasm_bindgen]
impl Session {
    pub fn info_json(&self) -> String {
        self.info.clone()
    }

    pub fn cases_json(&self) -> String {
        json!(CASES
            .iter()
            .map(|(id, label, gpu)| json!({"id": id, "label": label, "requires_gpu": gpu}))
            .collect::<Vec<_>>())
        .to_string()
    }

    pub fn image_rgba(&self) -> Vec<u8> {
        self.image.borrow().clone()
    }

    pub fn destroy(&self) {
        if !self.destroyed.replace(true) {
            if let Some(gpu) = &self.gpu {
                gpu.device.destroy();
            }
            self.image.borrow_mut().clear();
        }
    }

    pub async fn run_case(&self, id: String) -> String {
        let started = js_sys::Date::now();
        if self.running.replace(true) {
            return result(
                &id,
                started,
                Err(failure("session-busy", "Run cases sequentially")),
            );
        }
        let outcome = if self.destroyed.get() {
            Err(failure("session-destroyed", "Create a new session"))
        } else if !CASES.iter().any(|(known, _, _)| *known == id) {
            Err(failure("unknown-case", "Unknown case identifier"))
        } else if id == "nquin-wire-sieve-sort" {
            extensions::nquin()
        } else if id == "forge-determinism" {
            extensions::forge()
        } else if let Some(error) = self.health_error() {
            Err(error)
        } else if let Some(gpu) = &self.gpu {
            let scope = gpu.device.push_error_scope(etalon::ErrorFilter::Validation);
            let outcome = self.execute(&id, gpu).await;
            match scope.pop().await {
                Some(error) => Err(failure("unexpected-validation", &error.to_string())),
                None => self.health_error().map_or(outcome, Err),
            }
        } else {
            Err(self
                .gpu_error
                .clone()
                .unwrap_or_else(|| failure("gpu-unavailable", "No GPU session")))
        };
        self.running.set(false);
        result(&id, started, outcome)
    }
}

impl Session {
    fn health_error(&self) -> Option<Value> {
        let state = self.health.lock().ok()?;
        state
            .lost
            .as_ref()
            .map(|s| failure("device-lost", s))
            .or_else(|| {
                state
                    .uncaptured
                    .as_ref()
                    .map(|s| failure("uncaptured-error", s))
            })
    }

    async fn execute(&self, id: &str, gpu: &Gpu) -> Result<Value, Value> {
        let Gpu { device, queue } = gpu;
        match id {
            "empty-error-scope" => {
                let scope = device.push_error_scope(etalon::ErrorFilter::Validation);
                match scope.pop().await {
                    None => Ok(json!({"message": "Empty error scope resolved to None"})),
                    Some(error) => Err(failure("nonempty-scope", &error.to_string())),
                }
            }
            "buffer-roundtrip" => readback::buffer_case(device, queue, false).await,
            "buffer-zero-init" => readback::buffer_case(device, queue, true).await,
            "validation-negative" => {
                let scope = device.push_error_scope(etalon::ErrorFilter::Validation);
                let invalid = device.create_buffer(&etalon::BufferDescriptor {
                    label: Some("intentionally invalid buffer usage"),
                    size: 16,
                    usage: etalon::BufferUsages::MAP_READ | etalon::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                let error = scope.pop().await;
                drop(invalid);
                match error {
                    Some(etalon::Error::Validation { description, .. }) => {
                        Ok(json!({"expected_error": bounded(&description)}))
                    }
                    Some(other) => Err(failure("wrong-error-kind", &other.to_string())),
                    None => Err(failure(
                        "validation-not-rejected",
                        "Invalid MAP_READ | STORAGE buffer was accepted",
                    )),
                }
            }
            "render-readback" => {
                self.image.borrow_mut().clear();
                let pixels = render::run(device, queue).await?;
                *self.image.borrow_mut() = pixels;
                Ok(json!({"width": 64, "height": 64, "asserted_pixels": 4096,
                    "message": "Red triangle over blue clear; all pixel values and interior/exterior samples checked"}))
            }
            "compute-changed-inputs" => compute::changed_inputs(device, queue).await,
            _ => {
                let count = id
                    .strip_prefix("compute-")
                    .and_then(|s| s.parse::<u32>().ok())
                    .ok_or_else(|| failure("unknown-case", "Unknown case identifier"))?;
                compute::run(device, queue, count).await
            }
        }
    }
}

pub(crate) fn failure(code: &str, message: &str) -> Value {
    json!({"code": code, "message": bounded(message)})
}

fn bounded(text: &str) -> String {
    text.chars().take(1024).collect()
}

fn result(id: &str, started: f64, outcome: Result<Value, Value>) -> String {
    let (status, details) = match outcome {
        Ok(details) => ("pass", details),
        Err(details) => (
            if details["status"] == "unsupported" {
                "unsupported"
            } else {
                "fail"
            },
            details,
        ),
    };
    json!({"id": id, "status": status, "duration_ms": (js_sys::Date::now() - started).max(0.0), "details": details}).to_string()
}

fn limits_json(l: &etalon::Limits) -> Value {
    json!({
        "max_buffer_size": l.max_buffer_size,
        "max_storage_buffer_binding_size": l.max_storage_buffer_binding_size,
        "max_compute_workgroup_size_x": l.max_compute_workgroup_size_x,
        "max_compute_invocations_per_workgroup": l.max_compute_invocations_per_workgroup,
        "max_compute_workgroups_per_dimension": l.max_compute_workgroups_per_dimension,
        "max_texture_dimension_2d": l.max_texture_dimension_2d, "max_bind_groups": l.max_bind_groups,
    })
}
