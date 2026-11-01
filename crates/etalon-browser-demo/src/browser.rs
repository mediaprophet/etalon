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
    (
        "nquin-gpu-radix",
        "NQuin GPU stable radix sort and cancellation",
        true,
    ),
    (
        "nquin-gpu-sieve",
        "NQuin GPU field sieve and tail cases",
        true,
    ),
    (
        "tensor-gpu-topk",
        "Tensor GPU top-k, ties and invalid lanes",
        true,
    ),
    (
        "tensor-gpu-attention",
        "Tensor GPU fused attention and all-masked rows",
        true,
    ),
    (
        "tensor-forward-readback",
        "Forward arena budgets and async token readback",
        true,
    ),
    (
        "browser-canvas-capabilities",
        "Isolated async canvas capabilities",
        true,
    ),
    (
        "agent-gpu-lease",
        "Queue completion and retained GPU buffer leases",
        true,
    ),
];

#[derive(Default, Debug)]
pub(crate) struct Health {
    pub(crate) lost: Option<String>,
    pub(crate) uncaptured: Option<String>,
}

pub(crate) struct Gpu {
    pub(crate) instance: etalon::Instance,
    pub(crate) adapter: etalon::Adapter,
    pub(crate) device: etalon::Device,
    pub(crate) queue: etalon::Queue,
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
    create_session_for_profile("core".into()).await
}

#[wasm_bindgen]
pub async fn create_session_for_profile(profile: String) -> Session {
    let health = Arc::new(Mutex::new(Health::default()));
    let (gpu, gpu_error, mut info) = match crate::adapter::request_gpu(&health, &profile).await {
        Ok((gpu, info)) => (Some(gpu), None, info),
        Err(error) => (
            None,
            Some(error.clone()),
            json!({
                "gpu_status": error["status"], "gpu_reason": error,
            }),
        ),
    };
    info["requested_feature_level"] = json!(profile);
    info["source_revision"] =
        json!(option_env!("ETALON_DEMO_REVISION").unwrap_or("local-unidentified"));
    info["execution"] = json!("Etalon Rust/WASM; GPU cases delegate to browser WebGPU; two portable cases execute on WASM CPU; domain kernels execute on GPU");
    info["bounds"] = json!({"cases": CASES.len(), "max_buffer_bytes": 32768, "image_width": 64, "image_height": 64});
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
            let mut error = self
                .gpu_error
                .clone()
                .unwrap_or_else(|| failure("gpu-unavailable", "No GPU session"));
            if let Some(fields) = error.as_object_mut() {
                fields.remove("adapter_attempts");
                fields.insert("full_evidence".into(), json!("runtime.gpu_reason"));
            }
            Err(error)
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
        let Gpu { device, queue, .. } = gpu;
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
            "nquin-gpu-radix" => crate::domain_nquin::radix(device, queue).await,
            "nquin-gpu-sieve" => crate::domain_nquin::sieve(device, queue).await,
            "tensor-gpu-topk" => crate::domain_tensor::topk(device, queue).await,
            "tensor-gpu-attention" => crate::domain_tensor::attention(device, queue).await,
            "tensor-forward-readback" => crate::domain_tensor::forward(device, queue).await,
            "browser-canvas-capabilities" => crate::backend_contracts::canvas(gpu).await,
            "agent-gpu-lease" => crate::domain_agent::run(device, queue).await,
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

pub(crate) fn bounded(text: &str) -> String {
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
