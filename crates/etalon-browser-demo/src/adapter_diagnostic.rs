use serde_json::json;
use wasm_bindgen::prelude::*;

const PROBE_TIMEOUT_MS: u32 = 1000;

/// Test only browser adapter acquisition, returning failed steps without unexecuted GPU cases.
#[wasm_bindgen]
pub async fn adapter_diagnostic_errors() -> String {
    let started = js_sys::Date::now();
    let control = crate::adapter_control::acquisition_control(Some(PROBE_TIMEOUT_MS)).await;
    let mut errors = Vec::with_capacity(5);
    let mut successful_requests = 0;
    let request_shapes = if let Some(attempts) = control["attempts"].as_array() {
        for attempt in attempts {
            if attempt["result"]["status"] == "adapter-returned" {
                successful_requests += 1;
            } else {
                let mut error = attempt["result"].clone();
                error["test_id"] = attempt["test_id"].clone();
                error["request_shape"] = attempt["request"].clone();
                errors.push(error);
            }
        }
        attempts.len()
    } else {
        let mut error = json!({"test_id": match control["stage"].as_str() {
            Some("navigator.lookup") => "adapter-navigator",
            Some("gpu.lookup") => "adapter-navigator-gpu",
            _ => "adapter-requestAdapter",
        }});
        for key in [
            "status",
            "stage",
            "api_path",
            "value_kind",
            "name",
            "message",
        ] {
            if let Some(value) = control.get(key) {
                error[key] = value.clone();
            }
        }
        errors.push(error);
        0
    };
    let elapsed = (js_sys::Date::now() - started).max(0.0);
    json!({"schema":"etalon.adapter-diagnostic.v1","schema_version":1,"diagnostic":"adapter-acquisition-errors",
        "automatic_upload":false,
        "source_revision":option_env!("ETALON_DEMO_REVISION").unwrap_or("local-unidentified"),
        "library_version":env!("CARGO_PKG_VERSION"),"elapsed_ms":elapsed,
        "gpu_execution_tested":false,"probe_timeout_ms":PROBE_TIMEOUT_MS,
        "executed":{"preflight_steps":control["preflight_steps"],"request_shapes":request_shapes,
            "adapter_requests":control["adapter_requests"],"successful_requests":successful_requests},
        "error_count":errors.len(),"errors":errors,
        "summary": if request_shapes == 0 { "preflight-failed" } else { "adapter-probes-completed" }}).to_string()
}
