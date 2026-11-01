//! Independent browser acquisition controls; these do not qualify GPU execution.

use js_sys::{Function, Object, Promise, Reflect};
use serde_json::{json, Value};
use wasm_bindgen::{prelude::*, JsCast};

use crate::browser::bounded;

fn property(value: &JsValue, key: &str) -> Result<JsValue, JsValue> {
    Reflect::get(value, &JsValue::from_str(key))
}

fn error_value(error: JsValue, phase: &str) -> Value {
    let name = property(&error, "name").ok().and_then(|v| v.as_string());
    let message = property(&error, "message")
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| format!("{error:?}"));
    json!({"status":phase,"name":name,"message":bounded(&message)})
}

#[wasm_bindgen]
pub async fn adapter_acquisition_control() -> String {
    let gpu = property(&js_sys::global(), "navigator").and_then(|n| property(&n, "gpu"));
    let gpu = match gpu {
        Ok(gpu) if !gpu.is_null() && !gpu.is_undefined() => gpu,
        Ok(_) => {
            return json!({"status":"gpu-unavailable","gpu_execution_tested":false}).to_string()
        }
        Err(error) => return error_value(error, "access-threw").to_string(),
    };
    let request = match property(&gpu, "requestAdapter").and_then(|v| v.dyn_into::<Function>()) {
        Ok(request) => request,
        Err(error) => return error_value(error, "method-unavailable").to_string(),
    };
    let mut attempts = Vec::with_capacity(4);
    for mode in [
        "no-arguments",
        "empty-options",
        "explicit-core",
        "explicit-compatibility",
    ] {
        let options = Object::new();
        let level = match mode {
            "explicit-core" => Some("core"),
            "explicit-compatibility" => Some("compatibility"),
            _ => None,
        };
        if let Some(level) = level {
            if let Err(error) = Reflect::set(
                &options,
                &JsValue::from_str("featureLevel"),
                &JsValue::from_str(level),
            ) {
                attempts.push(json!({"request":mode,"result":error_value(error,"options-threw")}));
                continue;
            }
        }
        let called = if mode == "no-arguments" {
            request.call0(&gpu)
        } else {
            request.call1(&gpu, &options)
        };
        let result = match called {
            Err(error) => error_value(error, "call-threw"),
            Ok(value) => match value.dyn_into::<Promise>() {
                Err(value) => error_value(value, "non-promise-result"),
                Ok(promise) => match wasm_bindgen_futures::JsFuture::from(promise).await {
                    Err(error) => error_value(error, "rejected"),
                    Ok(adapter) if adapter.is_null() => json!({"status":"null"}),
                    Ok(adapter) if adapter.is_undefined() => json!({"status":"undefined"}),
                    Ok(adapter) if !adapter.is_object() => {
                        json!({"status":"invalid-adapter-result"})
                    }
                    Ok(adapter) => {
                        let info = property(&adapter, "info").unwrap_or(JsValue::UNDEFINED);
                        let mut identity = serde_json::Map::new();
                        for key in ["vendor", "architecture", "device", "description"] {
                            if let Some(value) =
                                property(&info, key).ok().and_then(|v| v.as_string())
                            {
                                identity.insert(key.into(), json!(bounded(&value)));
                            }
                        }
                        json!({"status":"adapter-returned","adapter_info":identity})
                    }
                },
            },
        };
        attempts.push(json!({"request":mode,"result":result}));
    }
    json!({"execution":"Rust/WASM direct browser API control","gpu_execution_tested":false,"gpu_present":true,"attempts":attempts}).to_string()
}
