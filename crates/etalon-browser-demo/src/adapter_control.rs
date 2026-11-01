//! Independent browser acquisition controls; these do not qualify GPU execution.

use js_sys::{Function, Object, Promise, Reflect};
use serde_json::{json, Value};
use wasm_bindgen::{prelude::*, JsCast};

use crate::browser::bounded;

fn property(value: &JsValue, key: &str) -> Result<JsValue, JsValue> {
    Reflect::get(value, &JsValue::from_str(key))
}

fn kind(value: &JsValue) -> &'static str {
    if value.is_null() {
        "null"
    } else if value.is_undefined() {
        "undefined"
    } else if value.is_object() {
        "object"
    } else {
        "primitive"
    }
}

fn error_value(error: JsValue, phase: &str) -> Value {
    let name = property(&error, "name")
        .ok()
        .and_then(|v| v.as_string())
        .map(|s| bounded(&s));
    let message = property(&error, "message")
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| format!("{error:?}"));
    json!({"status":phase,"name":name,"message":bounded(&message)})
}

fn step(mut value: Value, stage: &str, api_path: &str) -> Value {
    if value.get("stage").is_none() {
        value["stage"] = json!(stage);
    }
    if value.get("api_path").is_none() {
        value["api_path"] = json!(api_path);
    }
    value
}

fn early(result: Value, executed: u32) -> Value {
    let mut result = result;
    result["gpu_execution_tested"] = json!(false);
    result["preflight_steps"] = json!(executed);
    result["adapter_requests"] = json!(0);
    result
}

struct Timer {
    global: JsValue,
    clear: Function,
    id: JsValue,
}
impl Drop for Timer {
    fn drop(&mut self) {
        let _ = self.clear.call1(&self.global, &self.id);
    }
}

async fn await_adapter(promise: Promise, timeout_ms: Option<u32>) -> Result<JsValue, Value> {
    let Some(timeout_ms) = timeout_ms else {
        return wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|error| error_value(error, "rejected"));
    };
    let global: JsValue = js_sys::global().into();
    let set = property(&global, "setTimeout")
        .and_then(|v| v.dyn_into::<Function>())
        .map_err(|error| {
            step(
                error_value(error, "timeout-unavailable"),
                "timeout.lookup",
                "globalThis.setTimeout",
            )
        })?;
    let clear = property(&global, "clearTimeout")
        .and_then(|v| v.dyn_into::<Function>())
        .map_err(|error| {
            step(
                error_value(error, "timeout-unavailable"),
                "timeout.lookup",
                "globalThis.clearTimeout",
            )
        })?;
    let marker: JsValue = Object::new().into();
    let mut id = None;
    let mut setup_error = None;
    let timeout = Promise::new(&mut |resolve, _| {
        let callback = resolve.bind1(&JsValue::UNDEFINED, &marker);
        match set.call2(
            &global,
            callback.as_ref(),
            &JsValue::from_f64(f64::from(timeout_ms)),
        ) {
            Ok(value) => id = Some(value),
            Err(error) => setup_error = Some(error),
        }
    });
    if let Some(error) = setup_error {
        return Err(step(
            error_value(error, "timeout-setup-threw"),
            "timeout.call",
            "globalThis.setTimeout",
        ));
    }
    let _timer = Timer {
        global,
        clear,
        id: id.ok_or_else(|| json!({"status":"timeout-unavailable"}))?,
    };
    let promises = js_sys::Array::of2(promise.as_ref(), timeout.as_ref());
    let result = wasm_bindgen_futures::JsFuture::from(Promise::race(&promises))
        .await
        .map_err(|error| error_value(error, "rejected"))?;
    if Object::is(&result, &marker) {
        Err(json!({"status":"timeout","timeout_ms":timeout_ms}))
    } else {
        Ok(result)
    }
}

#[wasm_bindgen]
pub async fn adapter_acquisition_control() -> String {
    acquisition_control(None).await.to_string()
}

pub(crate) async fn acquisition_control(timeout_ms: Option<u32>) -> Value {
    let global: JsValue = js_sys::global().into();
    let navigator = match property(&global, "navigator") {
        Ok(value) if value.is_object() => value,
        Ok(value) => {
            return early(
                step(
                    json!({"status":"gpu-unavailable","value_kind":kind(&value)}),
                    "navigator.lookup",
                    "globalThis.navigator",
                ),
                1,
            )
        }
        Err(error) => {
            return early(
                step(
                    error_value(error, "access-threw"),
                    "navigator.lookup",
                    "globalThis.navigator",
                ),
                1,
            )
        }
    };
    let gpu = match property(&navigator, "gpu") {
        Ok(value) if value.is_object() => value,
        Ok(value) => {
            return early(
                step(
                    json!({"status":"gpu-unavailable","value_kind":kind(&value)}),
                    "gpu.lookup",
                    "navigator.gpu",
                ),
                2,
            )
        }
        Err(error) => {
            return early(
                step(
                    error_value(error, "access-threw"),
                    "gpu.lookup",
                    "navigator.gpu",
                ),
                2,
            )
        }
    };
    let request = match property(&gpu, "requestAdapter") {
        Ok(value) => match value.clone().dyn_into::<Function>() {
            Ok(request) => request,
            Err(_) => {
                return early(
                    step(
                        json!({"status":"method-unavailable","value_kind":kind(&value)}),
                        "requestAdapter.lookup",
                        "navigator.gpu.requestAdapter",
                    ),
                    3,
                )
            }
        },
        Err(error) => {
            return early(
                step(
                    error_value(error, "method-unavailable"),
                    "requestAdapter.lookup",
                    "navigator.gpu.requestAdapter",
                ),
                3,
            )
        }
    };
    let mut attempts = Vec::with_capacity(5);
    let mut requests = 0u32;
    for mode in [
        "no-arguments",
        "empty-options",
        "explicit-core",
        "explicit-compatibility",
        "force-fallback",
    ] {
        let options = Object::new();
        let assignment = match mode {
            "explicit-core" => Some(("featureLevel", JsValue::from_str("core"))),
            "explicit-compatibility" => Some(("featureLevel", JsValue::from_str("compatibility"))),
            "force-fallback" => Some(("forceFallbackAdapter", JsValue::TRUE)),
            _ => None,
        };
        if let Some((key, value)) = assignment {
            let error = match Reflect::set(&options, &JsValue::from_str(key), &value) {
                Ok(true) => None,
                Ok(false) => Some(json!({"status":"options-refused"})),
                Err(error) => Some(error_value(error, "options-threw")),
            };
            if let Some(error) = error {
                attempts.push(json!({"request":mode,"test_id":format!("adapter-{mode}"),
                    "result":step(error,"requestAdapter.options","navigator.gpu.requestAdapter.options")}));
                continue;
            }
        }
        requests += 1;
        let called = if mode == "no-arguments" {
            request.call0(&gpu)
        } else {
            request.call1(&gpu, &options)
        };
        let result = match called {
            Err(error) => step(
                error_value(error, "call-threw"),
                "requestAdapter.call",
                "navigator.gpu.requestAdapter",
            ),
            Ok(value) => match value.clone().dyn_into::<Promise>() {
                Err(_) => step(
                    json!({"status":"non-promise-result","value_kind":kind(&value)}),
                    "requestAdapter.return",
                    "navigator.gpu.requestAdapter.return",
                ),
                Ok(promise) => {
                    let result = match await_adapter(promise, timeout_ms).await {
                        Err(error) => error,
                        Ok(adapter) if adapter.is_null() => {
                            json!({"status":"null","value_kind":"null"})
                        }
                        Ok(adapter) if adapter.is_undefined() => {
                            json!({"status":"undefined","value_kind":"undefined"})
                        }
                        Ok(adapter) if !adapter.is_object() => {
                            json!({"status":"invalid-adapter-result","value_kind":kind(&adapter)})
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
                    };
                    step(
                        result,
                        "requestAdapter.promise.result",
                        "navigator.gpu.requestAdapter.Promise.result",
                    )
                }
            },
        };
        log::debug!(target: "etalon_browser_demo::adapter_control", "Direct {mode}: {result}");
        attempts.push(json!({"request":mode,"test_id":format!("adapter-{mode}"),"result":result}));
    }
    json!({"execution":"Rust/WASM direct browser API control","gpu_execution_tested":false,
        "gpu_present":true,"preflight_steps":3,"adapter_requests":requests,"attempts":attempts})
}
