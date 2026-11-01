//! Bounded WebGL renderer declarations; these do not establish WebGPU availability or driver identity.

use js_sys::{Array, Function, Reflect};
use serde_json::{json, Value};
use wasm_bindgen::{JsCast, JsValue};

use crate::environment_info::{bounded, exception};

fn error(stage: &str, value: &JsValue) -> Value {
    json!({"status":"error","stage":stage,"error":exception(value)})
}

fn missing(stage: &str) -> Value {
    json!({"status":"unavailable","stage":stage})
}

fn property(owner: &JsValue, key: &str, stage: &str) -> Result<JsValue, Value> {
    Reflect::get(owner, &JsValue::from_str(key)).map_err(|value| error(stage, &value))
}

fn method(owner: &JsValue, key: &str, stage: &str) -> Result<Function, Value> {
    let value = property(owner, key, stage)?;
    if value.is_null() || value.is_undefined() {
        return Err(missing(stage));
    }
    value
        .dyn_into::<Function>()
        .map_err(|_| json!({"status":"error","stage":stage,"reason":"unexpected-type"}))
}

fn parameter(gl: &JsValue, token: u32) -> Value {
    let result = method(gl, "getParameter", "getParameter.lookup").and_then(|get| {
        get.call1(gl, &JsValue::from_f64(f64::from(token)))
            .map_err(|value| error("getParameter.call", &value))
    });
    match result {
        Err(error) => error,
        Ok(value) if value.is_null() || value.is_undefined() => missing("getParameter.result"),
        Ok(value) => match value.as_string() {
            Some(value) => json!({"status":if value.is_empty() {"empty"} else {"available"},
                "value":bounded(&value),"truncated":value.len() > 512}),
            None => {
                json!({"status":"error","stage":"getParameter.result","reason":"unexpected-type"})
            }
        },
    }
}

fn extension(gl: &JsValue, name: &str) -> Result<JsValue, Value> {
    let get = method(gl, "getExtension", "getExtension.lookup")?;
    let value = get
        .call1(gl, &JsValue::from_str(name))
        .map_err(|value| error("getExtension.call", &value))?;
    if value.is_null() || value.is_undefined() {
        return Err(missing("getExtension.result"));
    }
    if !value.is_object() {
        return Err(
            json!({"status":"error","stage":"getExtension.result","reason":"unexpected-type"}),
        );
    }
    Ok(value)
}

struct ContextGuard {
    gl: JsValue,
    released: bool,
}
impl ContextGuard {
    fn release(&mut self) -> Value {
        self.released = true;
        let result = extension(&self.gl, "WEBGL_lose_context").and_then(|extension| {
            let lose = method(&extension, "loseContext", "loseContext.lookup")?;
            lose.call0(&extension)
                .map_err(|value| error("loseContext.call", &value))?;
            Ok(json!({"status":"requested"}))
        });
        let mut result = result.unwrap_or_else(|error| error);
        result["extension"] = json!("WEBGL_lose_context");
        result
    }
}
impl Drop for ContextGuard {
    fn drop(&mut self) {
        if !self.released {
            let _ = self.release();
        }
    }
}

fn query(gl: &JsValue) -> Value {
    let parameters = json!({"vendor":parameter(gl,0x1f00),"renderer":parameter(gl,0x1f01),
        "version":parameter(gl,0x1f02),"shading_language_version":parameter(gl,0x8b8c)});
    // Khronos WEBGL_debug_renderer_info defines tokens 0x9245 and 0x9246.
    let debug = match extension(gl, "WEBGL_debug_renderer_info") {
        Ok(_) => {
            json!({"status":"available","unmasked_vendor":parameter(gl,0x9245),"unmasked_renderer":parameter(gl,0x9246)})
        }
        Err(error) => error,
    };
    let query_error = parameters
        .as_object()
        .is_some_and(|fields| fields.values().any(|value| value["status"] == "error"))
        || debug["status"] == "error"
        || debug["unmasked_vendor"]["status"] == "error"
        || debug["unmasked_renderer"]["status"] == "error";
    json!({"status":if query_error {"error"} else {"available"},"parameters":parameters,"debug_renderer_info":debug})
}

fn collect() -> Value {
    let global: JsValue = js_sys::global().into();
    let constructor = match method(&global, "OffscreenCanvas", "OffscreenCanvas.lookup") {
        Ok(constructor) => constructor,
        Err(error) => return error,
    };
    let dimensions = Array::of2(&JsValue::from_f64(1.0), &JsValue::from_f64(1.0));
    let canvas: JsValue = match Reflect::construct(&constructor, &dimensions) {
        Ok(canvas) => canvas,
        Err(value) => return error("OffscreenCanvas.construct", &value),
    };
    let get_context = match method(&canvas, "getContext", "getContext.lookup") {
        Ok(method) => method,
        Err(error) => return error,
    };
    let mut attempts = Vec::with_capacity(2);
    for context_type in ["webgl2", "webgl"] {
        let gl = match get_context.call1(&canvas, &JsValue::from_str(context_type)) {
            Ok(value) if value.is_object() => value,
            Ok(value) => {
                attempts.push(json!({"context_type":context_type,"status":if value.is_null() || value.is_undefined() {"unavailable"} else {"error"},
                    "result":if value.is_null() {"null"} else if value.is_undefined() {"undefined"} else {"unexpected-type"}}));
                continue;
            }
            Err(value) => {
                let mut result = error("getContext.call", &value);
                result["context_type"] = json!(context_type);
                attempts.push(result);
                continue;
            }
        };
        attempts.push(json!({"context_type":context_type,"status":"available"}));
        let mut guard = ContextGuard {
            gl,
            released: false,
        };
        let mut result = query(&guard.gl);
        result["context_type"] = json!(context_type);
        result["context_attempts"] = json!(attempts);
        result["cleanup"] = guard.release();
        return result;
    }
    json!({"status":if attempts.iter().any(|value| value["status"] == "error") {"error"} else {"unavailable"},
        "stage":"getContext.result","context_attempts":attempts})
}

pub(crate) fn probe() -> Value {
    let mut result = collect();
    result["scope"] = json!("WebGL renderer declarations only");
    result["canvas"] = json!({"width":1,"height":1});
    result["webgpu_adapter_established"] = json!(false);
    result["webgpu_driver_established"] = json!(false);
    result["browser_gate_established"] = json!(false);
    result["bounds"] = json!({"max_context_attempts":2,"string_bytes":512});
    result
}
