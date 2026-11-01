use js_sys::{Array, Function, Promise, Reflect};
use serde_json::{json, Value};
use wasm_bindgen::{prelude::wasm_bindgen, JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

const MAX_STRING_BYTES: usize = 512;
const MAX_ARRAY_ITEMS: usize = 8;
const HIGH_ENTROPY_KEYS: [&str; 7] = [
    "architecture",
    "bitness",
    "model",
    "platformVersion",
    "fullVersionList",
    "uaFullVersion",
    "wow64",
];

fn bounded(text: &str) -> String {
    let mut end = text.len().min(MAX_STRING_BYTES);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_owned()
}

fn status(name: &str) -> Value {
    json!({"status": name})
}

fn exception(error: &JsValue) -> Value {
    let message = error.as_string().or_else(|| {
        Reflect::get(error, &JsValue::from_str("message"))
            .ok()?
            .as_string()
    });
    let name = Reflect::get(error, &JsValue::from_str("name"))
        .ok()
        .and_then(|v| v.as_string());
    json!({"name": name.map(|v| bounded(&v)), "message": message.map(|v| bounded(&v))})
}

fn get(parent: &JsValue, key: &str) -> Result<JsValue, Value> {
    match Reflect::get(parent, &JsValue::from_str(key)) {
        Ok(value) if value.is_undefined() || value.is_null() => Err(status("unavailable")),
        Ok(value) => Ok(value),
        Err(error) => Err(json!({"status": "read-error", "error": exception(&error)})),
    }
}

fn text(value: JsValue) -> Value {
    match value.as_string() {
        Some(value) => {
            json!({"status": if value.is_empty() { "empty" } else { "available" }, "value": bounded(&value),
            "truncated": value.len() > MAX_STRING_BYTES})
        }
        None => status("unexpected-type"),
    }
}

fn number(value: JsValue) -> Value {
    match value.as_f64() {
        Some(value) if value.is_finite() => json!({"status": "available", "value": value}),
        Some(_) => status("non-finite"),
        None => status("unexpected-type"),
    }
}

fn boolean(value: JsValue) -> Value {
    match value.as_bool() {
        Some(value) => json!({"status": "available", "value": value}),
        None => status("unexpected-type"),
    }
}

fn property(parent: &JsValue, key: &str, convert: fn(JsValue) -> Value) -> Value {
    get(parent, key).map(convert).unwrap_or_else(|error| error)
}

fn brand(value: JsValue) -> Value {
    if !value.is_object() {
        return status("unexpected-type");
    }
    json!({"status": "available", "brand": property(&value, "brand", text),
        "version": property(&value, "version", text)})
}

fn array(parent: &JsValue, key: &str, convert: fn(JsValue) -> Value) -> Value {
    let value = match get(parent, key) {
        Ok(value) => value,
        Err(error) => return error,
    };
    if !Array::is_array(&value) {
        return status("unexpected-type");
    }
    let length = match get(&value, "length").ok().and_then(|v| v.as_f64()) {
        Some(length) if length.is_finite() && length >= 0.0 => length,
        _ => return status("read-error"),
    };
    let values: Vec<Value> = (0..(length.min(MAX_ARRAY_ITEMS as f64) as usize))
        .map(|i| property(&value, &i.to_string(), convert))
        .collect();
    json!({"status": "available", "values": values, "truncated": length > MAX_ARRAY_ITEMS as f64})
}

async fn user_agent_data(navigator: &JsValue) -> Value {
    let data = match get(navigator, "userAgentData") {
        Ok(data) => data,
        Err(error) => return error,
    };
    if !data.is_object() {
        return status("unexpected-type");
    }
    let low_entropy = json!({"brands": array(&data, "brands", brand),
        "mobile": property(&data, "mobile", boolean), "platform": property(&data, "platform", text)});
    let high_entropy = high_entropy(&data).await;
    json!({"status": "available", "low_entropy": low_entropy, "high_entropy": high_entropy})
}

async fn high_entropy(data: &JsValue) -> Value {
    let method = match get(data, "getHighEntropyValues") {
        Ok(method) => method,
        Err(error) => return error,
    };
    let method = match method.dyn_into::<Function>() {
        Ok(method) => method,
        Err(_) => return status("unexpected-type"),
    };
    let requested = Array::new();
    for key in HIGH_ENTROPY_KEYS {
        requested.push(&JsValue::from_str(key));
    }
    let result = match method.call1(data, requested.as_ref()) {
        Ok(result) => result,
        Err(error) => return json!({"status": "request-error", "error": exception(&error)}),
    };
    let promise = match result.dyn_into::<Promise>() {
        Ok(promise) => promise,
        Err(_) => return status("unexpected-type"),
    };
    let result = match JsFuture::from(promise).await {
        Ok(result) => result,
        Err(error) => return json!({"status": "rejected", "error": exception(&error)}),
    };
    if !result.is_object() {
        return status("unexpected-type");
    }
    json!({"status": "available", "requested_keys": HIGH_ENTROPY_KEYS,
        "architecture": property(&result, "architecture", text),
        "bitness": property(&result, "bitness", text),
        "model": property(&result, "model", text),
        "platform_version": property(&result, "platformVersion", text),
        "full_version_list": array(&result, "fullVersionList", brand),
        "ua_full_version": property(&result, "uaFullVersion", text),
        "wow64": property(&result, "wow64", boolean)})
}

fn global_type(global: &JsValue) -> Value {
    let result = (|| {
        let object = get(global, "Object")?;
        let prototype = get(&object, "prototype")?;
        let method = get(&prototype, "toString")?
            .dyn_into::<Function>()
            .map_err(|_| status("unexpected-type"))?;
        method
            .call0(global)
            .map_err(|error| json!({"status": "read-error", "error": exception(&error)}))
    })();
    result.map(text).unwrap_or_else(|error| error)
}

fn screen(global: &JsValue) -> Value {
    let screen = match get(global, "screen") {
        Ok(screen) => screen,
        Err(error) => return error,
    };
    json!({"status": "available", "width": property(&screen, "width", number),
        "height": property(&screen, "height", number), "available_width": property(&screen, "availWidth", number),
        "available_height": property(&screen, "availHeight", number), "color_depth": property(&screen, "colorDepth", number),
        "pixel_depth": property(&screen, "pixelDepth", number)})
}

/// Read bounded browser-exposed environment metadata. Missing, empty and rejected values are explicit.
/// The caller may impose a deadline on the optional high-entropy user-agent request.
#[wasm_bindgen]
pub async fn browser_environment_json() -> String {
    let global: JsValue = js_sys::global().into();
    let navigator = match get(&global, "navigator") {
        Ok(navigator) => json!({"status": "available",
            "user_agent": property(&navigator, "userAgent", text),
            "platform": property(&navigator, "platform", text),
            "vendor": property(&navigator, "vendor", text),
            "language": property(&navigator, "language", text),
            "languages": array(&navigator, "languages", text),
            "hardware_concurrency": property(&navigator, "hardwareConcurrency", number),
            "device_memory": property(&navigator, "deviceMemory", number),
            "max_touch_points": property(&navigator, "maxTouchPoints", number),
            "user_agent_data": user_agent_data(&navigator).await}),
        Err(error) => error,
    };
    json!({"schema_version": 1, "navigator": navigator,
        "secure_context": property(&global, "isSecureContext", boolean),
        "cross_origin_isolated": property(&global, "crossOriginIsolated", boolean),
        "global_type": global_type(&global), "device_pixel_ratio": property(&global, "devicePixelRatio", number),
        "screen": screen(&global), "bounds": {"string_bytes": MAX_STRING_BYTES, "array_items": MAX_ARRAY_ITEMS}}).to_string()
}
