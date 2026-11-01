use super::{
    array, call, get, invoke, media_devices, response, MAX_HEIGHT, MAX_WIDTH, SPECTRAL_STATUS,
};
use js_sys::Array;
use serde_json::{json, Map, Value};
use wasm_bindgen::prelude::*;

const MAX_DEVICES: u32 = 32;
const FIELDS: &[&str] = &[
    "width",
    "height",
    "aspectRatio",
    "frameRate",
    "facingMode",
    "resizeMode",
    "deviceId",
    "groupId",
    "torch",
    "zoom",
    "focusMode",
    "focusDistance",
    "exposureMode",
    "exposureTime",
    "exposureCompensation",
    "whiteBalanceMode",
    "colorTemperature",
    "brightness",
    "contrast",
    "saturation",
    "sharpness",
];

fn scalar(value: &JsValue) -> Value {
    if let Some(v) = value.as_bool() {
        return json!(v);
    }
    if let Some(v) = value.as_f64().filter(|n| n.is_finite()) {
        return json!(v);
    }
    if let Some(v) = value.as_string() {
        return json!(v.chars().take(512).collect::<String>());
    }
    Value::Null
}

fn bounded_value(value: JsValue) -> Value {
    if Array::is_array(&value) {
        let array = Array::from(&value);
        return json!({"values": (0..array.length().min(16)).map(|i| scalar(&array.get(i))).collect::<Vec<_>>(), "truncated":array.length()>16});
    }
    if value.is_object() {
        let mut range = Map::new();
        for key in ["min", "max", "step"] {
            if let Ok(v) = get(&value, key) {
                range.insert(key.into(), scalar(&v));
            }
        }
        return Value::Object(range);
    }
    scalar(&value)
}

pub(super) fn snapshot(track: &JsValue, method: &str) -> Value {
    match call(track, method, &[]) {
        Ok(raw) => {
            let mut result = Map::new();
            for key in FIELDS {
                if let Ok(v) = get(&raw, key) {
                    result.insert((*key).into(), bounded_value(v));
                }
            }
            json!({"status":"reported","fields":result})
        }
        Err(error) => json!({"status":"unavailable","error":error}),
    }
}

pub(super) fn track_info(track: &JsValue) -> Value {
    json!({"label":get(track,"label").ok().map(|v| scalar(&v)),
        "ready_state":get(track,"readyState").ok().map(|v| scalar(&v)),
        "muted":get(track,"muted").ok().map(|v| scalar(&v)),
        "settings":snapshot(track,"getSettings"), "capabilities":snapshot(track,"getCapabilities"),
        "spectral_classification":SPECTRAL_STATUS,"thermal_radiometry_verified":false})
}

#[wasm_bindgen]
pub async fn camera_capabilities() -> String {
    response(async {
        let devices = media_devices()?;
        let entries = array(invoke(&devices,"enumerateDevices",&[]).await?)?;
        let mut cameras = Vec::new();
        for index in 0..entries.length().min(MAX_DEVICES) {
            let entry = entries.get(index);
            if get(&entry,"kind")?.as_string().as_deref() != Some("videoinput") { continue; }
            cameras.push(json!({"device_id":get(&entry,"deviceId").ok().map(|v| scalar(&v)),
                "label":get(&entry,"label").ok().map(|v| scalar(&v)),
                "group_id":get(&entry,"groupId").ok().map(|v| scalar(&v)),
                "spectral_classification":SPECTRAL_STATUS}));
        }
        Ok(json!({"ok":true,"schema":"etalon.camera-capabilities.v1","cameras":cameras,
            "enumeration_truncated":entries.length()>MAX_DEVICES,"max_enumerated_entries":MAX_DEVICES,
            "supported_constraints":snapshot(&devices,"getSupportedConstraints"),
            "permission_requested":false,"label_visibility":"Labels and device IDs may remain hidden until permission is granted",
            "frame_limits":{"max_width":MAX_WIDTH,"max_height":MAX_HEIGHT,"format":"RGBA8","max_bytes":MAX_WIDTH*MAX_HEIGHT*4},
            "spectral_classification":SPECTRAL_STATUS,"thermal_radiometry_verified":false}))
    }.await)
}
