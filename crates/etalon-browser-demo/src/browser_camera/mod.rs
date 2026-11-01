mod frame;
mod lifecycle;
mod metadata;

pub use frame::camera_ingest_rgba;
pub use lifecycle::{camera_start, camera_stop, camera_stream};
pub use metadata::camera_capabilities;

use js_sys::{Array, Function, Promise, Reflect};
use serde_json::{json, Value};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

const MAX_WIDTH: u32 = 640;
const MAX_HEIGHT: u32 = 480;
const SPECTRAL_STATUS: &str =
    "unknown; standard camera API does not expose IR classification or thermal radiometry";

fn response(result: Result<Value, String>) -> String {
    let mut value = match result {
        Ok(value) => value,
        Err(error) => json!({"ok":false,"error":error}),
    };
    value["source_revision"] =
        json!(option_env!("ETALON_DEMO_REVISION").unwrap_or("local-unidentified"));
    value["library_version"] = json!(env!("CARGO_PKG_VERSION"));
    value["schema"] = json!("etalon.browser-camera.v1");
    value.to_string()
}
fn error(value: JsValue) -> String {
    let name = Reflect::get(&value, &"name".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    let message = value
        .as_string()
        .or_else(|| Reflect::get(&value, &"message".into()).ok()?.as_string())
        .unwrap_or_else(|| "Camera API error".into());
    format!(
        "{}: {}",
        name.chars().take(64).collect::<String>(),
        message.chars().take(512).collect::<String>()
    )
}

fn get(value: &JsValue, name: &str) -> Result<JsValue, String> {
    let result = Reflect::get(value, &name.into()).map_err(error)?;
    if result.is_null() || result.is_undefined() {
        Err(format!("{name} unavailable"))
    } else {
        Ok(result)
    }
}

fn call(value: &JsValue, name: &str, args: &[JsValue]) -> Result<JsValue, String> {
    let method = get(value, name)?
        .dyn_into::<Function>()
        .map_err(|_| format!("{name} is not callable"))?;
    let values = Array::new();
    for arg in args {
        values.push(arg);
    }
    method.apply(value, &values).map_err(error)
}

async fn invoke(value: &JsValue, name: &str, args: &[JsValue]) -> Result<JsValue, String> {
    JsFuture::from(Promise::resolve(&call(value, name, args)?))
        .await
        .map_err(error)
}

fn media_devices() -> Result<JsValue, String> {
    get(&get(&js_sys::global(), "navigator")?, "mediaDevices")
}

fn array(value: JsValue) -> Result<Array, String> {
    if !Array::is_array(&value) {
        return Err("Camera API returned a non-array".into());
    }
    Ok(Array::from(&value))
}
