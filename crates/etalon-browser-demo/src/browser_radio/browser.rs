use js_sys::{Array, Function, Promise, Reflect};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

pub fn error(v: JsValue) -> String {
    let name = Reflect::get(&v, &"name".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    let message = v
        .as_string()
        .or_else(|| Reflect::get(&v, &"message".into()).ok()?.as_string())
        .unwrap_or_else(|| "Browser API error".into());
    format!(
        "{}: {}",
        name.chars().take(64).collect::<String>(),
        message.chars().take(512).collect::<String>()
    )
}
pub fn get(v: &JsValue, key: &str) -> Result<JsValue, String> {
    let result = Reflect::get(v, &key.into()).map_err(error)?;
    if result.is_null() || result.is_undefined() {
        Err(format!("{key} unavailable"))
    } else {
        Ok(result)
    }
}
pub fn call(v: &JsValue, key: &str, args: &[JsValue]) -> Result<JsValue, String> {
    let f = get(v, key)?
        .dyn_into::<Function>()
        .map_err(|_| format!("{key} not callable"))?;
    let a = Array::new();
    for x in args {
        a.push(x);
    }
    f.apply(v, &a).map_err(error)
}
pub async fn invoke(v: &JsValue, key: &str, args: &[JsValue]) -> Result<JsValue, String> {
    resolve(call(v, key, args)?).await
}
pub async fn resolve(v: JsValue) -> Result<JsValue, String> {
    JsFuture::from(Promise::resolve(&v)).await.map_err(error)
}
pub fn navigator() -> Result<JsValue, String> {
    get(&js_sys::global(), "navigator")
}
pub fn text(v: &JsValue, key: &str) -> Option<String> {
    get(v, key)
        .ok()?
        .as_string()
        .map(|s| s.chars().take(256).collect())
}
pub fn boolean(v: &JsValue, key: &str) -> Option<bool> {
    get(v, key).ok()?.as_bool()
}
