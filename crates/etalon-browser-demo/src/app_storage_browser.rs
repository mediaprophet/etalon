use js_sys::{Array, Function, Object, Promise, Reflect};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

pub(super) fn error(value: JsValue) -> String {
    let name = Reflect::get(&value, &"name".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    let name = name.chars().take(64).collect::<String>();
    let message = value
        .as_string()
        .or_else(|| Reflect::get(&value, &"message".into()).ok()?.as_string())
        .unwrap_or_else(|| "Browser API error".into());
    format!("{name}: {}", message.chars().take(512).collect::<String>())
}

pub(super) fn get(object: &JsValue, name: &str) -> Result<JsValue, String> {
    let value = Reflect::get(object, &name.into()).map_err(error)?;
    if value.is_null() || value.is_undefined() {
        Err(format!("{name} unavailable"))
    } else {
        Ok(value)
    }
}

pub(super) fn call(object: &JsValue, name: &str, args: &[JsValue]) -> Result<JsValue, String> {
    let function = get(object, name)?
        .dyn_into::<Function>()
        .map_err(|_| format!("{name} is not callable"))?;
    let array = Array::new();
    for arg in args {
        array.push(arg);
    }
    function.apply(object, &array).map_err(error)
}

pub(super) async fn invoke(
    object: &JsValue,
    name: &str,
    args: &[JsValue],
) -> Result<JsValue, String> {
    JsFuture::from(Promise::resolve(&call(object, name, args)?))
        .await
        .map_err(error)
}

pub(super) fn navigator() -> Result<JsValue, String> {
    get(&js_sys::global(), "navigator")
}
pub(super) fn storage() -> Result<JsValue, String> {
    get(&navigator()?, "storage")
}

pub(super) fn options(create: bool) -> JsValue {
    let result = Object::new();
    Reflect::set(&result, &"create".into(), &create.into()).expect("new object accepts property");
    result.into()
}

pub(super) async fn vault(create: bool) -> Result<Option<JsValue>, String> {
    let root = invoke(&storage()?, "getDirectory", &[]).await?;
    match invoke(
        &root,
        "getDirectoryHandle",
        &["etalon-receipts-v1".into(), options(create)],
    )
    .await
    {
        Ok(directory) => Ok(Some(directory)),
        Err(e) if !create && e.starts_with("NotFoundError:") => Ok(None),
        Err(e) => Err(e),
    }
}

pub(super) async fn file(directory: &JsValue, id: &str, create: bool) -> Result<JsValue, String> {
    invoke(directory, "getFileHandle", &[id.into(), options(create)]).await
}

pub(super) async fn file_size(handle: &JsValue) -> Result<usize, String> {
    let file = invoke(handle, "getFile", &[]).await?;
    size(&file)
}

fn size(file: &JsValue) -> Result<usize, String> {
    let size = get(file, "size")?.as_f64().ok_or("Invalid file size")?;
    if !size.is_finite()
        || size < 0.0
        || size.fract() != 0.0
        || size > super::MAX_REPORT_BYTES as f64
    {
        return Err("Receipt exceeds 256 KiB limit or has invalid size".into());
    }
    Ok(size as usize)
}

pub(super) async fn read(handle: &JsValue) -> Result<String, String> {
    let file = invoke(handle, "getFile", &[]).await?;
    size(&file)?;
    let encoded = invoke(&file, "text", &[])
        .await?
        .as_string()
        .ok_or("File.text returned non-text")?;
    if encoded.len() > super::MAX_REPORT_BYTES {
        return Err("Receipt exceeds 256 KiB limit".into());
    }
    Ok(encoded)
}

pub(super) async fn remove(directory: &JsValue, id: &str) -> Result<(), String> {
    super::validate_id(id)?;
    invoke(directory, "removeEntry", &[id.into()]).await?;
    Ok(())
}
