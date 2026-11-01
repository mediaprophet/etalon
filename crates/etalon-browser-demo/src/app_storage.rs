use js_sys::{Object, Promise, Reflect};
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{future_to_promise, JsFuture};

#[path = "app_storage_browser.rs"]
mod browser;
use browser::{call, file, file_size, get, invoke, navigator, read, remove, storage, vault};

const MAX_REPORT_BYTES: usize = 256 * 1024;
const MAX_REPORTS: usize = 16;
const MAX_TOTAL_BYTES: usize = 4 * 1024 * 1024;

fn response(result: Result<Value, String>) -> String {
    match result {
        Ok(value) => value,
        Err(error) => json!({"ok":false,"error":error}),
    }
    .to_string()
}

fn validate_id(id: &str) -> Result<(), String> {
    let Some(uuid) = id.strip_suffix(".json") else {
        return Err("Invalid receipt ID".into());
    };
    if uuid.len() != 36
        || !uuid.bytes().enumerate().all(|(i, b)| {
            if [8, 13, 18, 23].contains(&i) {
                b == b'-'
            } else {
                b.is_ascii_digit() || (b'a'..=b'f').contains(&b)
            }
        })
    {
        return Err("Invalid receipt ID".into());
    }
    Ok(())
}

fn validate_report(encoded: &str) -> Result<Value, String> {
    if encoded.len() > MAX_REPORT_BYTES {
        return Err("Receipt exceeds 256 KiB limit".into());
    }
    let report: Value = serde_json::from_str(encoded).map_err(|_| "Invalid receipt JSON")?;
    if !matches!(
        report["schema"].as_str(),
        Some("etalon.browser-suite.v1" | "etalon.adapter-diagnostic.v1")
    ) {
        return Err("Unsupported receipt schema".into());
    }
    let revision = report["source_revision"]
        .as_str()
        .ok_or("Receipt is missing source_revision")?;
    if revision.len() != 40 || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("Receipt source_revision must contain 40 hexadecimal digits".into());
    }
    Ok(report)
}

#[wasm_bindgen]
pub async fn storage_capabilities() -> String {
    let mut result = json!({"ok":true,"schema":"etalon.app-storage.v1", "source_revision":option_env!("ETALON_DEMO_REVISION").unwrap_or("local-unidentified"),
        "storage_api":false,"indexed_db":false,"indexed_db_evidence":"API exposure only; no database opened","opfs":false,"opfs_evidence":"Root handle access only; saving verifies write/read separately","web_locks":false,"persistence_request_available":false,"persisted":null,"usage_bytes":null,"quota_bytes":null,
        "limits":{"max_reports":MAX_REPORTS,"max_report_bytes":MAX_REPORT_BYTES,"max_total_bytes":MAX_TOTAL_BYTES},"errors":[]});
    let mut errors = Vec::new();
    match navigator() {
        Ok(nav) => {
            result["web_locks"] =
                json!(get(&nav, "locks").and_then(|l| get(&l, "request")).is_ok());
        }
        Err(e) => errors.push(e),
    }
    match get(&js_sys::global(), "indexedDB") {
        Ok(_) => result["indexed_db"] = json!(true),
        Err(e) => errors.push(e),
    }
    match storage() {
        Ok(manager) => {
            result["storage_api"] = json!(true);
            result["persistence_request_available"] = json!(get(&manager, "persist")
                .map(|v| v.is_function())
                .unwrap_or(false));
            match invoke(&manager, "estimate", &[]).await {
                Ok(estimate) => {
                    for (source, target) in [("usage", "usage_bytes"), ("quota", "quota_bytes")] {
                        match get(&estimate, source)
                            .ok()
                            .and_then(|v| v.as_f64())
                            .filter(|n| n.is_finite() && *n >= 0.0)
                        {
                            Some(n) => result[target] = json!(n),
                            None => errors.push(format!("Storage estimate {source} unavailable")),
                        }
                    }
                }
                Err(e) => errors.push(format!("estimate: {e}")),
            }
            match invoke(&manager, "persisted", &[]).await {
                Ok(v) => match v.as_bool() {
                    Some(value) => result["persisted"] = json!(value),
                    None => errors.push("persisted returned non-boolean".into()),
                },
                Err(e) => errors.push(format!("persisted: {e}")),
            }
            match invoke(&manager, "getDirectory", &[]).await {
                Ok(_) => result["opfs"] = json!(true),
                Err(e) => errors.push(format!("OPFS root: {e}")),
            }
        }
        Err(e) => errors.push(e),
    }
    if !result["web_locks"].as_bool().unwrap_or(false) {
        errors.push(
            "Web Locks unavailable: vault operations disabled to preserve cross-tab bounds".into(),
        );
    }
    result["errors"] = json!(errors);
    result.to_string()
}

#[wasm_bindgen]
pub async fn request_storage_persistence() -> String {
    response(
        async {
            let persisted = invoke(&storage()?, "persist", &[])
                .await?
                .as_bool()
                .ok_or("persist returned non-boolean")?;
            Ok(json!({"ok":true,"persisted":persisted}))
        }
        .await,
    )
}

async fn scan(directory: &JsValue) -> Result<(Vec<Value>, usize), String> {
    let iterator = call(directory, "values", &[])?;
    let mut reports = Vec::new();
    let mut total = 0;
    loop {
        let entry = invoke(&iterator, "next", &[]).await?;
        if get(&entry, "done")?.as_bool() == Some(true) {
            break;
        }
        if reports.len() == MAX_REPORTS {
            return Err(
                "Receipt vault exceeds 16 entries; no further storage access performed".into(),
            );
        }
        let handle = get(&entry, "value")?;
        let id = get(&handle, "name")?
            .as_string()
            .ok_or("Invalid receipt filename")?;
        validate_id(&id)?;
        if get(&handle, "kind")?.as_string().as_deref() != Some("file") {
            return Err("Unexpected directory in receipt vault".into());
        }
        let bytes = file_size(&handle).await?;
        total += bytes;
        if total > MAX_TOTAL_BYTES {
            return Err("Receipt vault exceeds 4 MiB limit".into());
        }
        let mut summary = json!({"id":id,"bytes":bytes});
        match read(&handle).await.and_then(|s| validate_report(&s)) {
            Ok(report) => {
                summary["schema"] = report["schema"].clone();
                summary["source_revision"] = report["source_revision"].clone();
            }
            Err(e) => summary["validation_error"] = json!(e),
        }
        reports.push(summary);
    }
    reports.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    Ok((reports, total))
}

async fn save(encoded: &str) -> Result<Value, String> {
    validate_report(encoded)?;
    let directory = vault(true).await?.ok_or("Receipt vault unavailable")?;
    let (reports, total) = scan(&directory).await?;
    if reports.len() >= MAX_REPORTS || total + encoded.len() > MAX_TOTAL_BYTES {
        return Err(
            "Receipt vault full; explicitly delete a saved receipt before saving another".into(),
        );
    }
    let crypto = get(&js_sys::global(), "crypto")?;
    let uuid = call(&crypto, "randomUUID", &[])?
        .as_string()
        .ok_or("UUID generation failed")?;
    let id = format!("{uuid}.json");
    validate_id(&id)?;
    match file(&directory, &id, false).await {
        Ok(_) => return Err("Receipt UUID collision; existing receipt preserved".into()),
        Err(e) if e.starts_with("NotFoundError:") => {}
        Err(e) => return Err(e),
    }
    let handle = file(&directory, &id, true).await?;
    let result = async {
        let writable = invoke(&handle, "createWritable", &[]).await?;
        let written = async {
            invoke(&writable, "write", &[encoded.into()]).await?;
            invoke(&writable, "close", &[]).await?;
            Ok::<(), String>(())
        }
        .await;
        if let Err(e) = written {
            let aborted = invoke(&writable, "abort", &[]).await;
            return Err(match aborted {
                Ok(_) => e,
                Err(a) => format!("{e}; abort: {a}"),
            });
        }
        let actual = read(&handle).await?;
        validate_report(&actual)?;
        if actual != encoded {
            return Err("Receipt read-back differs from saved JSON".into());
        }
        Ok(json!({"ok":true,"id":id,"bytes":encoded.len(),"verified":true}))
    }
    .await;
    match result {
        Ok(value) => Ok(value),
        Err(e) => match remove(&directory, &id).await {
            Ok(()) => Err(e),
            Err(cleanup) => Err(format!("{e}; failed receipt {id} cleanup: {cleanup}")),
        },
    }
}

#[derive(Clone, Copy)]
enum Operation {
    Save,
    List,
    Load,
    Delete,
}

async fn operate(operation: Operation, payload: String) -> Result<Value, String> {
    if matches!(operation, Operation::Save) {
        return save(&payload).await;
    }
    if matches!(operation, Operation::Load | Operation::Delete) {
        validate_id(&payload)?;
    }
    let directory = vault(false).await?;
    if matches!(operation, Operation::List) {
        let (reports, total) = match directory {
            Some(dir) => scan(&dir).await?,
            None => (Vec::new(), 0),
        };
        return Ok(json!({"ok":true,"reports":reports,"total_bytes":total}));
    }
    let directory = directory.ok_or("Receipt vault is empty")?;
    if matches!(operation, Operation::Delete) {
        // Confirm a file handle before removing a single, validated owned filename.
        file(&directory, &payload, false).await?;
        remove(&directory, &payload).await?;
        return Ok(json!({"ok":true,"id":payload}));
    }
    let encoded = read(&file(&directory, &payload, false).await?).await?;
    validate_report(&encoded)?;
    Ok(json!({"ok":true,"id":payload,"encoded":encoded}))
}

async fn locked(operation: Operation, payload: String) -> String {
    let result = async {
        let locks = get(&navigator()?, "locks").map_err(|_| {
            "Web Locks unavailable; vault operations require cross-tab coordination"
        })?;
        let callback = Closure::<dyn FnMut(JsValue) -> Promise>::new(move |lock: JsValue| {
            if lock.is_null() || lock.is_undefined() {
                return Promise::resolve(&JsValue::from_str(&response(Err(
                    "Receipt vault busy in another operation or tab; retry when it finishes".into(),
                ))));
            }
            let payload = payload.clone();
            future_to_promise(async move {
                Ok(JsValue::from_str(&response(
                    operate(operation, payload).await,
                )))
            })
        });
        let options = Object::new();
        Reflect::set(&options, &"ifAvailable".into(), &true.into()).map_err(browser::error)?;
        let pending = call(
            &locks,
            "request",
            &[
                "etalon-receipts-v1".into(),
                options.into(),
                callback.as_ref().clone(),
            ],
        )?;
        let value = JsFuture::from(Promise::resolve(&pending))
            .await
            .map_err(browser::error)?;
        drop(callback);
        value
            .as_string()
            .ok_or_else(|| "Invalid vault operation result".to_string())
    }
    .await;
    match result {
        Ok(value) => value,
        Err(e) => response(Err(e)),
    }
}

#[wasm_bindgen]
pub async fn save_local_report(encoded: String) -> String {
    if let Err(e) = validate_report(&encoded) {
        return response(Err(e));
    }
    locked(Operation::Save, encoded).await
}
#[wasm_bindgen]
pub async fn list_local_reports() -> String {
    locked(Operation::List, String::new()).await
}
#[wasm_bindgen]
pub async fn load_local_report(id: String) -> String {
    locked(Operation::Load, id).await
}
#[wasm_bindgen]
pub async fn delete_local_report(id: String) -> String {
    locked(Operation::Delete, id).await
}
