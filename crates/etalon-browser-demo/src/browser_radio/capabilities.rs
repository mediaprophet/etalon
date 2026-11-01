use super::browser::*;
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub async fn radio_capabilities() -> String {
    let mut result = json!({"ok":true,"schema":"etalon.browser-radio.v1","library_version":env!("CARGO_PKG_VERSION"),"source_revision":option_env!("ETALON_DEMO_REVISION").unwrap_or("local-unidentified"),"bluetooth":{"api_available":false,"available":null,"status":"unavailable"},"network":{"online":null,"type":null,"effective_type":null,"downlink_mbps":null,"rtt_ms":null,"save_data":null},"limitations":{"wifi_scan":false,"wifi_ssid_bssid":false,"raw_radio":false,"bluetooth_mac_address":false,"advertisement_scan":"not implemented; experimental browser API","identity":"Bluetooth device IDs are browser-scoped identifiers, not MAC addresses"},"errors":[]});
    let Ok(nav) = navigator() else {
        result["ok"] = json!(false);
        result["error"] = json!("navigator unavailable");
        return result.to_string();
    };
    result["network"]["online"] = json!(boolean(&nav, "onLine"));
    if let Ok(net) = get(&nav, "connection") {
        for (key, out) in [("type", "type"), ("effectiveType", "effective_type")] {
            result["network"][out] = json!(text(&net, key));
        }
        for (key, out) in [("downlink", "downlink_mbps"), ("rtt", "rtt_ms")] {
            result["network"][out] = json!(get(&net, key)
                .ok()
                .and_then(|v| v.as_f64())
                .filter(|v| v.is_finite() && *v >= 0.0));
        }
        result["network"]["save_data"] = json!(boolean(&net, "saveData"));
    }
    if let Ok(bt) = get(&nav, "bluetooth") {
        let exposed = get(&bt, "requestDevice").is_ok_and(|v| v.is_function());
        result["bluetooth"]["api_available"] = json!(exposed);
        result["bluetooth"]["status"] = json!(if exposed {
            "exposed; connection not tested"
        } else {
            "unavailable"
        });
        match invoke(&bt, "getAvailability", &[]).await {
            Ok(v) => result["bluetooth"]["available"] = json!(v.as_bool()),
            Err(e) => result["errors"] = Value::Array(vec![json!(e)]),
        }
    }
    result.to_string()
}
