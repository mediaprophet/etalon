use std::sync::{Arc, Mutex};

use js_sys::{Function, Promise, Reflect};
use serde_json::Value;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

use crate::browser::failure;

struct Unmap<'a>(&'a etalon::Buffer);
impl Drop for Unmap<'_> {
    fn drop(&mut self) {
        self.0.unmap();
    }
}

pub(crate) async fn read(
    device: &etalon::Device,
    buffer: &etalon::Buffer,
) -> Result<Vec<u8>, Value> {
    let state = Arc::new(Mutex::new(None));
    let callback = state.clone();
    let slice = buffer.slice(..);
    slice.map_async(etalon::MapMode::Read, move |result| {
        *callback.lock().unwrap() = Some(result);
    });
    let _unmap = Unmap(buffer);
    let started = js_sys::Date::now();
    for _ in 0..500 {
        device
            .poll(etalon::PollType::Poll)
            .map_err(|error| failure("webgl-poll-failed", &error.to_string()))?;
        if let Some(result) = state.lock().unwrap().take() {
            result.map_err(|error| failure("webgl-map-failed", &error.to_string()))?;
            return slice
                .get_mapped_range()
                .map(|view| view.to_vec())
                .map_err(|error| failure("webgl-mapped-range-failed", &error.to_string()));
        }
        if js_sys::Date::now() - started >= 5000.0 {
            break;
        }
        let global = js_sys::global();
        let set = Reflect::get(&global, &"setTimeout".into())
            .and_then(|value| value.dyn_into::<Function>())
            .map_err(|_| {
                failure(
                    "webgl-timer-unavailable",
                    "Cannot yield during GPU readback",
                )
            })?;
        let promise = Promise::new(&mut |resolve, reject| {
            if let Err(error) = set.call2(&global, &resolve, &JsValue::from_f64(10.0)) {
                let _ = reject.call1(&JsValue::UNDEFINED, &error);
            }
        });
        JsFuture::from(promise)
            .await
            .map_err(|_| failure("webgl-timer-failed", "Cannot yield during GPU readback"))?;
    }
    Err(failure(
        "webgl-map-timeout",
        "GPU readback did not complete within 5 seconds",
    ))
}
