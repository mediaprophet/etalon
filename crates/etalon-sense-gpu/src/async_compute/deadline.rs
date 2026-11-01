use crate::{GpuError, MapCompletion};
use js_sys::{Function, Reflect};
use std::{sync::Arc, time::Duration};
use wasm_bindgen::{closure::Closure, JsCast, JsValue};

pub(super) struct Deadline {
    handle: JsValue,
    clear: Function,
    global: JsValue,
    _callback: Closure<dyn FnMut()>,
}
impl Deadline {
    pub fn new(
        completion: Arc<MapCompletion>,
        generation: u64,
        timeout: Duration,
    ) -> Result<Self, GpuError> {
        let global: JsValue = js_sys::global().into();
        let lookup = |name| {
            Reflect::get(&global, &JsValue::from_str(name))
                .ok()
                .and_then(|value| value.dyn_into::<Function>().ok())
                .ok_or(GpuError::AsyncTimerUnavailable)
        };
        let set = lookup("setTimeout")?;
        let clear = lookup("clearTimeout")?;
        let callback = Closure::new(move || completion.fail(generation, GpuError::PollTimedOut));
        let millis =
            i32::try_from(timeout.as_millis().max(1)).map_err(|_| GpuError::InvalidTimeout)?;
        let handle = set
            .call2(
                &global,
                callback.as_ref(),
                &JsValue::from_f64(f64::from(millis)),
            )
            .map_err(|_| GpuError::AsyncTimerUnavailable)?;
        Ok(Self {
            handle,
            clear,
            global,
            _callback: callback,
        })
    }
}
impl Drop for Deadline {
    fn drop(&mut self) {
        let _ = self.clear.call1(&self.global, &self.handle);
    }
}
