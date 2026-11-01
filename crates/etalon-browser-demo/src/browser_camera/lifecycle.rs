use super::{array, call, get, invoke, media_devices, metadata, response, MAX_HEIGHT, MAX_WIDTH};
use serde_json::json;
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

struct OwnedStream {
    value: JsValue,
    stopped: bool,
}
impl OwnedStream {
    fn stop(&mut self) -> Vec<String> {
        let mut errors = Vec::new();
        match call(&self.value, "getTracks", &[]).and_then(array) {
            Ok(tracks) => {
                for index in 0..tracks.length() {
                    if let Err(error) = call(&tracks.get(index), "stop", &[]) {
                        errors.push(error);
                    }
                }
            }
            Err(error) => errors.push(error),
        }
        self.stopped = errors.is_empty();
        errors
    }
}
impl Drop for OwnedStream {
    fn drop(&mut self) {
        if !self.stopped {
            let _ = self.stop();
        }
    }
}

#[derive(Default)]
struct CameraState {
    stream: Option<OwnedStream>,
    epoch: u64,
    pending: bool,
}
thread_local! { static CAMERA: RefCell<CameraState> = RefCell::new(CameraState::default()); }

pub(super) fn live_stream() -> Result<JsValue, String> {
    CAMERA.with(|state| {
        let state = state.borrow();
        let stream = &state
            .stream
            .as_ref()
            .ok_or("No active camera stream")?
            .value;
        let tracks = array(call(stream, "getVideoTracks", &[])?)?;
        if tracks.length() != 1
            || get(&tracks.get(0), "readyState")?.as_string().as_deref() != Some("live")
        {
            return Err("Camera video track is not live".into());
        }
        Ok(stream.clone())
    })
}

#[wasm_bindgen]
pub fn camera_stream() -> Result<JsValue, JsValue> {
    live_stream().map_err(|e| JsValue::from_str(&e))
}

#[wasm_bindgen]
pub fn camera_stop() -> String {
    let stream = CAMERA.with(|state| {
        let mut state = state.borrow_mut();
        state.epoch = state.epoch.wrapping_add(1);
        state.pending = false;
        state.stream.take()
    });
    let errors = stream.map(|mut stream| stream.stop()).unwrap_or_default();
    response(if errors.is_empty() {
        Ok(json!({"ok":true,"stopped":true,"pending_acquisition_cancelled":true}))
    } else {
        Err(format!("Camera cleanup failed: {}", errors.join("; ")))
    })
}

#[wasm_bindgen]
pub async fn camera_start(device_id: String, facing_mode: String) -> String {
    response(async {
        if device_id.len()>512 || !matches!(facing_mode.as_str(), ""|"user"|"environment"|"left"|"right") {
            return Err("Invalid camera device ID or facing mode".into());
        }
        let epoch = CAMERA.with(|state| {
            let mut state = state.borrow_mut();
            if state.pending || state.stream.is_some() { return Err("Stop the current camera request or stream before starting another".to_string()); }
            state.pending=true;
            Ok(state.epoch)
        })?;
        let acquired = async {
            let devices=media_devices()?;
            let mut video=json!({"width":{"ideal":MAX_WIDTH,"max":MAX_WIDTH},"height":{"ideal":MAX_HEIGHT,"max":MAX_HEIGHT},"frameRate":{"ideal":15,"max":30}});
            if !device_id.is_empty() { video["deviceId"]=json!({"exact":device_id}); }
            else if !facing_mode.is_empty() { video["facingMode"]=json!({"ideal":facing_mode}); }
            let constraints=js_sys::JSON::parse(&json!({"video":video,"audio":false}).to_string()).map_err(super::error)?;
            let value=invoke(&devices,"getUserMedia",&[constraints]).await?;
            Ok::<_,String>(OwnedStream { value, stopped:false })
        }.await;
        let current=CAMERA.with(|state| {
            let mut state=state.borrow_mut();
            if state.epoch==epoch { state.pending=false; true } else { false }
        });
        let mut stream=acquired?;
        if !current {
            let errors=stream.stop();
            return Err(format!("Camera acquisition cancelled; late stream stopped{}", if errors.is_empty(){String::new()}else{format!("; cleanup errors: {}",errors.join("; "))}));
        }
        let tracks=array(call(&stream.value,"getVideoTracks",&[])?)?;
        if tracks.length()!=1 { return Err("Camera returned an unexpected video track count".into()); }
        if get(&tracks.get(0),"readyState")?.as_string().as_deref()!=Some("live") { return Err("Camera returned an ended video track".into()); }
        let info=metadata::track_info(&tracks.get(0));
        CAMERA.with(|state| state.borrow_mut().stream=Some(stream));
        Ok(json!({"ok":true,"schema":"etalon.camera-session.v1","track":info,"audio_requested":false,"automatic_upload":false,"automatic_save":false}))
    }.await)
}
