//! Bounded browser smoke tests through Etalon's Rust-to-WebGPU backend.

#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(target_arch = "wasm32")]
mod compute;
#[cfg(target_arch = "wasm32")]
mod extensions;
#[cfg(target_arch = "wasm32")]
mod readback;
#[cfg(target_arch = "wasm32")]
mod render;

#[cfg(target_arch = "wasm32")]
pub use browser::{create_session, create_session_for_profile, Session};

#[cfg(any(target_arch = "wasm32", test))]
mod requirements;

#[cfg(target_arch = "wasm32")]
mod adapter;
#[cfg(target_arch = "wasm32")]
mod domain_agent;
#[cfg(target_arch = "wasm32")]
mod domain_nquin;
#[cfg(target_arch = "wasm32")]
mod domain_tensor;

#[cfg(target_arch = "wasm32")]
mod backend_contracts;
#[cfg(target_arch = "wasm32")]
pub use backend_contracts::inspect_adapter_contract;

#[cfg(target_arch = "wasm32")]
mod adapter_control;
#[cfg(target_arch = "wasm32")]
pub use adapter_control::adapter_acquisition_control;

#[cfg(target_arch = "wasm32")]
mod debug_log;
#[cfg(target_arch = "wasm32")]
pub use debug_log::{configure_debug_logging, debug_log_json};

#[cfg(target_arch = "wasm32")]
mod environment_info;
#[cfg(target_arch = "wasm32")]
pub use environment_info::browser_environment_json;

#[cfg(target_arch = "wasm32")]
mod adapter_diagnostic;
#[cfg(target_arch = "wasm32")]
pub use adapter_diagnostic::adapter_diagnostic_errors;

#[cfg(target_arch = "wasm32")]
mod environment_webgl;

#[cfg(target_arch = "wasm32")]
mod domain_signal;

#[cfg(target_arch = "wasm32")]
mod webgl_fallback;
#[cfg(target_arch = "wasm32")]
mod webgl_readback;
#[cfg(target_arch = "wasm32")]
pub use webgl_fallback::webgl_render_fallback;

#[cfg(target_arch = "wasm32")]
mod app_storage;
#[cfg(target_arch = "wasm32")]
pub use app_storage::{
    delete_local_report, list_local_reports, load_local_report, request_storage_persistence,
    save_local_report, storage_capabilities,
};

#[cfg(target_arch = "wasm32")]
mod browser_radio;
#[cfg(target_arch = "wasm32")]
pub use browser_radio::{radio_capabilities, BrowserBluetooth};
#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "browser_radio/value.rs"]
mod radio_value_tests;

#[cfg(target_arch = "wasm32")]
mod browser_camera;
#[cfg(target_arch = "wasm32")]
pub use browser_camera::{
    camera_capabilities, camera_ingest_rgba, camera_start, camera_stop, camera_stream,
};
